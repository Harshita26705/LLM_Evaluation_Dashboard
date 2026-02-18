import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr
import io
import math
import re
import ast
import nltk
from collections import Counter
from PIL import Image
import numpy as np
from typing import Tuple

# Download required NLTK data silently
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.tokenize import TreebankWordTokenizer
from sentence_transformers import SentenceTransformer, util
from detoxify import Detoxify

# --- Setup with graceful fallbacks ---
print("⏳ Loading models...")

try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("   ✅ Loaded sentence embedder")
except Exception as e:
    print(f"   ⚠️ Could not load embedder: {e}")
    embedder = None

try:
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    print("   ✅ Loaded spacy model")
except Exception as e:
    print(f"   ⚠️ Spacy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

try:
    tokenizer = TreebankWordTokenizer()
    print("   ✅ Loaded tokenizer")
except Exception as e:
    print(f"   ⚠️ Could not load tokenizer: {e}")
    tokenizer = None

try:
    toxicity_model = Detoxify("original")
    print("   ✅ Loaded toxicity model")
except Exception as e:
    print(f"   ⚠️ Could not load toxicity model: {e}")
    toxicity_model = None

print("✅ Models loaded!\n")

# --- Hyperparameters / Weights ---
LAMBDA_LEN = 0.5
EXPECTED_LEN_RATIO = 2.0
GAMMA_TOX = 5.0
GAMMA_BIAS = 4.0
ALPHA_COH = 0.6
HALL_MULT = 0.3
SAFETY_THRESHOLD = 0.30
SAFETY_CAP = 0.50

WEIGHTS = {
    "relevance": 0.25,
    "length_fit": 0.15,
    "coherence": 0.20,
    "rouge1_f1": 0.10,
    "toxicity_inv": 0.15,
    "bias_inv": 0.10,
    "hall_inv": 0.05
}

# --- Utility ---
def unigrams(text):
    if tokenizer is None:
        # Fallback: simple whitespace tokenization
        return [t.lower() for t in text.split() if t.strip()]
    return [t.lower() for t in tokenizer.tokenize(text) if t.strip()]

def rouge1_f1(reference, candidate):
    ref_unigrams = unigrams(reference)
    cand_unigrams = unigrams(candidate)
    ref_counts = Counter(ref_unigrams)
    cand_counts = Counter(cand_unigrams)
    overlap = sum(min(ref_counts[t], cand_counts[t]) for t in set(ref_counts) | set(cand_counts))
    prec = overlap / (sum(cand_counts.values()) + 1e-5)
    rec  = overlap / (sum(ref_counts.values()) + 1e-5)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def length_fit(query_text, response_text, expected=EXPECTED_LEN_RATIO, lam=LAMBDA_LEN):
    q_len = max(1, len(unigrams(query_text)))
    r_len = max(1, len(unigrams(response_text)))
    ratio = r_len / q_len
    return math.exp(-lam * abs(math.log(ratio / expected)))

def toxicity_penalty(response_text, gamma=GAMMA_TOX):
    try:
        if toxicity_model is None:
            raise Exception("Toxicity model not loaded")
        tox_prob = float(toxicity_model.predict(response_text)["toxicity"])
        penalty = 1 - math.exp(-gamma * tox_prob)
    except Exception:
        toks = unigrams(response_text)
        toxic_terms = {"stupid", "idiot", "hate", "kill", "trash", "dumb"}
        toxic_count = sum(1 for t in toks if t in toxic_terms)
        ratio = toxic_count / (len(toks) + 1e-5)
        penalty = 1 - math.exp(-gamma * ratio)
    return max(0.0, min(1.0, penalty))

def bias_penalty(response_text, gamma=GAMMA_BIAS):
    if nlp is None:
        # Fallback: simple keyword matching
        toks = unigrams(response_text)
        biased_terms = {
            "he","him","his","man","male","father","boy",
            "she","her","hers","woman","female","mother","girl",
            "white","black","asian","latino","indian","african","european"
        }
        biased_count = sum(1 for t in toks if t in biased_terms)
        ratio = biased_count / (len(toks) + 1e-5)
        return 1 - math.exp(-gamma * ratio)
    
    try:
        doc = nlp(response_text)
        toks = [t.text.lower() for t in doc]
    except Exception:
        toks = unigrams(response_text)
    
    biased_terms = {
        "he","him","his","man","male","father","boy",
        "she","her","hers","woman","female","mother","girl",
        "white","black","asian","latino","indian","african","european"
    }
    biased_count = sum(1 for t in toks if t in biased_terms)
    ratio = biased_count / (len(toks) + 1e-5)
    return 1 - math.exp(-gamma * ratio)

def relevance_score(query_text, response_text):
    q = set(unigrams(query_text))
    r = set(unigrams(response_text))
    if not q:
        return 0.0
    return len(q & r) / (len(q) + 1e-5)

UNCERTAINTY_MARKERS = {"maybe","not","unsure","uncertain","perhaps","guess","might","could"}
def coherence_score(query_text, response_text, alpha=ALPHA_COH):
    toks = unigrams(response_text)
    if not toks:
        return 0.0
    unc_density = sum(1 for t in toks if t in UNCERTAINTY_MARKERS) / len(toks)
    rel = relevance_score(query_text, response_text)
    return alpha * (1 - unc_density) + (1 - alpha) * rel

def detect_hallucination(reference, response):
    hallucinated_entities = []
    
    if nlp is None:
        # Fallback: simple string matching
        ref_toks = set(unigrams(reference))
        resp_toks = set(unigrams(response))
        missing = resp_toks - ref_toks
        f1 = len(ref_toks & resp_toks) / (len(ref_toks | resp_toks) + 1e-5)
    else:
        try:
            ref_doc = nlp(reference)
            resp_doc = nlp(response)
            ref_ents = set((ent.text.lower(), ent.label_) for ent in ref_doc.ents)
            resp_ents = set((ent.text.lower(), ent.label_) for ent in resp_doc.ents)
            overlap = ref_ents & resp_ents
            prec = len(overlap) / (len(resp_ents) + 1e-5)
            rec  = len(overlap) / (len(ref_ents) + 1e-5)
            f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
            hallucinated_entities = list(resp_ents - ref_ents)
        except Exception:
            f1 = 0.5
            hallucinated_entities = []
    
    toks = unigrams(response)
    unc_density = sum(1 for t in toks if t in UNCERTAINTY_MARKERS) / (len(toks) + 1e-5)
    hall_risk_unc = min(1.0, unc_density * HALL_MULT)
    risk = (1 - f1) * 0.7 + hall_risk_unc * 0.3
    
    return round(risk, 3), hallucinated_entities

def detect_bias(response):
    entities = []
    if nlp is not None:
        try:
            doc = nlp(response)
            entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in {"PERSON", "NORP"}]
        except Exception:
            pass
    
    pen = bias_penalty(response)
    return {
        "Bias Penalty": round(pen, 3),
        "Named Entities": entities
    }

def check_code_quality(code_str):
    """Evaluate code quality metrics with explanation and improvements"""
    results = {
        "syntax_valid": False,
        "errors": [],
        "metrics": {},
        "suggestions": [],
        "explanation": "",
        "improved_code": ""
    }
    
    if not code_str.strip():
        results["explanation"] = "No code provided."
        return results
    
    # Check syntax
    try:
        tree = ast.parse(code_str)
        results["syntax_valid"] = True
    except SyntaxError as e:
        results["errors"].append(f"Syntax Error: {str(e)}")
        results["explanation"] = f"❌ Code contains syntax errors and cannot be executed.\n\nError: {str(e)}"
        return results
    
    # Count lines and complexity
    lines = code_str.split('\n')
    non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    comment_lines = [l for l in lines if l.strip().startswith('#')]
    
    # Analyze code structure
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    
    results["metrics"] = {
        "Total Lines": len(lines),
        "Code Lines": len(non_empty_lines),
        "Comment Lines": len(comment_lines),
        "Comment Ratio": round(len(comment_lines) / (len(lines) + 1e-5), 3),
        "Functions": len(functions),
        "Classes": len(classes),
        "Imports": len(imports)
    }
    
    # Generate explanation
    explanation = "📝 Code Analysis Summary\n\n"
    
    if functions:
        explanation += f"🔧 Functions Found: {', '.join(functions[:5])}"
        if len(functions) > 5:
            explanation += f" and {len(functions) - 5} more"
        explanation += "\n"
    
    if classes:
        explanation += f"📦 Classes Found: {', '.join(classes)}\n"
    
    if imports:
        explanation += f"📚 Imports: {len(imports)} module(s) imported\n"
    
    explanation += f"\n📊 Code Structure:\n"
    explanation += f"- Total Lines: {len(lines)}\n"
    explanation += f"- Actual Code: {len(non_empty_lines)} lines\n"
    explanation += f"- Comments: {len(comment_lines)} lines ({round(len(comment_lines) / (len(lines) + 1e-5) * 100, 1)}%)\n"
    explanation += f"- Blank Lines: {len(lines) - len(non_empty_lines) - len(comment_lines)}\n\n"
    
    # Purpose analysis
    explanation += "🎯 Code Purpose:\n"
    if functions and classes:
        explanation += "This code defines both functions and classes, suggesting it's part of a larger module or library.\n"
    elif functions:
        explanation += f"This code defines {len(functions)} function(s) for specific tasks.\n"
    elif classes:
        explanation += f"This code defines {len(classes)} class(es) for object-oriented programming.\n"
    else:
        explanation += "This appears to be a script with sequential instructions.\n"
    
    results["explanation"] = explanation
    
    # Check for code smells and issues
    improvements = []
    improved_lines = lines.copy()
    
    if len(non_empty_lines) > 100:
        results["suggestions"].append("⚠️ Function is quite long (>100 lines). Consider breaking it down.")
        improvements.append("Break down large functions into smaller, reusable components")
    
    if len(comment_lines) == 0:
        results["suggestions"].append("⚠️ No comments found. Add documentation.")
        improvements.append("Add comments to explain complex logic")
    
    if code_str.count('try:') == 0 and code_str.count('except') == 0:
        results["suggestions"].append("⚠️ No error handling detected. Consider adding try-catch blocks.")
        improvements.append("Add try-except blocks for robust error handling")
    
    # Security checks
    if 'eval(' in code_str:
        results["errors"].append("🚨 Security Issue: Use of eval() detected!")
        improvements.append("Replace eval() with safer alternatives like ast.literal_eval()")
    
    if 'exec(' in code_str:
        results["errors"].append("🚨 Security Issue: Use of exec() detected!")
        improvements.append("Avoid exec() - redesign code to not execute arbitrary strings")
    
    # Best practices
    if not functions and not classes and len(non_empty_lines) > 20:
        improvements.append("Organize code into functions for better maintainability")
    
    if len(comment_lines) / (len(lines) + 1e-5) < 0.1 and len(lines) > 50:
        improvements.append("Increase comment coverage for better code documentation")
    
    # Generate improved code
    if improvements:
        improved_code = f"# IMPROVED VERSION\n"
        improved_code += f"# Improvements suggested:\n"
        for i, imp in enumerate(improvements, 1):
            improved_code += f"# {i}. {imp}\n"
        improved_code += f"\n"
        
        # Add missing docstring if functions exist
        if functions and '"""' not in code_str and "'''" not in code_str:
            improved_code += '"""\nModule for [describe purpose].\n"""\n\n'
        
        # Add error handling wrapper if missing
        if 'try:' not in code_str and len(functions) > 0:
            improved_code += "# Add error handling:\n"
            improved_code += "try:\n"
            improved_code += "    " + code_str.replace("\n", "\n    ")
            improved_code += "\nexcept Exception as e:\n"
            improved_code += "    print(f'Error: {e}')\n"
            improved_code += "    # Handle error appropriately\n"
        else:
            improved_code += code_str
        
        results["improved_code"] = improved_code
    else:
        results["improved_code"] = code_str
        results["suggestions"].append("✅ Code quality is good!")
    
    if not results["suggestions"]:
        results["suggestions"].append("✅ Code looks good!")
    
    return results

def composite_score(metrics, weights=WEIGHTS):
    contrib = {
        "relevance": metrics["relevance"],
        "length_fit": metrics["length_fit"],
        "coherence": metrics["coherence"],
        "rouge1_f1": metrics["rouge1_f1"],
        "toxicity_inv": 1 - metrics["toxicity_penalty"],
        "bias_inv": 1 - metrics["bias_penalty"],
        "hall_inv": 1 - metrics["hallucination_risk"]
    }
    raw = sum(weights[k] * contrib[k] for k in contrib.keys()) / sum(weights.values())
    if metrics["toxicity_penalty"] > SAFETY_THRESHOLD:
        return min(raw, SAFETY_CAP)
    return raw

def evaluate(reference, responses_dict):
    if embedder is None:
        gr.Warning("⚠️ Embedder model not loaded. Download with: python -m sentence_transformers download all-MiniLM-L6-v2")
        return {}
    
    results = {}
    ref_emb = embedder.encode(reference, convert_to_tensor=True)
    for model_name, response in responses_dict.items():
        if not response.strip():
            continue
        resp_emb = embedder.encode(response, convert_to_tensor=True)
        cosine_sim = float(util.pytorch_cos_sim(ref_emb, resp_emb).item())
        r1 = rouge1_f1(reference, response)
        len_fit = length_fit(reference, response)
        tox_pen = toxicity_penalty(response)
        hall_risk, hallucinated_entities = detect_hallucination(reference, response)
        bias_info = detect_bias(response)
        bias_pen = bias_info["Bias Penalty"]
        rel = relevance_score(reference, response)
        coh = coherence_score(reference, response)
        final = composite_score({
            "relevance": rel,
            "length_fit": len_fit,
            "coherence": coh,
            "rouge1_f1": r1,
            "toxicity_penalty": tox_pen,
            "bias_penalty": bias_pen,
            "hallucination_risk": hall_risk
        })
        results[model_name] = {
            "Semantic Similarity": round(cosine_sim, 3),
            "ROUGE-1 F1": round(r1, 3),
            "Length Fit": round(len_fit, 3),
            "Relevance": round(rel, 3),
            "Coherence": round(coh, 3),
            "Toxicity Penalty": round(tox_pen, 3),
            "Bias Penalty": round(bias_pen, 3),
            "Hallucination Risk": round(hall_risk, 3),
            "Final Score": round(final, 3),
            "Hallucinated Entities": hallucinated_entities
        }
    return results

def plot_results(results):
    if not results:
        return None
    plot_keys = ["Semantic Similarity","ROUGE-1 F1","Length Fit","Relevance","Coherence",
                 "Toxicity Penalty","Bias Penalty","Hallucination Risk","Final Score"]
    df = pd.DataFrame(results).T[plot_keys]
    fig, ax = plt.subplots(figsize=(12, 5))
    df.plot(kind="bar", ax=ax)
    plt.title("Model Evaluation Metrics Comparison")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("evaluation_plot.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    return "evaluation_plot.png", df

def evaluate_dataset(file):
    if embedder is None:
        raise ValueError("❌ Embedder model not loaded. Cannot process embeddings. Try restarting the app or ensure sentence-transformers is properly installed.")
    
    df = pd.read_csv(file.name) if file.name.endswith(".csv") else pd.read_json(file.name)
    df.columns = [c.strip() for c in df.columns]
    required_cols = {"question", "model_name", "response"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")
    has_ref = "reference" in df.columns
    grouped = {}
    ref_emb_cache = {}
    for _, row in df.iterrows():
        ref = row["reference"].strip() if has_ref and isinstance(row["reference"], str) and row["reference"].strip() else row["question"]
        model = row["model_name"]
        resp = row["response"]
        if ref not in ref_emb_cache:
            ref_emb_cache[ref] = embedder.encode(ref, convert_to_tensor=True)
        ref_emb = ref_emb_cache[ref]
        resp_emb = embedder.encode(resp, convert_to_tensor=True)
        cosine_sim = float(util.pytorch_cos_sim(ref_emb, resp_emb).item())
        r1 = rouge1_f1(ref, resp)
        len_fit = length_fit(ref, resp)
        tox_pen = toxicity_penalty(resp)
        hall_risk, _ = detect_hallucination(ref, resp)
        bias_info = detect_bias(resp)
        bias_pen = bias_info["Bias Penalty"]
        rel = relevance_score(ref, resp)
        coh = coherence_score(ref, resp)
        final = composite_score({
            "relevance": rel,
            "length_fit": len_fit,
            "coherence": coh,
            "rouge1_f1": r1,
            "toxicity_penalty": tox_pen,
            "bias_penalty": bias_pen,
            "hallucination_risk": hall_risk
        })
        grouped.setdefault(model, {k: [] for k in [
            "Semantic Similarity","ROUGE-1 F1","Length Fit","Relevance","Coherence",
            "Toxicity Penalty","Bias Penalty","Hallucination Risk","Final Score"
        ]})
        grouped[model]["Semantic Similarity"].append(round(cosine_sim, 3))
        grouped[model]["ROUGE-1 F1"].append(round(r1, 3))
        grouped[model]["Length Fit"].append(round(len_fit, 3))
        grouped[model]["Relevance"].append(round(rel, 3))
        grouped[model]["Coherence"].append(round(coh, 3))
        grouped[model]["Toxicity Penalty"].append(round(tox_pen, 3))
        grouped[model]["Bias Penalty"].append(round(bias_pen, 3))
        grouped[model]["Hallucination Risk"].append(round(hall_risk, 3))
        grouped[model]["Final Score"].append(round(final, 3))
    
    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else None
    
    avg_results = {model: {k: avg(v) for k, v in metrics.items()} for model, metrics in grouped.items()}
    return avg_results

def export_csv(results):
    if not results:
        return None
    buf = io.BytesIO()
    pd.DataFrame(results).to_csv(buf)
    buf.seek(0)
    return buf

def analyze_image_properties(image):
    """Analyze basic image properties for multimodal evaluation"""
    if image is None:
        return {}
    
    img = Image.open(image) if isinstance(image, str) else image
    img_array = np.array(img)
    
    properties = {
        "Format": img.format or "Unknown",
        "Size (pixels)": f"{img.width}x{img.height}",
        "Mode": img.mode,
    }
    
    if len(img_array.shape) == 3:
        avg_color = img_array.mean(axis=(0, 1))
        properties["Avg Color (RGB)"] = f"({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})"
    
    return properties

def multimodal_evaluate(image, prompt_text, description_text=None):
    """Evaluate AI-generated image against the prompt used to create it"""
    if image is None or not prompt_text.strip():
        return None, None, {}, "Please provide both an image and the prompt used to generate it."
    
    try:
        image_props = analyze_image_properties(image)
        
        # If no description provided, we can only evaluate the prompt
        if not description_text or not description_text.strip():
            explanation = "⚠️ No image description provided for full evaluation.\n\n"
            explanation += f"📋 Prompt Analysis:\n"
            explanation += f"- Prompt Length: {len(unigrams(prompt_text))} words\n"
            explanation += f"- Image Properties: {image_props}\n\n"
            explanation += "💡 Tip: Describe what you see in the generated image for a complete accuracy evaluation."
            
            return {}, image_props, {}, explanation
        
        # Full evaluation: Compare description vs prompt
        description_length = len(unigrams(description_text))
        prompt_length = len(unigrams(prompt_text))
        
        # Semantic similarity between prompt and what was actually generated
        prompt_match_score = None
        if embedder:
            prompt_emb = embedder.encode(prompt_text, convert_to_tensor=True)
            desc_emb = embedder.encode(description_text, convert_to_tensor=True)
            prompt_match_score = float(util.pytorch_cos_sim(prompt_emb, desc_emb).item())
        
        # ROUGE score - how much of the prompt appears in the description
        rouge = rouge1_f1(prompt_text, description_text)
        
        # Relevance - keyword overlap
        relevance = relevance_score(prompt_text, description_text)
        
        # Coherence of the description
        coherence = coherence_score(prompt_text, description_text)
        
        # Safety check
        toxicity = toxicity_penalty(description_text)
        
        # Hallucination check - did the AI add things not in the prompt?
        hall_risk, hallucinated = detect_hallucination(prompt_text, description_text)
        
        # Overall accuracy score
        accuracy_score = (
            (prompt_match_score or 0) * 0.35 +
            rouge * 0.25 +
            relevance * 0.20 +
            coherence * 0.15 +
            (1 - hall_risk) * 0.05
        )
        
        eval_results = {
            "Prompt-Image Match Score": round(prompt_match_score, 3) if prompt_match_score else "N/A",
            "Keyword Overlap (ROUGE-1)": round(rouge, 3),
            "Relevance to Prompt": round(relevance, 3),
            "Description Coherence": round(coherence, 3),
            "Hallucination Risk": round(hall_risk, 3),
            "Overall Accuracy": round(accuracy_score, 3),
            "Safety Score": round(1 - toxicity, 3)
        }
        
        explanation = f"🎨 AI Image Generation Evaluation\n\n"
        explanation += f"📊 Accuracy Analysis:\n"
        explanation += f"✅ Overall Accuracy: {round(accuracy_score * 100, 1)}%\n\n"
        
        if prompt_match_score:
            explanation += f"🎯 Prompt Match: {round(prompt_match_score * 100, 1)}% "
            explanation += f"({'Excellent' if prompt_match_score > 0.8 else 'Good' if prompt_match_score > 0.6 else 'Fair' if prompt_match_score > 0.4 else 'Poor'})\n"
        
        explanation += f"🔤 Keyword Coverage: {round(rouge * 100, 1)}%\n"
        explanation += f"🎲 Relevance: {round(relevance * 100, 1)}%\n"
        explanation += f"🧠 Coherence: {round(coherence * 100, 1)}%\n\n"
        
        if hall_risk > 0.5:
            explanation += f"⚠️ High Hallucination Risk: The AI may have added elements not in your prompt\n"
            if hallucinated:
                explanation += f"   Found {len(hallucinated)} potential hallucinated elements\n"
        elif hall_risk > 0.3:
            explanation += f"⚡ Moderate Hallucination: Some creative interpretation detected\n"
        else:
            explanation += f"✅ Low Hallucination: Image closely follows prompt\n"
        
        explanation += f"\n🛡️ Safety: {'✅ Safe' if toxicity < 0.3 else '⚠️ Contains concerning content'}\n"
        explanation += f"\n📐 Image Properties: {image_props['Size (pixels)']} | {image_props['Mode']}\n"
        
        # Provide actionable feedback
        explanation += f"\n💡 Recommendations:\n"
        if accuracy_score > 0.8:
            explanation += "- Excellent match! The AI accurately interpreted your prompt.\n"
        elif accuracy_score > 0.6:
            explanation += "- Good match with minor deviations from the prompt.\n"
        elif accuracy_score > 0.4:
            explanation += "- Moderate match. Consider refining your prompt for better results.\n"
        else:
            explanation += "- Low match. The AI may have misunderstood the prompt. Try being more specific.\n"
        
        if hall_risk > 0.4:
            explanation += "- The AI added unexpected elements. Specify what NOT to include in prompts.\n"
        
        return eval_results, image_props, eval_results, explanation
    
    except Exception as e:
        return None, {}, {}, f"❌ Error analyzing multimodal content: {str(e)}"

# --- Custom Futuristic Blue AI Theme ---
# Use Gradio's built-in dark blue theme customized for AI
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
)

# --- Gradio UI ---
futuristic_css = """
.gradio-container {
    background: linear-gradient(135deg, #0a0e27 0%, #151b3d 100%) !important;
    color: #e3f2fd !important;
}
.gr-box {
    background-color: #151b3d !important;
    border-color: #0066ff !important;
}
.gr-input {
    background-color: #1a2847 !important;
    border-color: #0066ff !important;
    color: #e3f2fd !important;
}
.gr-button-primary {
    background: linear-gradient(90deg, #0066ff 0%, #00bcd4 100%) !important;
    border-color: #0066ff !important;
}
.gr-button-primary:hover {
    background: linear-gradient(90deg, #0052cc 0%, #0099b3 100%) !important;
}
h1, h2, h3 {
    background: linear-gradient(90deg, #42a5f5 0%, #00bcd4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.5em 0;
}
.gr-title {
    background: linear-gradient(90deg, #42a5f5 0%, #00bcd4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.tab-button {
    border-bottom: 2px solid transparent;
    transition: all 0.3s ease;
}
.tab-button.selected {
    border-bottom: 2px solid #0066ff;
    background: linear-gradient(90deg, rgba(0, 102, 255, 0.1) 0%, transparent 100%);
}
.gr-label {
    color: #64b5f6 !important;
}
"""

with gr.Blocks(title="LLM Evaluation Dashboard") as demo:
    gr.Markdown("# 🤖 Advanced LLM Evaluation Dashboard")
    gr.Markdown("⚡ Futuristic AI-Powered Multi-Modal Analysis Platform")
    gr.Markdown("Comprehensive evaluation of LLM responses across multiple dimensions with advanced metrics")
    
    with gr.Tab("📊 Single Response Evaluation"):
        gr.Markdown("### Evaluate a single LLM response against a reference")
        with gr.Row():
            with gr.Column():
                ref_input = gr.Textbox(label="Reference/Query Text", lines=4, placeholder="Enter reference or query text...")
                resp_input = gr.Textbox(label="Model Response", lines=4, placeholder="Enter model response...")
            with gr.Column():
                analyze_btn = gr.Button("Analyze Response", scale=1)
                
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Semantic Metrics")
                sim_out = gr.Number(label="Semantic Similarity", precision=3)
                rouge_out = gr.Number(label="ROUGE-1 F1", precision=3)
                len_fit_out = gr.Number(label="Length Fit", precision=3)
                rel_out = gr.Number(label="Relevance", precision=3)
            
            with gr.Column(scale=1):
                gr.Markdown("### Safety & Quality Metrics")
                tox_out = gr.Number(label="Toxicity Penalty", precision=3)
                bias_out = gr.Number(label="Bias Penalty", precision=3)
                coh_out = gr.Number(label="Coherence", precision=3)
                final_out = gr.Number(label="Final Score", precision=3)
            
            with gr.Column(scale=1):
                gr.Markdown("### Additional Info")
                hall_out = gr.Number(label="Hallucination Risk", precision=3)
                hall_entities = gr.JSON(label="Hallucinated Entities")
        
        def single_eval(ref, resp):
            if not ref.strip() or not resp.strip():
                gr.Warning("⚠️ Please provide both reference/query text and model response.")
                return None, None, None, None, None, None, None, None, None, []
            try:
                results = evaluate(ref, {"Response": resp})
                if "Response" not in results:
                    gr.Warning("⚠️ Evaluation failed. Please check your inputs.")
                    return None, None, None, None, None, None, None, None, None, []
                r = results["Response"]
                return (
                    r["Semantic Similarity"], r["ROUGE-1 F1"], r["Length Fit"], r["Relevance"],
                    r["Toxicity Penalty"], r["Bias Penalty"], r["Coherence"], r["Final Score"],
                    r["Hallucination Risk"], r["Hallucinated Entities"]
                )
            except Exception as e:
                gr.Error(f"❌ Evaluation error: {str(e)}")
                return None, None, None, None, None, None, None, None, None, []
        
        analyze_btn.click(
            single_eval,
            inputs=[ref_input, resp_input],
            outputs=[sim_out, rouge_out, len_fit_out, rel_out, tox_out, bias_out, coh_out, final_out, hall_out, hall_entities]
        )
    
    with gr.Tab("🔍 Hallucination Detection"):
        gr.Markdown("### Detect hallucinations in model responses")
        with gr.Row():
            with gr.Column():
                ref_hall = gr.Textbox(label="Reference/Source Text", lines=5, placeholder="Enter source text...")
                resp_hall = gr.Textbox(label="Model Response", lines=5, placeholder="Enter model response...")
                detect_btn = gr.Button("Detect Hallucinations")
            with gr.Column():
                hall_score = gr.Number(label="Hallucination Risk Score", precision=3)
                hall_entities_out = gr.JSON(label="Hallucinated Entities")
                hall_explanation = gr.Textbox(label="Explanation", lines=3, interactive=False)
        
        def hallucination_check(ref, resp):
            if not ref.strip() or not resp.strip():
                gr.Warning("⚠️ Please provide both reference and response text.")
                return None, [], "Please provide both reference and response text."
            try:
                risk, entities = detect_hallucination(ref, resp)
                explanation = f"Risk Score: {risk}/1.0 (higher = more likely to contain hallucinations)\n"
                explanation += f"Hallucinated Entities Found: {len(entities)}\n"
                if entities:
                    explanation += f"Examples: {', '.join([e[0] for e in entities[:5]])}"
                return risk, entities, explanation
            except Exception as e:
                gr.Error(f"❌ Hallucination detection error: {str(e)}")
                return None, [], f"Error: {str(e)}"
        
        detect_btn.click(
            hallucination_check,
            inputs=[ref_hall, resp_hall],
            outputs=[hall_score, hall_entities_out, hall_explanation]
        )
    
    with gr.Tab("⚖️ Bias Detection"):
        gr.Markdown("### Detect bias in model responses")
        with gr.Row():
            with gr.Column():
                bias_input = gr.Textbox(label="Model Response", lines=5, placeholder="Enter model response...")
                bias_btn = gr.Button("Detect Bias")
            with gr.Column():
                bias_score = gr.Number(label="Bias Penalty Score", precision=3)
                bias_json = gr.JSON(label="Named Entities Analysis")
                bias_explanation = gr.Textbox(label="Analysis", lines=3, interactive=False)
        
        def bias_check(text):
            if not text.strip():
                gr.Warning("⚠️ Please provide response text.")
                return None, {}, "Please provide response text."
            try:
                bias_info = detect_bias(text)
                bias_pen = bias_info["Bias Penalty"]
                explanation = f"Bias Score: {bias_pen}/1.0 (higher = more biased)\n"
                explanation += f"Named Entities (PERSON/ORG): {len(bias_info['Named Entities'])}\n"
                if bias_info['Named Entities']:
                    explanation += "Consider fairness in entity representation"
                return bias_pen, bias_info['Named Entities'], explanation
            except Exception as e:
                gr.Error(f"❌ Bias detection error: {str(e)}")
                return None, {}, f"Error: {str(e)}"
        
        bias_btn.click(
            bias_check,
            inputs=[bias_input],
            outputs=[bias_score, bias_json, bias_explanation]
        )
    
    with gr.Tab("💻 Code Quality Check"):
        gr.Markdown("### Evaluate code quality and detect issues")
        with gr.Row():
            with gr.Column():
                code_input = gr.Code(language="python", label="Code to Evaluate", lines=12)
                code_btn = gr.Button("🔍 Analyze Code Quality")
            with gr.Column():
                syntax_status = gr.Textbox(label="Syntax Status", interactive=False)
                code_metrics = gr.JSON(label="📊 Metrics")
                code_errors = gr.JSON(label="🚨 Errors & Issues")
                code_suggestions = gr.Textbox(label="💡 Suggestions", lines=4, interactive=False)
        
        with gr.Row():
            with gr.Column():
                code_explanation = gr.Textbox(label="📝 Code Explanation", lines=10, interactive=False)
            with gr.Column():
                improved_code = gr.Code(language="python", label="✨ Improved Code", lines=10, interactive=False)
        
        def code_quality_check(code):
            if not code.strip():
                return "No code provided", {}, [], "", "", ""
            
            results = check_code_quality(code)
            syntax_msg = "✅ Valid Syntax" if results["syntax_valid"] else "❌ Syntax Errors Found"
            
            errors_str = "\n".join(results["errors"]) if results["errors"] else "No errors detected"
            suggestions_str = "\n".join(results["suggestions"])
            explanation = results.get("explanation", "")
            improved = results.get("improved_code", code)
            
            return syntax_msg, results["metrics"], results["errors"], suggestions_str, explanation, improved
        
        code_btn.click(
            code_quality_check,
            inputs=[code_input],
            outputs=[syntax_status, code_metrics, code_errors, code_suggestions, code_explanation, improved_code]
        )
    
    with gr.Tab("📂 Dataset Evaluation"):
        gr.Markdown("### Batch evaluate LLM responses\n\nSupported formats: CSV, JSON\n\nRequired columns: `question`, `model_name`, `response` (optional: `reference`)")
        with gr.Row():
            with gr.Column():
                dataset_file = gr.File(label="Upload CSV/JSON", file_types=[".csv", ".json"])
                dataset_btn = gr.Button("Evaluate Dataset")
            with gr.Column():
                dataset_json = gr.JSON(label="Aggregated Results")
                download_btn = gr.File(label="Download Results as CSV")
        
        def dataset_eval_wrapper(file):
            if file is None:
                return {}, None
            try:
                results = evaluate_dataset(file)
                csv_buf = io.BytesIO()
                pd.DataFrame(results).to_csv(csv_buf)
                csv_buf.seek(0)
                return results, csv_buf
            except Exception as e:
                gr.Warning(f"Error: {str(e)}")
                return {}, None
        
        dataset_btn.click(
            dataset_eval_wrapper,
            inputs=[dataset_file],
            outputs=[dataset_json, download_btn]
        )
    
    with gr.Tab("📈 Results Visualization"):
        gr.Markdown("### Compare multiple model responses")
        with gr.Group():
            gr.Markdown("#### Enter Model Responses")
            ref_input_viz = gr.Textbox(label="Reference/Query Text", lines=3)
            
            with gr.Row():
                model1_input = gr.Textbox(label="Model 1 Response", lines=3)
                model2_input = gr.Textbox(label="Model 2 Response", lines=3)
                model3_input = gr.Textbox(label="Model 3 Response", lines=3)
            
            viz_btn = gr.Button("Generate Comparison Chart")
        
        with gr.Row():
            plot_output = gr.Image(label="Metrics Comparison Chart")
            results_table = gr.Dataframe(label="Detailed Results")
        
        def visualize_results(ref, m1, m2, m3):
            responses = {}
            if m1.strip():
                responses["Model 1"] = m1
            if m2.strip():
                responses["Model 2"] = m2
            if m3.strip():
                responses["Model 3"] = m3
            
            if not responses:
                return None, None
            
            results = evaluate(ref, responses)
            plot_image, df = plot_results(results)
            return plot_image, df
        
        viz_btn.click(
            visualize_results,
            inputs=[ref_input_viz, model1_input, model2_input, model3_input],
            outputs=[plot_output, results_table]
        )
    
    with gr.Tab("🖼️ AI Image Generation Evaluation"):
        gr.Markdown("### 🎨 Evaluate how accurately AI generated an image from your prompt")
        gr.Markdown("Upload an AI-generated image and provide: (1) the prompt you used, and (2) what you actually see in the generated image.")
        with gr.Row():
            with gr.Column():
                multimodal_image = gr.Image(label="📸 Upload AI-Generated Image", type="pil")
                multimodal_prompt = gr.Textbox(
                    label="📝 Original Prompt (What you asked the AI to create)",
                    lines=4,
                    placeholder="Enter the prompt you used to generate this image..."
                )
                multimodal_description = gr.Textbox(
                    label="🔍 Image Description (What you actually see in the image)",
                    lines=4,
                    placeholder="Describe what you see in the generated image..."
                )
                multimodal_btn = gr.Button("🚀 Evaluate AI Generation Accuracy")
            
            with gr.Column():
                gr.Markdown("### 📊 Accuracy Analysis")
                multimodal_metrics = gr.JSON(label="📈 Accuracy Metrics")
                multimodal_image_props = gr.JSON(label="🎨 Image Properties")
                multimodal_explanation = gr.Textbox(
                    label="📋 Detailed Evaluation",
                    lines=12,
                    interactive=False
                )
        
        def multimodal_check(image, prompt, description):
            if image is None or not prompt.strip():
                return {}, {}, "Please provide both an AI-generated image and the prompt used to create it."
            metrics, props, _, explanation = multimodal_evaluate(image, prompt, description)
            return metrics or {}, props or {}, explanation
        
        multimodal_btn.click(
            multimodal_check,
            inputs=[multimodal_image, multimodal_prompt, multimodal_description],
            outputs=[multimodal_metrics, multimodal_image_props, multimodal_explanation]
        )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        # 🚀 Advanced LLM Evaluation Dashboard v2.5
        
        An AI-powered platform for comprehensive evaluation of Large Language Model (LLM) responses across multiple dimensions with advanced multimodal capabilities.
        
        ### 📊 Core Evaluation Metrics:
        - **Semantic Similarity**: Cosine similarity between embeddings
        - **ROUGE-1 F1**: Unigram overlap between responses
        - **Length Fit**: How well response length matches query
        - **Relevance**: Keyword overlap with query
        - **Coherence**: Logical consistency and uncertainty markers
        - **Toxicity Penalty**: Detects harmful language
        - **Bias Penalty**: Identifies gender/demographic bias
        - **Hallucination Risk**: Detects unsupported claims
        - **Code Quality**: Analyzes Python code syntax, structure, and best practices
        
        ### 🎨 AI Image Generation Evaluation:
        - **Prompt Accuracy**: How well AI followed your prompt
        - **Semantic Matching**: Compare generated image description vs. original prompt
        - **Hallucination Detection**: Identify elements not in the original prompt
        - **Keyword Coverage**: ROUGE-1 score between prompt and description
        - **Safety Analysis**: Detect concerning content in generated images
        - **Actionable Feedback**: Get specific recommendations for better prompts
        
        ### 💻 Advanced Code Analysis:
        - **Syntax Validation**: Check for Python syntax errors
        - **Code Explanation**: Understand what your code does
        - **Metrics Analysis**: Lines of code, functions, classes, imports
        - **Security Scanning**: Detect unsafe practices (eval, exec)
        - **Improved Code**: Get automatically improved versions of your code
        - **Best Practices**: Suggestions for better code quality
        
        ### 💼 Use Cases:
        - ✅ Compare multiple LLM models side-by-side
        - ✅ Evaluate AI image generation accuracy (Stable Diffusion, DALL-E, Midjourney, etc.)
        - ✅ Validate if AI-generated images match your prompts
        - ✅ Evaluate dataset quality at scale
        - ✅ Ensure safety and fairness in responses
        - ✅ Detect hallucinations and inconsistencies
        - ✅ Analyze and improve Python code quality
        - ✅ Get code explanations and refactoring suggestions
        - ✅ Quality assurance for AI systems
        
        ### 🎯 Features:
        - **Single Response Evaluation**: Deep analysis of individual responses
        - **Hallucination Detection**: Identify unsupported claims with entity extraction
        - **Bias Detection**: Find demographic and gender bias in AI responses
        - **Code Quality Check**: Full Python code analysis with improvements
        - **Batch Processing**: Evaluate multiple responses from CSV/JSON
        - **Visualization**: Generate comparison charts and metrics
        - **AI Image Evaluation**: Comprehensive prompt-to-image accuracy analysis
        
        ### 🎨 Design:
        - **Futuristic Blue Theme**: Advanced cyan/blue color scheme optimized for AI
        - **Dark Mode**: Eye-friendly interface with gradient backgrounds
        - **Responsive Layout**: Works on desktop and mobile
        
        ### 🔧 Technical Stack:
        - **Sentence Transformers**: all-MiniLM-L6-v2 for semantic embeddings
        - **Detoxify**: Toxicity detection with neural models
        - **spaCy**: Named entity recognition (optional)
        - **NLTK**: Text tokenization and processing
        - **AST Analysis**: Python code structure analysis
        - **Pillow**: Image processing and analysis
        
        **Version**: 2.5.0 (Enhanced Edition) | **Last Updated**: February 2026
        """)


demo.launch(share=False, theme=custom_theme, css=futuristic_css)

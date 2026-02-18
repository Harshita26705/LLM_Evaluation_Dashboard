"""
Flask Application for LLM Evaluation Dashboard
A professional web-based LLM evaluation platform
"""

from flask import Flask, render_template, request, jsonify
import json
import io
import math
import re
import ast
import nltk
from collections import Counter
from PIL import Image
import numpy as np
import pandas as pd

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.tokenize import TreebankWordTokenizer
from sentence_transformers import SentenceTransformer, util
from detoxify import Detoxify
import base64

# Try to load spaCy for enhanced NER
try:
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    print("   ✅ Loaded spacy model")
except Exception as e:
    print(f"   ⚠️ Spacy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Lazy-loaded models (avoid blocking server startup)
embedder = None
tokenizer = None
toxicity_model = None
models_loaded = False

def ensure_models_loaded():
    """Load models once on first request to avoid startup delays."""
    global embedder, tokenizer, toxicity_model, models_loaded
    if models_loaded:
        return
    print("⏳ Loading models...")
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("   ✅ Loaded sentence embedder")
    except Exception as e:
        print(f"   ⚠️ Could not load embedder: {e}")
        embedder = None

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

    models_loaded = True
    print("✅ Models loaded!\n")

# --- Evaluation Metrics ---
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

# --- Utility Functions ---
def unigrams(text):
    """Extract unigrams from text"""
    if tokenizer is None:
        return [t.lower() for t in text.split() if t.strip()]
    return [t.lower() for t in tokenizer.tokenize(text) if t.strip()]

def rouge1_f1(reference, candidate):
    """Calculate ROUGE-1 F1 score"""
    ref_unigrams = unigrams(reference)
    cand_unigrams = unigrams(candidate)
    ref_counts = Counter(ref_unigrams)
    cand_counts = Counter(cand_unigrams)
    overlap = sum(min(ref_counts[t], cand_counts[t]) for t in set(ref_counts) | set(cand_counts))
    prec = overlap / (sum(cand_counts.values()) + 1e-5)
    rec = overlap / (sum(ref_counts.values()) + 1e-5)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def length_fit(query_text, response_text, expected=EXPECTED_LEN_RATIO, lam=LAMBDA_LEN):
    """Evaluate response length fitness"""
    q_len = max(1, len(unigrams(query_text)))
    r_len = max(1, len(unigrams(response_text)))
    ratio = r_len / q_len
    return math.exp(-lam * abs(math.log(ratio / expected)))

UNCERTAINTY_MARKERS = {"maybe","not","unsure","uncertain","perhaps","guess","might","could"}

def coherence_score(query_text, response_text, alpha=ALPHA_COH):
    """Measure coherence of response with uncertainty and relevance blend"""
    toks = unigrams(response_text)
    if not toks:
        return 0.0
    unc_density = sum(1 for t in toks if t in UNCERTAINTY_MARKERS) / len(toks)
    rel = relevance_score(query_text, response_text)
    return alpha * (1 - unc_density) + (1 - alpha) * rel

def toxicity_penalty(response_text, gamma=GAMMA_TOX):
    """Detect toxicity in text with convex penalty"""
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
    """Calculate bias penalty using convex function"""
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

def detect_bias(response):
    """Detect bias in response with entity extraction"""
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

def detect_hallucination(reference, response):
    """Detect hallucinations in response with entity extraction"""
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

def relevance_score(query, response):
    """Calculate relevance between query and response"""
    query_words = set(unigrams(query))
    response_words = set(unigrams(response))
    
    if not query_words:
        return 0.0
    
    overlap = query_words & response_words
    return len(overlap) / len(query_words)

def composite_score(metrics):
    """Calculate composite score from all metrics"""
    return sum(metrics.get(k, 0) * v for k, v in WEIGHTS.items())

def evaluate_response(reference, response):
    """Evaluate a single response"""
    ensure_models_loaded()
    if not embedder:
        return {"error": "Models not loaded"}
    
    try:
        ref_emb = embedder.encode(reference, convert_to_tensor=True)
        resp_emb = embedder.encode(response, convert_to_tensor=True)
        cosine_sim = float(util.pytorch_cos_sim(ref_emb, resp_emb).item())
    except:
        cosine_sim = 0.0
    
    r1 = rouge1_f1(reference, response)
    len_fit = length_fit(reference, response)
    tox_pen = toxicity_penalty(response)
    hall_risk, hall_entities = detect_hallucination(reference, response)
    bias_info = detect_bias(response)
    bias_pen = bias_info["Bias Penalty"]
    rel = relevance_score(reference, response)
    coh = coherence_score(reference, response)
    
    final = composite_score({
        "relevance": rel,
        "length_fit": len_fit,
        "coherence": coh,
        "rouge1_f1": r1,
        "toxicity_inv": 1 - tox_pen,
        "bias_inv": 1 - bias_pen,
        "hall_inv": 1 - hall_risk
    })
    
    return {
        "semantic_similarity": round(cosine_sim, 3),
        "rouge1_f1": round(r1, 3),
        "length_fit": round(len_fit, 3),
        "relevance": round(rel, 3),
        "coherence": round(coh, 3),
        "toxicity_penalty": round(tox_pen, 3),
        "bias_penalty": round(bias_pen, 3),
        "hallucination_risk": round(hall_risk, 3),
        "final_score": round(final, 3),
        "hallucinated_entities": hall_entities
    }

# --- Flask Routes ---
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Main evaluation dashboard"""
    return render_template('dashboard.html')

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """API endpoint for single response evaluation"""
    try:
        ensure_models_loaded()
        data = request.json
        reference = data.get('reference', '').strip()
        response = data.get('response', '').strip()
        
        if not reference or not response:
            return jsonify({"error": "Reference and response are required"}), 400
        
        results = evaluate_response(reference, response)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500

@app.route('/api/detect-hallucination', methods=['POST'])
def api_detect_hallucination():
    """API endpoint for hallucination detection"""
    try:
        ensure_models_loaded()
        data = request.json
        reference = data.get('reference', '').strip()
        response = data.get('response', '').strip()
        
        if not reference or not response:
            return jsonify({"error": "Reference and response are required"}), 400
        
        risk, entities = detect_hallucination(reference, response)
        return jsonify({
            "hallucination_risk": round(risk, 3),
            "hallucinated_entities": entities,
            "explanation": f"Risk Score: {risk:.2f}/1.0 (higher = more likely to contain hallucinations)"
        })
    except Exception as e:
        return jsonify({"error": f"Hallucination detection failed: {str(e)}"}), 500

@app.route('/api/detect-bias', methods=['POST'])
def api_detect_bias():
    """API endpoint for bias detection"""
    try:
        ensure_models_loaded()
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        bias_info = detect_bias(text)
        return jsonify({
            "bias_score": round(bias_info["Bias Penalty"], 3),
            "entity_analysis": bias_info["Named Entities"],
            "explanation": f"Bias Score: {bias_info['Bias Penalty']:.2f}/1.0"
        })
    except Exception as e:
        return jsonify({"error": f"Bias detection failed: {str(e)}"}), 500

@app.route('/api/check-toxicity', methods=['POST'])
def api_check_toxicity():
    """API endpoint for toxicity check"""
    try:
        ensure_models_loaded()
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        toxicity = toxicity_penalty(text)
        return jsonify({
            "toxicity_score": round(toxicity, 3),
            "is_safe": toxicity < SAFETY_THRESHOLD,
            "explanation": f"Toxicity Level: {toxicity:.2f}/1.0"
        })
    except Exception as e:
        return jsonify({"error": f"Toxicity check failed: {str(e)}"}), 500

@app.route('/api/compare-models', methods=['POST'])
def api_compare_models():
    """API endpoint for multi-model comparison"""
    try:
        ensure_models_loaded()
        data = request.json
        question = data.get('question', '').strip()
        models = data.get('models', [])
        
        if not question or not models or len(models) < 2:
            return jsonify({"error": "Question and at least 2 models required"}), 400
        
        comparisons = []
        for model in models:
            model_name = model.get('name', 'Unknown')
            model_response = model.get('response', '').strip()
            
            if not model_response:
                continue
            
            eval_result = evaluate_response(question, model_response)
            eval_result['model_name'] = model_name
            comparisons.append(eval_result)
        
        if not comparisons:
            return jsonify({"error": "No valid model responses provided"}), 400
        
        # Find winner (highest final score)
        winner_index = max(range(len(comparisons)), key=lambda i: comparisons[i]['final_score'])
        
        return jsonify({
            "comparisons": comparisons,
            "winner_index": winner_index,
            "winner": comparisons[winner_index]['model_name']
        })
    except Exception as e:
        return jsonify({"error": f"Model comparison failed: {str(e)}"}), 500

def analyze_image_properties(image_data):
    """Analyze basic image properties"""
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        img = Image.open(io.BytesIO(img_bytes))
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
    except Exception as e:
        return {"error": str(e)}

def multimodal_evaluate(image_data, prompt_text, description_text=None):
    """Evaluate AI-generated image against the prompt used to create it"""
    if not image_data or not prompt_text.strip():
        return None, None, {}, "Please provide both an image and the prompt used to generate it."
    
    try:
        image_props = analyze_image_properties(image_data)
        
        # If no description provided, we can only evaluate the prompt
        if not description_text or not description_text.strip():
            explanation = "⚠️ No image description provided for full evaluation.\n\n"
            explanation += f"📋 Prompt Analysis:\n"
            explanation += f"- Prompt Length: {len(unigrams(prompt_text))} words\n"
            explanation += f"- Image Properties: {image_props}\n\n"
            explanation += "💡 Tip: Describe what you see in the generated image for a complete accuracy evaluation."
            
            return {}, image_props, {}, explanation
        
        # Full evaluation: Compare description vs prompt
        if not embedder:
            return None, image_props, {}, "Error: Embedder model not loaded"
        
        # Semantic similarity between prompt and what was actually generated
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
            prompt_match_score * 0.35 +
            rouge * 0.25 +
            relevance * 0.20 +
            coherence * 0.15 +
            (1 - hall_risk) * 0.05
        )
        
        eval_results = {
            "Prompt-Image Match Score": round(prompt_match_score, 3),
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
        explanation += f"\n📐 Image Properties: {image_props.get('Size (pixels)', 'N/A')} | {image_props.get('Mode', 'N/A')}\n"
        
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

@app.route('/api/evaluate-image', methods=['POST'])
def api_evaluate_image():
    """API endpoint for AI image generation evaluation"""
    ensure_models_loaded()
    data = request.json
    image_data = data.get('image', '')
    prompt = data.get('prompt', '').strip()
    description = data.get('description', '').strip()
    
    if not image_data or not prompt:
        return jsonify({"error": "Image and prompt are required"}), 400
    
    try:
        metrics, props, _, explanation = multimodal_evaluate(image_data, prompt, description)
        
        if metrics is None:
            return jsonify({"error": explanation}), 500
        
        return jsonify({
            "metrics": metrics,
            "image_properties": props,
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"error": f"Image evaluation failed: {str(e)}"}), 500

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

@app.route('/api/analyze-code', methods=['POST'])
def api_analyze_code():
    """API endpoint for code quality analysis"""
    ensure_models_loaded()
    data = request.json
    code = data.get('code', '').strip()
    language = data.get('language', 'python').lower()
    
    if not code:
        return jsonify({"error": "Code is required"}), 400
    
    try:
        results = check_code_quality(code)
        return jsonify({
            "syntax_valid": results["syntax_valid"],
            "metrics": results["metrics"],
            "errors": results["errors"],
            "suggestions": results["suggestions"],
            "explanation": results["explanation"],
            "improved_code": results["improved_code"],
            "language": language
        })
    except Exception as e:
        return jsonify({"error": f"Code analysis failed: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)

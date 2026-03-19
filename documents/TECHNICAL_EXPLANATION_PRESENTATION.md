# LLM Evaluation Dashboard - Technical Explanation for Presentation

## 1. Executive Summary (non-technical audience)
This project is a web platform that measures quality and safety of AI model outputs.
It evaluates text responses, compares multiple model outputs, analyzes generated code, and evaluates AI-generated images against prompts.

At a high level:
- The frontend is a Flask-rendered web UI with tabs.
- The backend exposes JSON APIs for each evaluation feature.
- The scoring engine combines NLP metrics, safety checks, and heuristics.
- There is a legacy Gradio app still in the repo for earlier workflows.

## 2. System Architecture

### 2.1 Current production path (Flask)
- UI pages: `templates/*.html`
- Browser logic: inline script in `templates/dashboard.html`, plus `static/js/script.js` and `static/js/home.js`
- API server: `flask_app.py`
- Advanced code analysis engine: `code_analyzer.py`

### 2.2 Legacy/alternate path (Gradio)
- Standalone app: `app.py`
- This includes similar metrics and tabs but runs as a Gradio app.
- Startup scripts `run_app.bat` and `start_app.bat` point to this path.

## 3. Feature-by-Feature Technical Flow

## 3.1 Single Response Evaluation
Used for: Quality scoring of one model response against a reference/query.

Backend flow:
1. UI sends `reference` + `response` to `POST /api/evaluate`.
2. `flask_app.py` computes:
   - semantic similarity
   - ROUGE-1
   - length fit
   - relevance
   - coherence
   - toxicity penalty
   - bias penalty
   - hallucination risk
3. A weighted composite score is returned as JSON.

Tech used:
- Optional `sentence-transformers` for semantic similarity.
- `nltk` tokenizer for lexical metrics.
- Optional `detoxify` for toxicity, with lexical fallback.
- Custom weighted scoring logic.

## 3.2 Hallucination Detection
Used for: Detecting unsupported entities/claims in generated text.

Backend flow:
1. UI sends source/reference + response to `POST /api/detect-hallucination`.
2. `flask_app.py` extracts entities from both texts.
3. Unsupported response entities are marked hallucinated.
4. Risk score is built from entity mismatch + uncertainty markers.

Tech used:
- Optional spaCy NER if available.
- Regex/entity heuristics fallback.
- F1-style entity support scoring.

## 3.3 Bias Detection
Used for: Fairness checks across demographic and gender references.

Backend flow:
1. UI sends text to `POST /api/detect-bias`.
2. Bias term frequencies are counted by group buckets.
3. Imbalance becomes a convex penalty score.
4. Entity analysis metadata is returned.

Tech used:
- Token-based demographic group counters.
- Optional spaCy entity extraction.

## 3.4 Toxicity Check
Used for: Flagging harmful language.

Backend flow:
1. UI sends text to `POST /api/check-toxicity`.
2. If Detoxify is available, model predicts toxicity probability.
3. Otherwise, lexical toxic-term heuristic is used.
4. Response includes toxicity score and safe/unsafe flag.

Tech used:
- Optional `detoxify` model.
- Safe fallback keyword heuristic.

## 3.5 Multi-Model Comparison
Used for: Comparing several model outputs on the same question.

Backend flow:
1. UI sends one question and 2-3 model responses to `POST /api/compare-models`.
2. Each response is evaluated through the same single-response pipeline.
3. Results are sorted and winner is selected by highest final score.

Tech used:
- Reuse of the same scoring engine for consistency.
- Frontend renders table and highlights winner.

## 3.6 Code Analysis
Used for: Static quality checks, bug/security checks, and improvement suggestions.

There are three API modes:
- `POST /api/analyze-code` (basic deterministic analysis)
- `POST /api/analyze-code-enhanced` (enhanced analysis by selected type)
- `POST /api/analyze-llm-code` (checks prompt-to-code alignment)

### Basic mode
- Implemented in `check_code_quality()` in `flask_app.py`.
- Uses Python AST parsing, structural metrics, smell checks, and suggestions.
- Returns explanation text, metrics, and improved-code scaffold.

### Enhanced mode
- Delegates to `code_analyzer.py` via singleton `get_analyzer()`.
- Supports `full`, `bugs`, `security`, `improve`, `documentation`.
- Uses deterministic heuristics plus optional AI text generation path.

### LLM-generated code evaluation mode
- Scores whether generated code matches original prompt intent.
- Includes requirement checks, syntax/logic checks, and issue list.

Tech used:
- Python `ast` for syntax/structure analysis.
- Heuristic security checks (`eval`, `exec`, `pickle.loads`, `shell=True`, etc.).
- Optional local transformers pipeline first; Ollama HTTP fallback next.

## 3.7 Multimodal (AI Image Prompt Accuracy)
Used for: Measuring how well an AI-generated image matches the original prompt.

Backend flow:
1. UI reads uploaded image as Base64 data URL.
2. Sends image + prompt (+ optional manual description) to `POST /api/evaluate-image`.
3. `flask_app.py` extracts pixel-level features:
   - dominant color and color strengths
   - brightness and contrast
   - edge density (detail proxy)
   - grayscale score
4. Prompt keyword cues are aligned to pixel features.
5. If description is provided, text metrics are blended in.
6. Final metrics and explanation are returned.

Tech used:
- `Pillow` + `numpy` for image processing.
- Pixel-first deterministic scoring.
- Optional text-semantic refinement.

Note: This is not a full vision-language model inference pipeline. It is a fast, explainable heuristic evaluator.

## 4. Module-by-Module Explanation (what is used for what)

## 4.1 Core backend modules
- `flask_app.py`
  - Main Flask server, routes, evaluation engine, model loading, and API error handling.
  - Includes Render-safe fallback behavior (`DISABLE_HEAVY_MODELS`).
- `code_analyzer.py`
  - Advanced code analysis service used by enhanced endpoints.
  - Provides static analysis, security checks, bug checks, code improvement, and documentation generation.

## 4.2 Frontend modules
- `templates/base.html`
  - Shared layout shell (header, nav, footer).
- `templates/index.html`
  - Landing page and feature entry points.
- `templates/dashboard.html`
  - Main interactive evaluation UI and API-calling JavaScript.
- `templates/learn-more.html`
  - Product/feature education page.
- `templates/404.html`, `templates/500.html`
  - Error pages for browser routes.
- `static/css/style.css`
  - Shared visual system, layout, responsive behavior, tab/result styling.
- `static/js/script.js`
  - Navigation behavior, sticky header, tab URL state handling.
- `static/js/home.js`
  - Home-page reveal animations via IntersectionObserver.
- `static/js/dashboard.js`
  - Contains helper formatting utilities; most active dashboard logic is inline in `dashboard.html`.

## 4.3 Legacy and alternate runtime modules
- `app.py`
  - Legacy Gradio implementation with similar evaluation capabilities.
  - Still useful for notebook/demo style interaction.
- `flask_app_backup.py`
  - Older backup of the Flask backend.
- `debug_app.py`
  - Import and execution sanity-check wrapper for `app.py`.

## 4.4 Setup and operations modules
- `requirements_flask.txt`
  - Lightweight dependency set for Flask deployment.
- `requirements.txt`
  - Heavier dependency set (legacy Gradio/full model stack).
- `render.yaml`
  - Render deployment config, Gunicorn start, environment flags.
- `init_models.py`, `setup.py`
  - Local initialization scripts for NLP assets (NLTK/spaCy) and dependency checks.
- `init_setup.bat`, `start_flask.bat`, `start_app.bat`, `run_app.bat`
  - Windows entry-point scripts for setup and app startup.

## 4.5 Local model helper modules
- `setup_local_ai.py`
  - CLI helper intended to manage project-local GGUF models.
- `local_model_manager.py`
  - Currently a stub placeholder in this repository state.

## 4.6 QA and validation modules
- `test_code_analysis.py`
  - Endpoint-level smoke tests for code-analysis APIs.
- `test_json_responses.py`, `test_render_fix.py`
  - Validate JSON-safe API behavior and fallback compatibility.
- `test_template.py`
  - Template rendering sanity check.
- `sample_data.csv`
  - Example dataset for evaluation workflows.

## 4.7 Documentation modules
- `documents/*.md`
  - Setup guides, deployment notes, implementation summaries, and troubleshooting docs.
- `RENDER_FIX.md`
  - Specific reliability fix notes for hosted deployment.

## 5. Reliability and Deployment Design

## 5.1 Hosted safety strategy
For memory-constrained hosting, heavy NLP models can be disabled and lexical fallbacks are used.
This keeps API responses reliable and prevents worker timeouts.

## 5.2 JSON reliability
API routes parse input safely (`request.get_json(silent=True)`) and API error handlers return JSON for `/api/*` routes, avoiding frontend parsing failures.

## 5.3 Deployment path
- Gunicorn serves `flask_app:app` in `render.yaml`.
- `requirements_flask.txt` is tuned for lighter hosted runtime.

## 6. Talking Points for Presentation (ready to say)

Use this 90-second explanation:

"We built a full-stack LLM evaluation platform. The browser UI is tab-based and calls Flask APIs for each evaluation task. The backend computes quality, safety, and factuality metrics using a mix of NLP models and deterministic scoring. For code use cases, we provide both static analysis and enhanced analysis with optional AI assistance. For image generation, we evaluate prompt-image alignment through explainable pixel features and optional textual refinement. We designed the system to be production-resilient: in low-memory environments, heavy models are automatically disabled and replaced with lexical fallbacks, so APIs stay responsive and return valid JSON consistently. The repository also includes a legacy Gradio implementation, deployment scripts, and test utilities for endpoint reliability." 

## 7. Honest Limitations (good to mention in Q&A)
- Multimodal scoring is heuristic/pixel-first, not a full vision-language model benchmark.
- Some advanced NLP checks are less nuanced when heavy models are disabled.
- Local model manager path is scaffolded but currently placeholder in this repo snapshot.

## 8. Future Roadmap Suggestions
- Add CLIP/BLIP/VLM-based multimodal semantic scoring.
- Add async job queue for large batch evaluations.
- Add authentication and per-project history storage.
- Consolidate legacy Gradio and Flask branches into one runtime.

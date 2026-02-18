# 🎯 LLM Evaluation Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)

A comprehensive web-based dashboard for evaluating Large Language Model (LLM) outputs with advanced metrics and AI-generated content analysis.

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation)

</div>

---

## 📋 Overview

The LLM Evaluation Dashboard is a professional web application built with Flask that provides comprehensive analysis and evaluation of LLM-generated content. It offers multiple evaluation modes including response quality assessment, hallucination detection, bias analysis, toxicity checking, multi-model comparison, code quality analysis, and AI image generation evaluation.

## ✨ Features

### 🔍 **Single Response Evaluation**
- Semantic similarity analysis using SentenceTransformers
- ROUGE-1 F1 score computation
- Text coherence measurement with uncertainty markers
- Toxicity detection with convex penalty functions
- Bias detection using spaCy NER
- Hallucination risk assessment
- **Responsive card layout** (3 cards per row on desktop, wraps on mobile)

### 🔮 **Hallucination Detection**
- Named Entity Recognition (NER) using spaCy
- Entity extraction and comparison
- F1 score calculation for accuracy
- Hallucinated entity identification

### ⚖️ **Bias Detection**
- Gender bias analysis
- Demographic bias identification
- Named entity-based bias scoring
- Convex penalty functions for fair evaluation

### 🛡️ **Toxicity Checking**
- Detoxify model integration
- Multi-category toxicity analysis
- Convex transformation for accurate scoring
- Safety threshold validation

### 🔄 **Multi-Model Comparison**
- Side-by-side comparison of multiple LLM outputs
- **Professional table format** with large, readable fonts
- Comprehensive metric comparison
- Automatic winner determination with 🏆 trophy
- Color-coded scores and winner highlighting

### 💻 **Code Quality Analysis**
Analyzes LLM-generated code with:
1. **Metrics Cards**: Total Lines, Code Lines, Functions, Classes, Comments, Imports
2. **Code Quality Score**: Letter grades (A+ to F) with color coding
3. **Suggestions & Errors**: Color-coded recommendations with emojis
4. **Code Explanation**: Detailed analysis of code structure
5. **Improved Code**: Enhanced version with **copy-to-clipboard** functionality
- **Code editor styling** with syntax highlighting
- Dark theme (#1e1e1e) with monospace fonts
- One-click copy button with visual feedback

### 🖼️ **AI Image Generation Evaluation**
Evaluates AI-generated images (Stable Diffusion, DALL-E, Midjourney) against prompts:
- **Vertical card layout** with flex-wrap for better readability
- Overall accuracy with color-coded indicators
- Prompt-to-image match scoring
- Keyword coverage analysis
- Relevance and coherence metrics
- Hallucination risk for generated images
- Safety score assessment
- Image property analysis (format, size, mode)

## 🛠️ Technology Stack

### Backend
- **Flask 3.0.3** - Web framework
- **SentenceTransformers** - Semantic similarity (all-MiniLM-L6-v2)
- **Detoxify** - Toxicity detection
- **spaCy** - Named Entity Recognition (en_core_web_sm)
- **NLTK** - Natural language toolkit
- **ROUGE** - Text similarity metrics
- **Python AST** - Code structure analysis
- **Pillow** - Image processing
- **Base64** - Image encoding/decoding

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with flex-wrap layouts
- **JavaScript (ES6+)** - Interactive functionality
- **Font Awesome** - Icons
- **Google Fonts** - Nunito Sans typography

### Features
- Responsive design (Desktop → Tablet → Mobile)
- Dark theme with cyan/blue accents
- Animated transitions and hover effects
- Real-time API interactions
- Copy-to-clipboard functionality
- Professional data tables
- Code editor styling

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Harshita26705/LLM_Evaluation_Dashboard.git
cd LLM_Evaluation_Dashboard
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements_flask.txt
```

### Step 4: Download spaCy Model (Optional but Recommended)
```bash
python -m spacy download en_core_web_sm
```

## 🚀 Usage

### Running the Application

#### Option 1: Using Batch Script (Windows)
```bash
start_flask.bat
```

#### Option 2: Manual Start
```bash
python flask_app.py
```

The application will start at: **http://127.0.0.1:5000**

### Accessing the Dashboard

1. Open your browser and navigate to `http://127.0.0.1:5000`
2. Click on **"Dashboard"** in the navigation menu
3. Select the evaluation tab you need:
   - **Single Response** - Evaluate individual LLM outputs
   - **Hallucination** - Detect unsupported claims
   - **Bias Detection** - Identify demographic biases
   - **Toxicity** - Check for harmful content
   - **Multi-Model** - Compare multiple models
   - **Code Analysis** - Analyze generated code quality
   - **Multimodal** - Evaluate AI-generated images

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/dashboard` | GET | Dashboard interface |
| `/api/evaluate` | POST | Single response evaluation |
| `/api/detect-hallucination` | POST | Hallucination detection |
| `/api/detect-bias` | POST | Bias analysis |
| `/api/check-toxicity` | POST | Toxicity checking |
| `/api/compare-models` | POST | Multi-model comparison |
| `/api/analyze-code` | POST | Code quality analysis |
| `/api/evaluate-image` | POST | AI image evaluation |

## 📖 Documentation

Detailed documentation available in the `documents/` folder:
- **[QUICKSTART.md](documents/QUICKSTART.md)** - Quick start guide
- **[IMPLEMENTATION_SUMMARY.md](documents/IMPLEMENTATION_SUMMARY.md)** - Implementation details
- **[DEPLOYMENT_CHECKLIST.md](documents/DEPLOYMENT_CHECKLIST.md)** - Deployment guide
- **[HF_DEPLOYMENT.md](documents/HF_DEPLOYMENT.md)** - Hugging Face deployment

## 🎨 UI Highlights

### New Design Features
- ✅ **Flex-wrap card layouts** - Result cards adapt to screen size (2-3 per row)
- ✅ **Professional data tables** - Large, readable fonts for comparisons
- ✅ **Code editor styling** - Dark theme with syntax-friendly formatting
- ✅ **Copy functionality** - One-click code copying with visual feedback
- ✅ **Quality grading** - Letter grades (A+ to F) with color indicators
- ✅ **Responsive design** - Optimized for all device sizes
- ✅ **Color-coded metrics** - Green (good), yellow (warning), red (critical)

## 🔧 Configuration

### Environment Variables (Optional)
Create a `.env` file in the root directory:
```env
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here
PORT=5000
```

### Model Configuration
Models are loaded lazily on first use:
- **Sentence Embedder**: `sentence-transformers/all-MiniLM-L6-v2`
- **Toxicity Model**: `detoxify/original`
- **spaCy Model**: `en_core_web_sm` (optional)

## 📁 Project Structure

```
LLM_Evaluation_Dashboard/
├── flask_app.py              # Main Flask application
├── requirements_flask.txt    # Python dependencies
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── static/
│   ├── css/
│   │   └── style.css        # Enhanced UI styles
│   └── js/
│       ├── dashboard.js     # Dashboard functionality
│       ├── home.js          # Home page scripts
│       └── script.js        # Common scripts
├── templates/
│   ├── base.html            # Base template
│   ├── index.html           # Home page
│   ├── dashboard.html       # Dashboard interface
│   ├── 404.html             # 404 error page
│   └── 500.html             # 500 error page
└── documents/
    ├── QUICKSTART.md
    ├── IMPLEMENTATION_SUMMARY.md
    └── DEPLOYMENT_CHECKLIST.md
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Harshita Suri**
- GitHub: [@Harshita26705](https://github.com/Harshita26705)
- Repository: [LLM_Evaluation_Dashboard](https://github.com/Harshita26705/LLM_Evaluation_Dashboard)

## 🙏 Acknowledgments

- **SentenceTransformers** - For semantic similarity models
- **Detoxify** - For toxicity detection
- **spaCy** - For NER capabilities
- **Flask** - For the web framework
- **Font Awesome** - For beautiful icons

## 📊 Statistics

- **29 Files** committed
- **6,312 Lines** of code
- **7 Evaluation Modes**
- **7 API Endpoints**
- **Fully Responsive** UI

---

<div align="center">
Made with ❤️ by Harshita Suri
</div>

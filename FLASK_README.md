# LLM Evaluation Dashboard - Flask Edition

A professional, enterprise-grade web application for comprehensive Large Language Model (LLM) evaluation using Flask, HTML, CSS, and JavaScript.

## 🌟 Features

### Evaluation Capabilities
- **Semantic Similarity**: Transformer-based embedding similarity scoring
- **ROUGE-1 F1**: Unigram overlap analysis
- **Coherence Score**: Text consistency and logical flow analysis
- **Toxicity Detection**: Harmful language identification
- **Bias Detection**: Gender and demographic bias analysis
- **Hallucination Detection**: Unsupported claim identification
- **Length Fit**: Response length appropriateness evaluation
- **Relevance Score**: Query-response alignment analysis
- **Composite Scoring**: Overall quality assessment

### User Interface
- **Modern Design**: Dark theme with cyan/neon accents (matching portfolio style)
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Enterprise Quality**: Professional UI/UX with smooth animations
- **Intuitive Tabs**: 5 different evaluation modules
- **Real-time Feedback**: Instant analysis with visual feedback

### Technical Stack
- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **NLP**: Sentence-Transformers, NLTK, Detoxify
- **Deployment**: Production-ready with Gunicorn

## 📁 Project Structure

```
LLM_Dashboard/
├── flask_app.py              # Main Flask application
├── requirements_flask.txt    # Python dependencies
├── templates/
│   ├── base.html            # Base template (navigation, footer)
│   ├── index.html           # Home/landing page
│   ├── dashboard.html       # Main evaluation dashboard
│   ├── 404.html             # 404 error page
│   └── 500.html             # 500 error page
├── static/
│   ├── css/
│   │   └── style.css        # All styling
│   └── js/
│       ├── script.js        # Navigation and general utilities
│       ├── dashboard.js     # Dashboard-specific functions
│       └── home.js          # Home page animations
└── [other project files]
```

## 🚀 Installation & Setup

### 1. Clone/Setup the Project
```bash
cd "LLM_Dashboard"
```

### 2. Create Virtual Environment (if not already done)
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements_flask.txt
```

### 4. Download Required Models
```bash
python -m spacy download en_core_web_sm
```

### 5. Run the Application
```bash
python flask_app.py
```

The application will start on `http://localhost:5000`

## 🎯 Usage

### Home Page (`/`)
- Overview of the platform
- Feature highlights
- Statistics and benefits
- Call-to-action buttons

### Dashboard (`/dashboard`)
- **Single Response Evaluation Tab**
  - Input: Reference text + Model response
  - Output: 9 evaluation metrics + composite score

- **Hallucination Detection Tab**
  - Input: Source text + Response
  - Output: Risk score + Hallucinated entities

- **Bias Detection Tab**
  - Input: Text to analyze
  - Output: Bias score + Entity distribution

- **Toxicity Detection Tab**
  - Input: Text to check
  - Output: Toxicity level + Safety status

- **Multimodal Evaluation Tab**
  - Input: Image + Description
  - Output: Image properties + Description quality metrics

## 📊 API Endpoints

### POST `/api/evaluate`
Evaluate a single LLM response against a reference.

**Request:**
```json
{
    "reference": "Source or query text",
    "response": "LLM generated response"
}
```

**Response:**
```json
{
    "semantic_similarity": 0.87,
    "rouge1_f1": 0.75,
    "length_fit": 0.92,
    "relevance": 0.85,
    "coherence": 0.88,
    "toxicity_penalty": 0.05,
    "bias_penalty": 0.10,
    "hallucination_risk": 0.15,
    "final_score": 0.82
}
```

### POST `/api/detect-hallucination`
Detect hallucinations in responses.

### POST `/api/detect-bias`
Detect bias in text.

### POST `/api/check-toxicity`
Check for toxic content.

## 🎨 Design Theme

The application uses a professional dark theme with cyan/neon accents:

- **Primary Color**: Cyan (#00f0ff)
- **Secondary Color**: Blue (#0066ff)
- **Background**: Dark Navy (#191f36)
- **Secondary Background**: Darker Navy (#262B40)
- **Text**: White (#fff)

All elements feature smooth transitions and hover effects for enhanced user experience.

## 📱 Responsive Design

Breakpoints:
- **Desktop**: Full layout (1200px+)
- **Tablet**: Optimized columns (991px - 1200px)
- **Mobile**: Single column, full-width (< 991px)

## 🔧 Configuration

Edit `flask_app.py` to modify:
- Port: Change `port=5000` in `app.run()`
- Debug mode: Change `debug=True/False`
- File size limit: Modify `app.config['MAX_CONTENT_LENGTH']`

## 📈 Evaluation Metrics Explained

1. **Semantic Similarity** (0-1): Embedding-based similarity between texts
2. **ROUGE-1 F1** (0-1): Unigram overlap between response and reference
3. **Coherence** (0-1): Presence of uncertainty markers and logical flow
4. **Toxicity** (0-1): Level of harmful or toxic language
5. **Bias** (0-1): Gender/demographic bias presence
6. **Hallucination Risk** (0-1): Likelihood of unsupported claims
7. **Length Fit** (0-1): How appropriate response length is
8. **Relevance** (0-1): How well response matches query
9. **Final Score** (0-1): Weighted composite of all metrics

## 🛡️ Error Handling

The application includes:
- Custom 404 error page for missing routes
- Custom 500 error page for server errors
- Try-except blocks for model loading
- Graceful degradation if models fail

If models fail to load, all functions return safe default values.

## 🚢 Deployment

### Using Gunicorn (Production)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 flask_app:app
```

### Using Docker (Optional)
Create a `Dockerfile` for containerized deployment.

### Environment Variables
Create a `.env` file for sensitive data:
```
FLASK_ENV=production
FLASK_DEBUG=False
```

## 📝 Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 🐛 Troubleshooting

**Models not loading?**
```bash
python -m spacy download en_core_web_sm
pip install --upgrade sentence-transformers
```

**Port already in use?**
```python
# In flask_app.py, change:
app.run(debug=True, host='0.0.0.0', port=5001)
```

**CSS not loading?**
Clear browser cache or do a hard refresh (Ctrl+Shift+R)

## 📞 Support

For issues or questions:
- GitHub: [Harshita26705](https://github.com/Harshita26705)
- LinkedIn: [harshita-suri](https://linkedin.com/in/harshita-suri)

## 📄 License

This project is open source and available for educational and commercial use.

## 🌟 Version

**LLM Evaluation Dashboard v2.0 (Flask Edition)**
- Built: February 2026
- Framework: Flask + HTML/CSS/JavaScript
- Status: Production Ready

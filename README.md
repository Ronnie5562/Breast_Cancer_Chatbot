# 🎗️ Breast Cancer QA Chatbot

A specialized AI-powered chatbot providing accurate, accessible breast cancer information using transformer-based natural language processing.

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗-transformers-yellow.svg)](https://huggingface.co/transformers/)


![Image](https://github.com/user-attachments/assets/d4d00210-658f-4ff2-bc10-1ab66754c680)

## 🌟 Overview

This project addresses critical healthcare accessibility challenges by providing 24/7 access to reliable breast cancer information. Built with Microsoft's DialoGPT-small and fine-tuned on curated medical Q&A data, the chatbot delivers evidence-based responses while maintaining appropriate medical boundaries.

### Key Features

- 🔬 **Domain-Specific**: Focused exclusively on breast cancer information
- 🚀 **Fast Training**: Optimized pipeline achieving convergence in ~22 minutes
- 💻 **Dual Interface**: Both CLI and web UI for different user needs
- 🛡️ **Safety First**: Built-in domain validation and medical disclaimers
- 📱 **Accessible**: Responsive design with screen reader support

## 📊 Performance Highlights

| Metric | Value |
|--------|-------|
| Training Loss | 0.95 → 0.0 (100% improvement) |
| Validation Loss | 0.515 |
| Training Time | ~ 3.6 hours |
| Dataset Size | 527 Q&A pairs (expanded from 187) |
| Model Architecture | DialoGPT-small fine-tuned |

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
CUDA-compatible GPU (recommended)
8GB+ RAM
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Ronnie5562/breast_cancer_chatbot.git
cd breast-cancer-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### CLI Interface

```bash
# Interactive mode
python cli.py
```

#### Web Interface

```bash
# Start the web server
python app.py

# Open browser to http://localhost:5000
```


## 📁 Project Structure

```
breast-cancer-chatbot/
├── data/
│   ├── raw/            # Original datasets
│   └── processed/      # Processed datasets
├── models/
│   ├── fine_tuned_breast_cancer_model/  # Trained model
│   └── checkpoints/                # Training checkpoints
├── scripts/
│   ├── cli.py                    # Command-line interface
│   ├── run.py                     # Web application
│   └── ...
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔧 Configuration

### Training Configuration

```json
{
  "model": {
    "base_model_name": "microsoft/DialoGPT-small",
    "max_sequence_length": 512
  },
  "training": {
    "learning_rate": 5e-05,
    "batch_size": 4,
    "num_epochs": 5,
    "early_stopping_patience": 3
  },
  "data": {
    "train_split": 0.7,
    "val_split": 0.1,
    "test_split": 0.2
  }
}
```

## 📈 Model Performance

### Training Evolution

| Iteration | Learning Rate | Final Train Loss | Final Eval Loss | Training Time |
|-----------|---------------|------------------|-----------------|---------------|
| 1 | 1e-4 | 0.95 | 0.89 | 6h 45m |
| 2 | 5e-5 | 0.72 | 0.78 | 4h 20m |
| 3 | 5e-5 | 0.48 | 0.62 | 2h 15m |
| 4 | 5e-5 | 0.31 | 0.58 | 1h 45m |
| 5 | 5e-5 | 0.0 | 0.515 | 21m 48s |


## 🛠️ Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test suite
python -m pytest tests/test_model.py -v

# Run with coverage
python -m pytest --cov=src tests/
```

### Adding New Data

```bash
# Process new medical Q&A data
python src/data_preprocessing.py --input new_data.csv --expand --validate

# Retrain with new data
python src/model_training.py --resume-from-checkpoint --new-data
```

## 🔒 Safety & Ethical Considerations

- **Medical Disclaimers**: All responses include appropriate medical disclaimers
- **Domain Boundaries**: Strict validation ensures responses stay within breast cancer scope
- **Privacy**: No user conversations are stored or logged
- **Bias Mitigation**: Regular evaluation for potential biases in medical advice
- **Professional Guidance**: Consistent emphasis on consulting healthcare professionals


### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ interfaces/
flake8 src/ interfaces/
```

## 🙏 Acknowledgments

- **Dataset**: Curated medical Q&A pairs from verified medical sources
- **Model**: Built on Microsoft's DialoGPT architecture
- **Libraries**: HuggingFace Transformers, PyTorch, Flask
- **Inspiration**: The need for accessible, reliable medical information


## ⚠️ Disclaimer

This chatbot is designed to provide educational information about breast cancer and should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.

# Author

## [`Abimbola Ronald`](https://www.linkedin.com/in/abimbola-ronald-977299224/)

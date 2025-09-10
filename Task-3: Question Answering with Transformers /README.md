# Question Answering with BERT on SQuAD

A professional implementation of a Question Answering system using **BERT** fine-tuned on the **Stanford Question Answering Dataset (SQuAD)**.  
This project demonstrates state-of-the-art natural language processing capabilities for **extractive question answering**.

---

## 🚀 Features
- **BERT-based Question Answering**: Fine-tuned BERT model for extractive QA tasks  
- **Efficient Training**: Uses Hugging Face Transformers + Accelerate  
- **Evaluation**: Implements SQuAD metrics (EM & F1)  
- **Modular Architecture**: Clean and maintainable codebase  
- **GPU Acceleration**: Mixed precision training support  
- **Reproducibility**: Configurable and consistent setup  

---

## 📊 Performance
Fine-tuned on **SQuAD v1.1** validation set:

| Metric       | Value  |
|--------------|--------|
| Exact Match  | ~80.8% |
| F1 Score     | ~88.5% |

---

## 🏗️ Project Structure
```
question-answering-project/
├── src/
│   ├── main.py              # Training pipeline
│   ├── preprocessing.py     # Data preprocessing
│   ├── train.py             # Training loop
│   ├── models/
│   │   └── get_model.py     # Model management
│   └── utils/
│       └── metrics.py       # Evaluation metrics
├── tests/
│   └── evaluate.py          # Evaluation scripts
├── config/
│   └── config.yaml          # Training configuration
├── requirements.txt         # Dependencies
└── README.md
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+  
- CUDA-enabled GPU (recommended)  

### Setup
```bash
# Clone repository
git clone https://github.com/Abdullah1Allnami/Elevvo-Internship.git
cd Elevvo-Internship/Task-3: Question Answering with Transformers 

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Training
```bash
# Train with default config
python src/main.py

# Train with custom config
python src/main.py --config config/custom.yaml
```

### Inference
```python
from transformers import pipeline

model_path = "bert-finetuned-squad-accelerate"
qa_pipeline = pipeline("question-answering", model=model_path)

context = "The Amazon rainforest is a moist broadleaf tropical rainforest in South America."
question = "Where is the Amazon rainforest located?"
result = qa_pipeline(question=question, context=context)

print(result["answer"], result["score"])
```

### Evaluation
```bash
python -m tests.evaluate --model_path bert-finetuned-squad-accelerate --num_samples 10
```

---

## ⚙️ Configuration
Config file (`config/config.yaml`):
```yaml
model_checkpoint: "bert-base-cased"
max_length: 384
stride: 128
batch_size: 8
learning_rate: 2e-5
num_epochs: 3
output_dir: "bert-finetuned-squad-accelerate"
mixed_precision: "fp16"
logging_steps: 100
```

---

## Training Pipeline
- **Preprocessing**: Tokenization, context chunking, answer alignment  
- **Optimization**: FP16 training, gradient accumulation, LR scheduling  
- **Evaluation**: Exact Match (EM), F1 score  

---

## Testing
```bash
# Run all tests
pytest tests/ -v

# Run evaluation only
python -m tests.evaluate --model_path your-model --num_samples 5
```

---

## 📚 References
- Hugging Face Transformers  
- SQuAD Dataset  
- BERT Paper  
- Accelerate Library  

---

## 📄 License
Licensed under the **MIT License** – see `LICENSE`.

---

## 🙏 Acknowledgments
- Hugging Face for Transformers  
- Stanford NLP Group for SQuAD  
- Google Research for BERT  

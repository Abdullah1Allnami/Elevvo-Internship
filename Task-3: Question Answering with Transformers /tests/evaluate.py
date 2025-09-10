import torch
from transformers import pipeline
from src.models.get_model import load_fine_tuned_model

def evaluate_model_on_sample(model_path: str, dataset, sample_index: int = 0):
    """Evaluate the model on a single sample from the dataset"""
    model, tokenizer, device = load_fine_tuned_model(model_path)
    
    sample = dataset[sample_index]
    context = sample["context"]
    question = sample["question"]
    
    question_answerer = pipeline("question-answering", model=model_path, tokenizer=tokenizer)
    
    print("Question:", question)
    print("Context:", context[:200], "...")
    
    result = question_answerer(question=question, context=context)
    print("Predicted answer:", result["answer"])
    print("Ground truth:", sample["answers"])
    
    return result

def batch_evaluate(model_path: str, dataset, num_samples: int = 5):
    """Evaluate the model on multiple samples"""
    results = []
    for i in range(min(num_samples, len(dataset))):
        print(f"\n--- Sample {i+1} ---")
        result = evaluate_model_on_sample(model_path, dataset, i)
        results.append(result)
    
    return results
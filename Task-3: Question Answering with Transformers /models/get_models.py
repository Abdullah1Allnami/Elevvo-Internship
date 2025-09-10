from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

def get_model_and_tokenizer(model_checkpoint: str = "bert-base-cased", device: str = None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    return model, tokenizer, device

def load_fine_tuned_model(model_path: str, device: str = None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.to(device)
    return model, tokenizer, device

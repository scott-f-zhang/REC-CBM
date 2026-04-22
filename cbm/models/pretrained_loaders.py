import os
import torch
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    BertModel,
    BertTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    T5Tokenizer,
    T5EncoderModel,
    BartTokenizer,
    BartModel,
)
from typing import Tuple, Optional

# Disable Hugging Face Hub online API calls to avoid network errors
# This allows loading from cache without attempting to fetch additional_chat_templates
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def _load_from_pretrained(loader_class, model_name: str, **kwargs):
    """Helper function to load from pretrained with offline fallback.
    
    Tries to load normally first, then falls back to local_files_only=True
    if network errors occur.
    """
    try:
        return loader_class.from_pretrained(model_name, **kwargs)
    except Exception as e:
        # If network error and model might be cached, try loading from cache only
        if 'network' in str(e).lower() or 'connection' in str(e).lower() or 'unreachable' in str(e).lower():
            try:
                return loader_class.from_pretrained(model_name, local_files_only=True, **kwargs)
            except Exception:
                # Re-raise original error if local_files_only also fails
                raise e
        else:
            raise


def load_model_and_tokenizer(model_name: str) -> Tuple[object, object, Optional[int]]:
    """Load model and tokenizer from Hugging Face.
    
    Args:
        model_name: Name of the model to load
    
    Returns:
        Tuple of (model, tokenizer, hidden_size)
    
    Note:
        HF_HUB_OFFLINE is set to disable online API calls (e.g., additional_chat_templates)
        to avoid network errors when running on servers without internet access.
        Models will be loaded from cache if available. If network errors occur,
        the function will automatically fall back to local_files_only=True.
    """
    model_name_lower = model_name.lower()
    if model_name == 'roberta-base':
        tokenizer = _load_from_pretrained(RobertaTokenizer, model_name)
        model = _load_from_pretrained(RobertaModel, model_name)
        hidden_size = model.config.hidden_size
        return model, tokenizer, hidden_size
    if model_name == 'bert-base-uncased':
        tokenizer = _load_from_pretrained(BertTokenizer, model_name)
        model = _load_from_pretrained(BertModel, model_name)
        hidden_size = model.config.hidden_size
        return model, tokenizer, hidden_size
    if model_name == 'gpt2':
        tokenizer = _load_from_pretrained(GPT2Tokenizer, model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = _load_from_pretrained(GPT2Model, model_name)
        hidden_size = model.config.hidden_size
        return model, tokenizer, hidden_size
    if "t5" in model_name_lower:
        tokenizer = _load_from_pretrained(T5Tokenizer, model_name)
        model = _load_from_pretrained(T5EncoderModel, model_name)
        hidden_size = model.config.d_model
        return model, tokenizer, hidden_size
    if "bart" in model_name_lower:
        tokenizer = _load_from_pretrained(BartTokenizer, model_name)
        base_model = _load_from_pretrained(BartModel, model_name)
        model = base_model.get_encoder()
        hidden_size = base_model.config.d_model
        return model, tokenizer, hidden_size
    raise ValueError(f"Unsupported model_name: {model_name}")

# Import necessary libraries
import numpy as np
from evaluate import load
import torch
import re

from evaluate import load
rouge = load('rouge')
bleu = load('bleu')
meteor = load('meteor')
bertscore = load('bertscore')

def create_conversation(example):
    """
    Transform the dataset into a conversational format.
    The user provides the text, and the assistant provides the summary.
    """
    # Create a conversation with user and assistant roles
    messages = [
        {"role": "user", "content": example["text"]},  # User provides the text
        {"role": "assistant", "content": example["summary"]}  # Assistant provides the summary
    ]
    # Return the conversation as a dictionary
    return {"messages": messages}

def create_testing_conversation(example):
    """
    Transform the dataset into a conversational format.
    The user provides the text, and the assistant must output summary.
    """
    # Create a conversation with user and assistant roles
    messages = [
        {"role": "user", "content": example["text"]},  # User provides the text
        {"role": "assistant", "content": ""}  # Assistant must output the summary
    ]
    # Return the conversation as a dictionary
    return {"messages": messages}


def apply_chat_template(example, tokenizer):
    """ Apply the chat template to the dataset. """
    example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return example

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed. 
    Taken from https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/22
    """
    
    # Handle tuple logits (happens when the model is trained using LoRA)
    if isinstance(logits, tuple):
        logits = logits[1]          # logits[0] is the loss value and logits[1] are the logits used to compute loss
                                    # logits: (tensor(2.0426, device='cuda:0'), tensor([[[ 7.8750,  5.3750,  7.0938,  ..., -4.2500, -4.2500, -4.2500],
                                    #          [ 5.0938,  5.0625,  7.3750,  ..., -1.5312, -1.5312, -1.5312],
                                    #          [ 2.6562, -0.9609,  0.0728,  ..., -2.0312, -2.0312, -2.0312],
                                    #          ...,
                                    #          [ 4.1562,  1.4375, -3.6250,  ..., -2.1250, -2.1250, -2.1250],
                                    #          [ 3.7344, -1.6641, -3.8125,  ..., -1.9688, -1.9688, -1.9688],
                                    #          [ 8.1875, -1.2344, -1.6094,  ..., -3.0938, -3.0938, -3.0938]]],
                                    #        device='cuda:0'))

    # Proceed with argmax
    pred_ids = torch.argmax(logits, dim=-1)

    return pred_ids

@torch.no_grad()
def compute_metrics_causal_lm(eval_pred, tokenizer):
    """Compute ROUGE and BLEU scores for evaluation."""
    predictions, references = eval_pred

    # Clip token IDs to the valid range
    vocab_size = tokenizer.vocab_size

    def clip_token_ids(token_ids):
        """Clip token IDs to the valid range [0, vocab_size - 1]."""
        return [min(max(token_id, 0), vocab_size - 1) for token_id in token_ids]

    # Decode predictions and references
    decoded_preds = [
        tokenizer.decode(clip_token_ids(pred), skip_special_tokens=True)
        for pred in predictions
    ]
    decoded_refs = [
        tokenizer.decode(clip_token_ids(ref), skip_special_tokens=True)
        for ref in references
    ]
    
    # Clean summaries
    def clean_summary(text):
        special_tokens = ["<|im_end|>", "<|assistant|>", "<|user|>", "<|system|>"]
        for token in special_tokens:
            text = text.replace(token, "")
        return re.sub(r"\s+", " ", text).strip()
    
    pred_summaries = []
    for pred in decoded_preds:
        if "<|assistant|>" in pred:
            summary = pred.split("<|assistant|>")[-1].strip()
            summary = clean_summary(summary)
            pred_summaries.append(summary)
        else:
            summary = pred.strip()
            summary = clean_summary(summary)
            pred_summaries.append(summary)
            
    # apply the same to the references
    ref_summaries = []
    for ref in decoded_refs:
        if "<|assistant|>" in ref:
            summary = ref.split("<|assistant|>")[-1].strip()
            summary = clean_summary(summary)
            ref_summaries.append(summary)
        else:
            summary = ref.strip()
            summary = clean_summary(summary)
            ref_summaries.append(summary)
            
    # print(f'0 - ref_summaries[0]: {ref_summaries[0]}')
    
    # Convert to token IDs
    pred_token_ids = [tokenizer.encode(p, add_special_tokens=False) for p in pred_summaries]
    ref_token_ids = [tokenizer.encode(r, add_special_tokens=False) for r in ref_summaries]

    # Use the exact same metric function from training
    eval_pred = (pred_token_ids, ref_token_ids)
    
    predictions, references = eval_pred

    # Clip token IDs to the valid range
    vocab_size = tokenizer.vocab_size

    # Decode predictions and references in batches
    decoded_preds = tokenizer.batch_decode([clip_token_ids(pred) for pred in predictions], skip_special_tokens=True)
    decoded_refs = tokenizer.batch_decode([clip_token_ids(ref) for ref in references], skip_special_tokens=True)
    
    # Print decoded examples to inspect issues
    print(f'decoded_preds[0]: {decoded_preds[0]}')
    print(f'decoded_refs[0]: {decoded_refs[0]}')

    # Compute ROUGE and BLEU scores
    rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_refs, use_stemmer=True)
    bleu_results = bleu.compute(predictions=decoded_preds, references=decoded_refs)

    metrics = {key: rouge_results[key] * 100 for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}
    metrics["bleu"] = bleu_results["bleu"] * 100

    return metrics


@torch.no_grad()
def compute_metrics_seq2seq(eval_pred, tokenizer):
    preds, labels = eval_pred
    
    # Clip token IDs to valid range
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
    labels = np.clip(labels, 0, tokenizer.vocab_size - 1)
    
    # Ensure labels are not masked
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels directly using batch_decode
    text_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    print(f'text_preds[0]: {text_preds[0]}')
    print(f'text_labels[0]: {text_labels[0]}')
    
    # Compute metrics
    rouge_results = rouge.compute(predictions=text_preds, references=text_labels, use_stemmer=True)
    bleu_results = bleu.compute(predictions=text_preds, references=text_labels)
    
    return {
        "rouge1": rouge_results["rouge1"] * 100,
        "rouge2": rouge_results["rouge2"] * 100,
        "rougeL": rouge_results["rougeL"] * 100,
        "rougeLsum": rouge_results["rougeLsum"] * 100,
        "bleu": bleu_results["bleu"] * 100,
    }

def set_seed(seed):
    """ Sets the seed for reproducibility """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_trainable_params_info(model):
    """
    Prints the total and trainable parameters in the model, 
    along with the percentage reduction in trainable parameters.
    
    Parameters:
    - model: The PyTorch model (could be wrapped with LoRA).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reduction_percent = (1 - trainable_params / total_params) * 100

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Reduction in Trainable Parameters: {reduction_percent:.2f}%")
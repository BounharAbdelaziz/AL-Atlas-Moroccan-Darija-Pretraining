# Import necessary libraries
from transformers import Seq2SeqTrainer
from trl import SFTTrainer
# from evaluate import load
# rouge = load("rouge")
# bleu = load("bleu")

from utils import compute_metrics_causal_lm

from trl import SFTTrainer
import torch
from transformers import Seq2SeqTrainer

def evaluate_model(model, eval_dataset, tokenizer, batch_size, max_length, device):
    """Custom evaluation function computing ROUGE and BLEU."""
    import torch

    model.eval()
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)
    predictions, references = [], []

    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = torch.LongTensor(batch["input_ids"]).to(device)
            labels = torch.LongTensor(batch["labels"]).to(device)

            outputs = model.generate(inputs, max_length=max_length)
            predictions.extend(outputs.cpu().tolist())
            references.extend(labels.cpu().tolist())

    eval_pred = (predictions, references)
    return compute_metrics_causal_lm(eval_pred, tokenizer)

class MySFTTrainer(SFTTrainer):
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Override evaluation loop to include custom metrics."""
        metrics = evaluate_model(
            model=self.model,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.args.per_device_eval_batch_size,
            max_length=self.args.max_length if hasattr(self.args, "max_length") else 2048,
            device=self.args.device,
        )

        parent_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        all_metrics = {**parent_output.metrics, **metrics}

        return type(parent_output)(
            parent_output.predictions, parent_output.label_ids, all_metrics, parent_output.num_samples
        )


class MySeq2SeqTrainer(Seq2SeqTrainer):
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Override evaluation loop to include custom metrics."""
        metrics = evaluate_model(
            model=self.model,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.args.per_device_eval_batch_size,
            max_length=self.args.max_length if hasattr(self.args, "max_length") else 2048,
            device=self.args.device,
        )

        parent_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        all_metrics = {**parent_output.metrics, **metrics}

        return type(parent_output)(
            parent_output.predictions, parent_output.label_ids, all_metrics, parent_output.num_samples
        )

# src/models/robertaabsa.py
import torch
import torch.nn as nn
from transformers import RobertaModel


class RobertaABSA(nn.Module):
    """Standard RoBERTa classifier: [CLS] review [SEP] aspect [SEP] → sentiment."""

    def __init__(self, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))

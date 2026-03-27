# src/models/bert_absa.py
import torch
import torch.nn as nn
from transformers import BertModel


class BertABSA(nn.Module):
    """Standard BERT classifier: [CLS] review [SEP] aspect [SEP] → sentiment."""

    def __init__(self, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(self.dropout(cls))

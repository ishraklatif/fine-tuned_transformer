import math
import torch
import torch.nn as nn
from transformers import AutoModel


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks for the custom Transformer
# ──────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention.
    Each head learns to focus on a different sub-space of the embedding,
    which is then concatenated and projected back to d_model.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V)

    def split_heads(self, x):
        B, L, _ = x.size()
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]

    def combine_heads(self, x):
        B, _, L, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)  # [B, L, d_model]

    def forward(self, Q, K, V):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V)
        return self.W_o(self.combine_heads(attn_output))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Custom Transformer Classifier
# ──────────────────────────────────────────────────────────────────────────────

class TransformerClassifier(nn.Module):
    """
    Transformer encoder stack built from scratch for question classification.
    Requires a DataManager instance to infer vocab_size, num_classes, max_seq_len.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout_rate: float = 0.2,
        data_manager=None,
    ):
        super().__init__()
        assert data_manager is not None, "data_manager is required"
        self.vocab_size = data_manager.vocab_size
        self.num_classes = data_manager.num_classes
        self.embed_dim = embed_dim
        self.max_seq_len = data_manager.max_seq_len
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.build()

    def build(self):
        self.token_embed = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(self.embed_dim, self.max_seq_len)
        self.encoders = nn.ModuleList([
            EncoderLayer(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate)
            for _ in range(self.num_layers)
        ])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        """x: LongTensor [B, L] of token IDs. Returns logits [B, num_classes]."""
        x_embed = self.positional_encoding(self.token_embed(x))
        h = x_embed
        for layer in self.encoders:
            h = layer(h)
        mask = (x != 0).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))


# ──────────────────────────────────────────────────────────────────────────────
# BERT Prefix Tuning Classifier
# ──────────────────────────────────────────────────────────────────────────────

class PrefixTuningForClassification(nn.Module):
    """
    Frozen BERT encoder with a trainable soft prefix prepended to the input
    embeddings, plus a linear classification head.
    Only the prefix embeddings and the classifier head are trainable.
    """

    def __init__(self, model_name: str, prefix_length=None, data_manager=None):
        super().__init__()
        assert data_manager is not None, "data_manager is required"

        self.model = AutoModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False

        self.hidden_size = self.model.config.hidden_size
        self.num_classes = data_manager.num_classes
        self.prefix_length = prefix_length

        if self.prefix_length is not None and self.prefix_length > 0:
            self.prefix_embeddings = nn.Parameter(
                torch.zeros(self.prefix_length, self.hidden_size)
            )
            nn.init.normal_(self.prefix_embeddings, mean=0.0, std=0.02)
            self.prefix = self.prefix_embeddings  # alias

        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def _masked_mean(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, input_ids, attention_mask):
        B = input_ids.shape[0]

        if self.prefix_length is not None and self.prefix_length > 0:
            word_embeds = self.model.embeddings.word_embeddings(input_ids)
            prefix_batch = self.prefix_embeddings.unsqueeze(0).expand(B, -1, -1)
            inputs_embeds = torch.cat([prefix_batch, word_embeds], dim=1)
            prefix_mask = torch.ones(
                B, self.prefix_length,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attn_mask_ext = torch.cat([prefix_mask, attention_mask], dim=1)
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask_ext)
            pooled = self._masked_mean(outputs.last_hidden_state, attn_mask_ext)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self._masked_mean(outputs.last_hidden_state, attention_mask)

        return self.classifier(pooled)

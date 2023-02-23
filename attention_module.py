import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    ""
    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Define query, key, and value linear transformations
        self.q_linear = nn.Linear(input_dim, embed_dim)
        self.k_linear = nn.Linear(input_dim, embed_dim)
        self.v_linear = nn.Linear(input_dim, embed_dim)

        # Define multi-head linear transformation
        self.multihead_linear = nn.Linear(embed_dim, embed_dim)

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask=None):
        # Apply query, key, and value linear transformations
        q = self.q_linear(inputs)  # shape: (batch_size, seq_len, embed_dim)
        k = self.k_linear(inputs)  # shape: (batch_size, seq_len, embed_dim)
        v = self.v_linear(inputs)  # shape: (batch_size, seq_len, embed_dim)

        # Split the embeddings into num_heads pieces
        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, embed_dim // num_heads)
        k = k.view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, embed_dim // num_heads)
        v = v.view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, embed_dim // num_heads)

        # Compute the attention scores and apply the mask if provided
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_dim // self.num_heads, dtype=torch.float32))  # shape: (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply the attention scores to the value embeddings and concatenate the results
        attention = torch.softmax(scores, dim=-1)
        x = torch.matmul(self.dropout(attention), v)  # shape: (batch_size, num_heads, seq_len, embed_dim // num_heads)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # shape: (batch_size, seq_len, embed_dim)

        # Apply the multi-head linear transformation
        x = self.multihead_linear(x)  # shape: (batch_size, seq_len, embed_dim)

        return x

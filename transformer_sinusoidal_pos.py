import torch
import math
import torch.nn as nn
class MultiheadAttention(nn.Module):
    ""
    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Define query, key, and value linear transformations
        self.q_linear = nn.Linear(input_dim, embed_dim, bias= False)
        self.k_linear = nn.Linear(input_dim, embed_dim, bias= False)
        self.v_linear = nn.Linear(input_dim, embed_dim, bias= False)

        # Define multi-head linear transformation
        self.multihead_linear = nn.Linear(embed_dim, embed_dim, bias= False)

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs_Q,inputs_K,inputs_V, mask=None):
        # Apply query, key, and value linear transformations
        q = self.q_linear(inputs_Q)  # shape: (batch_size, seq_len, embed_dim)
        k = self.k_linear(inputs_K)  # shape: (batch_size, seq_len, embed_dim)
        v = self.v_linear(inputs_V)  # shape: (batch_size, seq_len, embed_dim)

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

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size = 65, num_heads = 1, feedforward_size = 20, max_sequence_length=4000):
        self.attention = MultiheadAttention(input_dim = hidden_size, embed_dim = hidden_size, num_heads = num_heads)
        self.layer_norm = nn.LayerNorm()
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.ReLU(),
            nn.Linear(feedforward_size, hidden_size)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Sinusoidal Positional Encoding
        self.positional_encoding = self.get_sinusoidal_encoding_table(max_sequence_length=max_sequence_length, hidden_size=hidden_size)

    def get_sinusoidal_encoding(self, max_sequence_length, hidden_size):

        position = torch.arrange(0,max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arrange(0,hidden_size,2) * -(match.log(10000.0)))
        sinusoidal_matrix = torch.zeros((max_sequence_length,hidden_size))
        sinusoidal_matrix[:,0::2] =  torch.sin(position * div_term)
        sinusoidal_matrix[:,1::2] =  torch.cos(position * div_term)
        return sinusoidal_matrix.unsqueeze(0)

    def forward(self, inputs):
        # inputs: batch_size x sequence_length x hidden_size
        
        # positional encoding
        seq_length = inputs.size(1)
        pos_enc = self.positional_encoding[:, :seq_length, :].cuda()
        inputs = inputs + pos_enc
        
        # self-attention
        attention_output, _ = self.attention(inputs, inputs, inputs)
        # residual connection and layer normalization
        attention_output = self.layer_norm1(inputs + attention_output)
        
        # feedforward network
        ff_output = self.feedforward(attention_output)
        # residual connection and layer normalization
        output = self.layer_norm2(attention_output + ff_output)
        
        return output

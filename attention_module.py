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
        self.q_linear = nn.Linear(input_dim, embed_dim, bias= False)
        self.k_linear = nn.Linear(input_dim, embed_dim, bias= False)
        self.v_linear = nn.Linear(input_dim, embed_dim, bias= False)

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

   class Attention_block(nn.Module):
    "Attention block"
    def __init__(self, hidden_dim, num_heads, norm_dim):
        super(Attention_block,self).__init__()
        self.hidden_dim = hidden_dim
        self.norm_dim = norm_dim
        self.mh_att = nn.MultiheadAttention(embed_dim = hidden_dim,
                                             num_heads = num_heads,
                                             batch_first =True)
        self.batch_norm = nn.BatchNorm1d(norm_dim)
        self.activation = nn.ReLU()

    def forward(self,Q,K,V):
        attention_output, _ = self.mh_att(Q,K,V)
        batch, length, hid_dim =  attention_output.size()
        # Reshape input tensor to (batch_size*hid_dim, length)
        attention_output = attention_output.transpose(1, 2).contiguous().view(-1, self.norm_dim)
        attention_output = self.batch_norm(attention_output)
        attention_output = attention_output.view(batch, hid_dim, self.norm_dim)

        return self.activation(attention_output.transpose(1, 2))
    

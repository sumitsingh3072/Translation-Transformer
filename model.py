import torch
import torch.nn as nn
import math

# Main Input Embeddings

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int , d_model : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)
    
## Positional Encoding 

class PositionalEncoding(nn.Module):

    def __init__(self, d_model , seq_len:int , dropout: float) -> None :
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # create matrix of shape (seq_len, d_model)

        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len,1)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        # Apply the sin to even and cos to odd positions

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0) # (1 , seq_len , d_model)
        self.register_buffer('pe', pe)
    
    def forward(self,x) :
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x) 

## Layer Normalization

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) ## Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) ## Added
    def forward(self,x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

## Feed Forward NN

class FeedForwardLock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) ## W1 and B1 
        self.dropout = nn.Dropout(dropout) ## Dropout
        self.linear2 = nn.Linear(d_ff, d_model) ## W2 and B2

    def forward(self, x):
        ## (Batch, Siq_len, d_model) ---> (Batch, Siq_len, d_ff) ---> (Batch, Siq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
## Multi Head Attention

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 , "d_model is not divisble by h"
        self.d_k = d_model // h ## computes the dimension of each head in the multi-head attention block
        self.w_q = nn.Linear(d_model,d_model) # WQ
        self.w_k = nn.Linear(d_model,d_model)  # WK
        self.w_v = nn.Linear(d_model,d_model) # WV
        self.w_o = nn.Linear(d_model,d_model) # WO
        self.droput = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key, value, mask , dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k) ## (Batch, h, seq_len, d_k) @ (Batch, h, d_k, seq_len) --> (Batch, h, seq_len, seq_len)

    def forward(self, q, k, v, mask): 
        query = self.w_q(q) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k)  # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # (Batch,seq_len , d_model) --> (Batch, Seq_Len, h, d_k) ---> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1],self.h , self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h , self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1] , self.h , self.d_k).transpose(1,2)

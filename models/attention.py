import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask = None):
    d_k = Q.size(-1)
    
    # QK/d
    score = Q @ K.transpose(-2, -1) / d_k ** 0.5
    
    if mask is not None:
        score = score.masked_fill(mask, float('-inf'))
    
    score = F.softmax(score, dim=-1)

    # V
    score = score @ V

    return score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, head=8):
        super().__init__()

        self.d_model = d_model
        self.nhead = head
        self.d_k = d_model // head

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        B, S, _ = Q.size()
        _, S_K, _ = K.size()

        Q = Q.view(B, S, self.nhead, self.d_k)
        K = K.view(B, S_K, self.nhead, self.d_k)
        V = V.view(B, S_K, self.nhead, self.d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        O = scaled_dot_product_attention(Q, K, V, mask)
        O = O.transpose(1, 2)
        O = O.reshape(B, S, self.d_model)
        O = self.W_O(O)

        return O
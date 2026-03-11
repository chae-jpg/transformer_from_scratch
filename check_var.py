import torch
import torch.nn as nn
from models.attention import MultiHeadAttention

torch.manual_seed(42)

def check_variance(init_fn=None):
    B, S, D = 4, 16, 512
    # simulate embedding
    emb = nn.Embedding(1000, D)
    if init_fn:
        init_fn(emb.weight)
    
    x = torch.randint(0, 1000, (B, S))
    x = emb(x) * (D ** 0.5)
    
    attn = MultiHeadAttention(d_model=D, d_k=64, d_v=64, head=8)
    if init_fn:
        for p in attn.parameters():
            if p.dim() > 1:
                init_fn(p)
    
    Q = attn.W_Q(x)
    K = attn.W_K(x)
    Q = Q.view(B, S, 8, 64).transpose(1, 2)
    K = K.view(B, S, 8, 64).transpose(1, 2)
    
    score = Q @ K.transpose(-2, -1) / (64 ** 0.5)
    print("Variance of score matrix before Softmax:", score.var().item())

print("--- Default PyTorch Init ---")
check_variance()

print("--- Xavier Uniform Init ---")
check_variance(nn.init.xavier_uniform_)

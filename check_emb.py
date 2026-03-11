import torch
import torch.nn as nn
import math

class Test(nn.Module):
    def __init__(self, vocab=1000, d_model=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        
    def forward(self, x):
        return self.embedding(x)

t = Test()
emb_out = t(torch.tensor([[1,2,3]]))
print("Emb std:", emb_out.std().item())
print("Scaled Emb std:", (emb_out * math.sqrt(512)).std().item())

import torch
import torch.nn as nn
from models.model import Transformer

torch.manual_seed(42)
VOCAB_SIZE = 58100 

model = Transformer(
    src_vocab = VOCAB_SIZE, tgt_vocab = VOCAB_SIZE, max_len = 256, d_model = 512, nhead = 8, num_layers = 6, d_ff = 2048, dropout = 0.1
)

src = torch.randint(0, VOCAB_SIZE, (4, 10))
tgt = torch.randint(0, VOCAB_SIZE, (4, 10))
src_mask = torch.zeros((4, 1, 1, 10)).bool()
tgt_mask = torch.zeros((4, 1, 1, 10)).bool()

outputs = model(src, tgt, src_mask, tgt_mask)
loss = outputs.sum()
loss.backward()

enc_grad = model.enc.blocks[-1].attn.W_Q.weight.grad
if enc_grad is None or enc_grad.abs().max().item() == 0:
    print("ENCODER GRADS ARE ZERO - BUG IN ATTENTION / GRADIENT FLOW!")
else:
    print(f"Encoder gradient max: {enc_grad.abs().max().item()}")
    
dec_cross_grad = model.dec.blocks[0].attn2.W_Q.weight.grad
if dec_cross_grad is None or dec_cross_grad.abs().max().item() == 0:
    print("CROSS-ATTENTON GRADS ARE ZERO - BUG IN CROSS ATTENTION!")
else:
    print(f"Cross attention gradient max: {dec_cross_grad.abs().max().item()}")

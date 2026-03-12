from . import attention
import torch
import torch.nn.functional as F
import torch.nn as nn

def pos_encoder(emb, max_len, d_model):
    # emb: max_len x d_model
    
    pos = torch.arange(max_len).unsqueeze(1) # max_len x 1
    i = torch.arange(0, d_model, 2)
    div = 10000 ** (i / d_model)

    emb[:, 0::2] = torch.sin(pos / div)
    emb[:, 1::2] = torch.cos(pos / div)

    return emb

class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.attn = attention.MultiHeadAttention(d_model=d_model, head=nhead)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, pad_mask):
        x2 = self.attn(x, x, x, pad_mask)
        x = x + self.dropout(x2)
        x = self.ln1(x)

        x2 = self.ff1(x)
        x2 = self.act(x2)
        x2 = self.dropout(x2)
        x2 = self.ff2(x2)
        x = x + self.dropout(x2)
        x = self.ln2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=512, nhead=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, nhead, d_ff, dropout) for _ in range(num_layers)
        ])
        pos = torch.zeros(max_len, d_model)
        pos = pos_encoder(pos, max_len, d_model)
        self.register_buffer("pos", pos)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    def forward(self, x, pad_mask):
        x = self.dropout(self.embedding(x)* (self.d_model ** 0.5) + self.pos[:x.size(1), :])
        for block in self.blocks:
            x = block(x, pad_mask)
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, max_len, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.attn1 = attention.MultiHeadAttention(d_model=d_model, head=nhead)
        self.attn2 = attention.MultiHeadAttention(d_model=d_model, head=nhead)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.act = nn.ReLU()

        self.register_buffer("mask", torch.triu(torch.ones(max_len, max_len), diagonal=1).bool())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, src_mask, tgt_mask):
        # x: batch_size x seq_len
        causal = self.mask[:x.size(1), :x.size(1)].unsqueeze(0).unsqueeze(0)
        comb = causal | tgt_mask
        # masked multi-head attention
        x2 = self.attn1(x, x, x, comb)
        x = x + self.dropout(x2)
        x = self.ln1(x)

        # cross-attention
        x2 = self.attn2(x, y, y, src_mask)
        x = x + self.dropout(x2)
        x = self.ln2(x)

        # ff layer
        x2 = self.ff1(x)
        x2 = self.act(x2)
        x2 = self.dropout(x2)
        x2 = self.ff2(x2)
        x = x + self.dropout(x2)
        x = self.ln3(x)
            
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=512, nhead=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(max_len, d_model, nhead, d_ff, dropout) for _ in range(num_layers)
        ])

        pos = torch.zeros(max_len, d_model)
        pos = pos_encoder(pos, max_len, d_model)
        self.register_buffer("pos", pos)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    def forward(self, x, y, src_mask, tgt_mask):
        x = self.dropout(self.embedding(x) * (self.d_model ** 0.5)+ self.pos[:x.size(1), :])
        for block in self.blocks:
            x = block(x, y, src_mask, tgt_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, max_len, d_model=512, nhead=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.enc = Encoder(src_vocab, max_len, d_model, nhead, num_layers, d_ff, dropout)
        self.dec = Decoder(tgt_vocab, max_len, d_model, nhead, num_layers, d_ff, dropout)
        self.fc = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.enc(src, src_mask)
        dec_out = self.dec(tgt, enc_out, src_mask, tgt_mask)
        return self.fc(dec_out)
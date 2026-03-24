"""
오버피팅 테스트: 10문장을 완벽히 외울 수 있는지 확인.
- loss → 0 근처, BLEU → 높음: 아키텍처 정상
- loss 안 내려감 or BLEU 낮음: 코드에 버그 있음
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from models.model import Transformer
from sacrebleu import corpus_bleu
from datasets import load_dataset

# ── 설정 ──
MAX_LEN = 64
BATCH_SIZE = 10
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 6
D_FF = 2048
DROPOUT = 0.0  # 오버피팅 테스트니까 dropout 끔
NUM_STEPS = 500
PRINT_EVERY = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 토크나이저 ──
tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
tok.add_special_tokens({'bos_token': '[BOS]', 'eos_token': '[EOS]', 'pad_token': '[PAD]'})
PAD_ID = tok.pad_token_id
BOS = tok.bos_token_id
EOS = tok.eos_token_id
VOCAB_SIZE = len(tok)

# ── 데이터: 딱 10문장 ──
raw = load_dataset("wmt19", "de-en", split="validation[:10]")

X, Y_in, Y_out = [], [], []
src_texts, tgt_texts = [], []
for item in raw:
    src = item['translation']['de']
    tgt = item['translation']['en']
    src_texts.append(src)
    tgt_texts.append(tgt)

    s = tok.encode(src, add_special_tokens=False)
    X.append(torch.tensor([BOS] + s[:MAX_LEN-2] + [EOS]))

    t = tok.encode(tgt, add_special_tokens=False)
    y = [BOS] + t[:MAX_LEN-2] + [EOS]
    Y_in.append(torch.tensor(y[:-1]))
    Y_out.append(torch.tensor(y[1:]))

def collate(batch):
    x, yi, yo = zip(*batch)
    x = pad_sequence(x, batch_first=True, padding_value=PAD_ID)
    yi = pad_sequence(yi, batch_first=True, padding_value=PAD_ID)
    yo = pad_sequence(yo, batch_first=True, padding_value=PAD_ID)
    return x, yi, yo

class DS(Dataset):
    def __len__(self): return len(X)
    def __getitem__(self, i): return X[i], Y_in[i], Y_out[i]

loader = DataLoader(DS(), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

def make_pad_mask(seq, pad_id):
    return (seq == pad_id).unsqueeze(1).unsqueeze(2)

# ── 모델 ──
model = Transformer(
    src_vocab=VOCAB_SIZE, tgt_vocab=VOCAB_SIZE, max_len=MAX_LEN,
    d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, d_ff=D_FF, dropout=DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

# ── inference ──
def inference(text):
    model.eval()
    with torch.no_grad():
        s = tok.encode(text, add_special_tokens=False)
        src = torch.tensor([[BOS] + s[:MAX_LEN-2] + [EOS]]).to(device)
        src_mask = make_pad_mask(src, PAD_ID)
        enc_out = model.enc(src, src_mask)

        tgt = torch.tensor([[BOS]]).to(device)
        for _ in range(MAX_LEN):
            tgt_mask = make_pad_mask(tgt, PAD_ID)
            dec_out = model.dec(tgt, enc_out, src_mask, tgt_mask)
            logits = model.fc(dec_out[:, -1, :])
            nxt = logits.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, nxt], dim=1)
            if nxt.item() == EOS:
                break
        return tok.decode(tgt.squeeze().tolist(), skip_special_tokens=True)

# ── 학습 ──
print(f"Device: {device}")
print(f"Vocab: {VOCAB_SIZE}, Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"10문장 오버피팅 테스트 시작\n")

model.train()
for step in range(1, NUM_STEPS + 1):
    for x, y_in, y_out in loader:
        x, y_in, y_out = x.to(device), y_in.to(device), y_out.to(device)
        src_mask = make_pad_mask(x, PAD_ID)
        tgt_mask = make_pad_mask(y_in, PAD_ID)

        out = model(x, y_in, src_mask, tgt_mask)
        loss = criterion(out.view(-1, VOCAB_SIZE), y_out.view(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    if step % PRINT_EVERY == 0 or step == 1:
        # BLEU 측정
        preds = []
        for s in src_texts:
            preds.append(inference(s))
        model.train()
        bleu = corpus_bleu(preds, [tgt_texts]).score

        print(f"[Step {step:3d}] loss={loss.item():.4f}  BLEU={bleu:.1f}")
        # 첫 3문장 샘플 출력
        for i in range(min(3, len(preds))):
            print(f"  SRC: {src_texts[i][:80]}")
            print(f"  PRD: {preds[i][:80]}")
            print(f"  REF: {tgt_texts[i][:80]}")
            print()

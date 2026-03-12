import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from models.model import Transformer
from tqdm import tqdm
import math
from sacrebleu import corpus_bleu
import wandb

# ── 전역 상수 ──────────────────────────────────────────
MAX_LEN    = 256
BATCH_SIZE = 64
D_MODEL    = 512
NHEAD      = 8
NUM_LAYERS = 6
D_FF       = 2048
DROPOUT    = 0.1
LR         = 0.5
NUM_EPOCHS = 1
MAX_SAMPLES = 5_000_000
PAD_ID     = None  # 토크나이저 로드 후 설정

# ── Dataset ────────────────────────────────────────────
class TransDataset(IterableDataset):
    def __init__(self, data, tok, max_len, max_samples=None):
        self.data = data
        self.tok = tok
        self.max_len = max_len
        self.max_samples = max_samples
        self.BOS = tok.bos_token_id
        self.EOS = tok.eos_token_id

    def __iter__(self):
        count = 0
        for item in self.data:
            if self.max_samples and count >= self.max_samples:
                return
            src_enc = self.tok.encode(item['translation']['de'], add_special_tokens=False)
            tgt_enc = self.tok.encode(item['translation']['en'], add_special_tokens=False)
            x   = [self.BOS] + src_enc[:self.max_len-2] + [self.EOS]
            y   = [self.BOS] + tgt_enc[:self.max_len-2] + [self.EOS]
            yield x, y[:-1], y[1:]
            count += 1

# ── collate ────────────────────────────────────────────
def make_collate_fn(pad_id):
    def collate_fn(batch):
        x, y_in, y_out = zip(*batch)
        x     = pad_sequence([torch.tensor(s) for s in x],     batch_first=True, padding_value=pad_id)
        y_in  = pad_sequence([torch.tensor(s) for s in y_in],  batch_first=True, padding_value=pad_id)
        y_out = pad_sequence([torch.tensor(s) for s in y_out], batch_first=True, padding_value=pad_id)
        return x, y_in, y_out
    return collate_fn

def make_pad_mask(seq, pad_id):
    return (seq == pad_id).unsqueeze(1).unsqueeze(2)

# ── Scheduler ──────────────────────────────────────────
class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * self.warmup_steps ** -1.5
        )
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.optimizer.step()

# ── inference ──────────────────────────────────────────
def inference(text, model, tok, max_len, device):
    model.eval()
    with torch.no_grad():
        src_enc = tok.encode(text, add_special_tokens=False)
        src = [tok.bos_token_id] + src_enc[:max_len-2] + [tok.eos_token_id]
        src = torch.tensor(src).unsqueeze(0).to(device)
        src_mask = make_pad_mask(src, tok.pad_token_id)
        enc_out  = model.enc(src, src_mask)
        tgt = torch.tensor([[tok.bos_token_id]]).to(device)
        for _ in range(max_len):
            tgt_mask = make_pad_mask(tgt, tok.pad_token_id)
            dec_out  = model.dec(tgt, enc_out, src_mask, tgt_mask)
            selected = model.fc(dec_out[:, -1, :]).argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, selected], dim=1)
            if selected.item() == tok.eos_token_id:
                break
    return tok.decode(tgt.squeeze().tolist(), skip_special_tokens=True)

# ── main ───────────────────────────────────────────────
if __name__ == "__main__":
    # 토크나이저
    tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")  # 방향 수정
    PAD_ID = tok.pad_token_id

    # 데이터
    train_raw = load_dataset("wmt19", "de-en", split="train", streaming=True)
    train_raw = train_raw.shuffle(seed=42, buffer_size=10000)
    val       = load_dataset("wmt19", "de-en", split="validation")
    split     = val.train_test_split(test_size=0.5, seed=42)
    dev_raw, test_raw = split["train"], split["test"]
    collate_fn = make_collate_fn(PAD_ID)
    trainloader = DataLoader(
        TransDataset(train_raw, tok, MAX_LEN, max_samples=MAX_SAMPLES),
        batch_size=BATCH_SIZE, collate_fn=collate_fn,pin_memory=True
    )
    devloader = DataLoader(
        TransDataset(dev_raw, tok, MAX_LEN),
        batch_size=BATCH_SIZE, collate_fn=collate_fn
    )
    testloader = DataLoader(
        TransDataset(test_raw, tok, MAX_LEN),
        batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    VOCAB_SIZE = len(tok)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(
        src_vocab=VOCAB_SIZE, tgt_vocab=VOCAB_SIZE, max_len=MAX_LEN,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, d_ff=D_FF, dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, D_MODEL, warmup_steps=4000)

    wandb.init(project="transformer-translation", config={
        "lr": LR, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
        "d_model": D_MODEL, "max_samples": MAX_SAMPLES
    })

    min_loss  = 1e9
    save_interval = 500
    current_step  = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(trainloader))  # total 없음 — streaming이라 모름

        for i, (x, y_in, y_out) in pbar:
            x, y_in, y_out = x.to(device), y_in.to(device), y_out.to(device)
            src_mask = make_pad_mask(x, PAD_ID)
            tgt_mask = make_pad_mask(y_in, PAD_ID)

            outputs = model(x, y_in, src_mask, tgt_mask)
            loss    = criterion(outputs.view(-1, VOCAB_SIZE), y_out.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            scheduler.step()

            total_loss   += loss.item()
            current_step += 1

            wandb.log({
                "train/loss": loss.item(),
                "train/avg_loss": total_loss / (i+1),
                "lr": optimizer.param_groups[0]['lr'],
                "step": current_step
            })

            if current_step % save_interval == 0:
                avg = total_loss / (i+1)
                if avg < min_loss:
                    min_loss = avg
                    torch.save(model.state_dict(), 'best_model_step.pth')
                    print(f"\n[Step {current_step}] Best Loss: {min_loss:.4f} saved")

            if i % 100 == 0:
                pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")

        # validation
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for x, y_in, y_out in tqdm(devloader):
                x, y_in, y_out = x.to(device), y_in.to(device), y_out.to(device)
                src_mask = make_pad_mask(x, PAD_ID)
                tgt_mask = make_pad_mask(y_in, PAD_ID)
                outputs  = model(x, y_in, src_mask, tgt_mask)
                loss     = criterion(outputs.view(-1, VOCAB_SIZE), y_out.view(-1))
                eval_loss += loss.item()
        avg_eval_loss = eval_loss / len(devloader)
        print(f"Epoch [{epoch+1}] Val Loss: {avg_eval_loss:.4f}")
        wandb.log({"val/loss": avg_eval_loss})

        if avg_eval_loss < min_loss:
            min_loss = avg_eval_loss
            torch.save(model.state_dict(), 'weights.pth')

    # 평가
    model.load_state_dict(torch.load('weights.pth'))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y_in, y_out in tqdm(testloader):
            x, y_in, y_out = x.to(device), y_in.to(device), y_out.to(device)
            outputs  = model(x, y_in, make_pad_mask(x, PAD_ID), make_pad_mask(y_in, PAD_ID))
            test_loss += criterion(outputs.view(-1, VOCAB_SIZE), y_out.view(-1)).item()
    perplexity = math.exp(test_loss / len(testloader))

    pred, target = [], []
    for item in tqdm(test_raw.select(range(10))):
        pred.append(inference(item['translation']['de'], model, tok, MAX_LEN, device))
        target.append(item['translation']['en'])

    bleu = corpus_bleu(pred, [target]).score
    print(f"Perplexity: {perplexity:.2f}, BLEU: {bleu:.2f}")
    for i in range(min(5, len(pred))):
        print(f"[{i}] PRD: {pred[i]}")
        print(f"    REF: {target[i]}")
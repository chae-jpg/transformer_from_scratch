import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from models.model import Transformer  # 본인의 모델 경로 확인
from tqdm import tqdm
import torch.optim as optim
import math
import wandb
from sacrebleu import corpus_bleu

# 1. 스트리밍 데이터를 위한 IterableDataset 정의
class StreamingTransDataset(IterableDataset):
    def __init__(self, dataset, tok, max_len):
        self.dataset = dataset
        self.tok = tok
        self.max_len = max_len
        self.BOS = tok.bos_token_id
        self.EOS = tok.eos_token_id

    def __iter__(self):
        for item in self.dataset:
            src = item['translation']['de']
            tgt = item['translation']['en']
            
            # 소스 인코딩
            src_enc = self.tok.encode(src, add_special_tokens=False)
            x = [self.BOS] + src_enc[:self.max_len-2] + [self.EOS]

            # 타겟 인코딩 (Input용/Output용 분리)
            tgt_enc = self.tok.encode(tgt, add_special_tokens=False)
            y = [self.BOS] + tgt_enc[:self.max_len-2] + [self.EOS]
            
            yield torch.tensor(x), torch.tensor(y[:-1]), torch.tensor(y[1:])

# 2. 배치 패딩을 위한 collate_fn
def collate_fn(batch):
    x, y_in, y_out = zip(*batch)
    x = pad_sequence(x, batch_first=True, padding_value=PAD_ID)
    y_in = pad_sequence(y_in, batch_first=True, padding_value=PAD_ID)
    y_out = pad_sequence(y_out, batch_first=True, padding_value=PAD_ID)
    return x, y_in, y_out

# 3. 패딩 마스크 생성
def make_pad_mask(seq, pad_id):
    return (seq == pad_id).unsqueeze(1).unsqueeze(2)

# 4. 스케줄러 정의
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

# 5. 추론(Inference) 함수
def inference(text, model, tok, max_len, device):
    model.eval()
    with torch.no_grad():
        bos_id = tok.bos_token_id
        eos_id = tok.eos_token_id
        src_enc = tok.encode(text, add_special_tokens=False)
        src = [bos_id] + src_enc[:max_len-2] + [eos_id]
        src = torch.tensor(src).unsqueeze(0).to(device)
        
        src_mask = make_pad_mask(src, tok.pad_token_id)
        enc_out = model.enc(src, src_mask)
        tgt = torch.tensor([[bos_id]]).to(device)

        for _ in range(max_len):
            tgt_mask = make_pad_mask(tgt, tok.pad_token_id)
            dec_out = model.dec(tgt, enc_out, src_mask, tgt_mask)
            last = model.fc(dec_out[:, -1, :])
            selected = last.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, selected], dim=1)
            if selected.item() == eos_id:
                break
        
        return tok.decode(tgt.squeeze().tolist(), skip_special_tokens=True)

if __name__ == "__main__":
    # 설정값
    MAX_LEN = 256
    BATCH_SIZE = 64
    D_MODEL = 512
    NHEAD = 8
    NUM_LAYERS = 6
    D_FF = 2048
    DROPOUT = 0.1
    WARMUP_STEPS = 4000
    SAVE_INTERVAL = 1000 # 1000 스텝마다 검증 및 저장
    TOTAL_SAMPLES = 5_000_000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 토크나이저 로드
    tok = AutoTokenizer.from_pretrained("t5-base")
    special_tokens_dict = {'bos_token': '[BOS]', 'eos_token': '[EOS]', 'pad_token': '[PAD]'}
    tok.add_special_tokens(special_tokens_dict)
    PAD_ID = tok.pad_token_id

    # 데이터 로드 (Validation/Test는 작으니까 기존 방식 유지)
    print("Loading Validation/Test data...")
    test_raw_full = load_dataset("wmt19", "de-en", split="validation")
    split = test_raw_full.train_test_split(test_size=0.5, seed=42)
    dev_raw, test_raw = split["train"], split["test"]

    # Training 데이터는 Streaming 방식으로 로드 (핵심!)
    print("Loading Training data (Streaming)...")
    train_stream = load_dataset("wmt19", "de-en", split="train", streaming=True)
    # buffer_size 만큼 쌓아두고 랜덤하게 섞음 (데이터 전체를 골고루 먹기 위함)
    train_stream = train_stream.shuffle(seed=42, buffer_size=10000)
    
    train_dataset = StreamingTransDataset(train_stream, tok, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Validation 로더 준비 (작으니까 미리 인코딩)
    # (기존 set_and_loader 기능을 간단히 구현)
    def prep_eval_loader(raw_data, tok, batch_size):
        X, Y_in, Y_out = [], [], []
        for item in raw_data:
            s, t = item['translation']['de'], item['translation']['en']
            s_e = [tok.bos_token_id] + tok.encode(s, add_special_tokens=False)[:MAX_LEN-2] + [tok.eos_token_id]
            t_e = [tok.bos_token_id] + tok.encode(t, add_special_tokens=False)[:MAX_LEN-2] + [tok.eos_token_id]
            X.append(torch.tensor(s_e))
            Y_in.append(torch.tensor(t_e[:-1]))
            Y_out.append(torch.tensor(t_e[1:]))
        
        # 간단한 Dataset 클래스 대용
        class SimpleDS(torch.utils.data.Dataset):
            def __init__(self, x, yi, yo): self.x, self.yi, self.yo = x, yi, yo
            def __len__(self): return len(self.x)
            def __getitem__(self, i): return self.x[i], self.yi[i], self.yo[i]
            
        return DataLoader(SimpleDS(X, Y_in, Y_out), batch_size=batch_size, collate_fn=collate_fn)

    dev_loader = prep_eval_loader(dev_raw, tok, BATCH_SIZE)
    test_loader = prep_eval_loader(test_raw, tok, BATCH_SIZE)

    # 모델 및 최적화 설정
    model = Transformer(
        src_vocab=len(tok), tgt_vocab=len(tok), max_len=MAX_LEN, 
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, d_ff=D_FF, dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, D_MODEL, WARMUP_STEPS)

    # WandB 초기화
    wandb.init(project="transformer-translation", config={"batch_size": BATCH_SIZE, "d_model": D_MODEL})

    print("Start Training...")
    min_loss = 1e9
    current_step = 0
    total_train_loss = 0

    model.train()
    # Streaming이므로 epoch은 데이터 소모량으로 판단
    for epoch in range(1):
        pbar = tqdm(train_loader, total=TOTAL_SAMPLES // BATCH_SIZE)
        for x, y_in, y_out in pbar:
            x, y_in, y_out = x.to(device), y_in.to(device), y_out.to(device)
            
            src_mask = make_pad_mask(x, PAD_ID)
            tgt_mask = make_pad_mask(y_in, PAD_ID)
            
            outputs = model(x, y_in, src_mask, tgt_mask)
            loss = criterion(outputs.view(-1, len(tok)), y_out.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            scheduler.step()
            
            current_step += 1
            total_train_loss += loss.item()

            # 로그 기록
            if current_step % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": total_train_loss / current_step,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "step": current_step
                })

            # 일정 주기마다 검증
            if current_step % SAVE_INTERVAL == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for vx, vy_in, vy_out in dev_loader:
                        vx, vy_in, vy_out = vx.to(device), vy_in.to(device), vy_out.to(device)
                        v_src_mask = make_pad_mask(vx, PAD_ID)
                        v_tgt_mask = make_pad_mask(vy_in, PAD_ID)
                        v_outputs = model(vx, vy_in, v_src_mask, v_tgt_mask)
                        v_loss = criterion(v_outputs.view(-1, len(tok)), vy_out.view(-1))
                        val_loss += v_loss.item()
                
                avg_val_loss = val_loss / len(dev_loader)
                wandb.log({"val/loss": avg_val_loss})
                print(f" Step {current_step}: Val Loss {avg_val_loss:.4f}")

                if avg_val_loss < min_loss:
                    min_loss = avg_val_loss
                    torch.save(model.state_dict(), 'best_model.pth')
                    print("New best model saved!")
                
                model.train()

            pbar.set_postfix(loss=f"{total_train_loss/current_step:.4f}")

    # 최종 평가 및 샘플 출력
    print("Training Finished. Evaluating on Test Set...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    preds, targets = [], []
    for item in test_raw.select(range(20)): # 샘플 20개
        src_text = item['translation']['de']
        ref_text = item['translation']['en']
        pred_text = inference(src_text, model, tok, MAX_LEN, device)
        preds.append(pred_text)
        targets.append(ref_text)
        print(f"SRC: {src_text}\nPRD: {pred_text}\nREF: {ref_text}\n")

    bleu_score = corpus_bleu(preds, [targets]).score
    print(f"Final BLEU Score: {bleu_score:.2f}")
    wandb.log({"final_bleu": bleu_score})
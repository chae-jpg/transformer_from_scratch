import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from models.model import Transformer
from tqdm import tqdm
import torch.optim as optim
import math
from sacrebleu import corpus_bleu

def make_pad_mask(seq, pad_id):
    # seq: batch x seq_len
    # return: batch x 1 x 1 x seq_len
    return (seq == pad_id).unsqueeze(1).unsqueeze(2)

class TransDataset(Dataset):
    def __init__(self, X, Y_input, Y_output):
        self.X = X
        self.Y_input = Y_input
        self.Y_output = Y_output
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_input[idx], self.Y_output[idx]

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    print("====Tokenizer succesfully loaded!====")
    return tok

def load_data():
    train_raw = load_dataset("wmt19", "de-en", split="train[:1000000]")
    test = load_dataset("wmt19", "de-en", split="validation")
    split = test.train_test_split(test_size=0.5, seed=42)
    dev_raw = split["train"]
    test_raw  = split["test"]

    print("====Data succesfully loaded!====")

    return train_raw, dev_raw, test_raw

def encode_data(data, tok, max_len=256):
    X, Y_input, Y_output = [], [], []
    BOS = tok.bos_token_id
    EOS = tok.eos_token_id

    for item in tqdm(data, desc="encoding"):
        src = item['translation']['de']
        tgt = item['translation']['en']
        src_enc = tok.encode(src, add_special_tokens=False)
        X.append([BOS] + src_enc[:max_len-2] + [EOS])

        tgt_enc = tok.encode(tgt, add_special_tokens=False)
        Y = [BOS] + tgt_enc[:max_len-2] + [EOS]
        Y_input.append(Y[:-1])
        Y_output.append(Y[1:])
    return X, Y_input, Y_output

def collate_fn(batch):
    x, y_in, y_out = zip(*batch)

    x = pad_sequence([torch.tensor(s) for s in x], batch_first=True, padding_value=PAD_ID)
    y_in = pad_sequence([torch.tensor(s) for s in y_in], batch_first=True, padding_value=PAD_ID)
    y_out = pad_sequence([torch.tensor(s) for s in y_out], batch_first=True, padding_value=PAD_ID)

    return x, y_in, y_out

def set_and_loader(data, tok, batch_size):
    X, Y_in, Y_out = encode_data(data, tok)
    dataset = TransDataset(X, Y_in, Y_out)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataset, dataloader
def inference_beam_search(text, model, tok, max_len, device, beam_size=5):
    model.eval()
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    with torch.no_grad():
        # 1. Source Encoding (한 번만 수행)
        src_enc = tok.encode(text, add_special_tokens=False)
        src = [bos_id] + src_enc[:max_len-2] + [eos_id]
        src = torch.tensor(src).unsqueeze(0).to(device)
        src_mask = make_pad_mask(src, pad_id)
        enc_out = model.enc(src, src_mask)

        # 2. Beam 초기화: (확률 합, 현재 시퀀스 리스트)
        # 로그 확률을 더하는 방식(log_softmax)을 써서 수치적 안정성을 확보함
        beams = [(0.0, [bos_id])]

        for _ in range(max_len):
            new_beams = []
            
            for score, seq in beams:
                # 이미 EOS를 만난 시퀀스는 더 이상 확장하지 않고 유지
                if seq[-1] == eos_id:
                    new_beams.append((score, seq))
                    continue
                
                # 현재 시퀀스로 다음 토큰 예측
                tgt = torch.tensor([seq]).to(device)
                tgt_mask = make_pad_mask(tgt, pad_id)
                dec_out = model.dec(tgt, enc_out, src_mask, tgt_mask)
                
                # 마지막 타임스텝의 로짓에 log_softmax 적용
                logits = model.fc(dec_out[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)
                
                # 상위 beam_size개의 후보 추출
                topk_probs, topk_ids = log_probs.topk(beam_size)
                
                for i in range(beam_size):
                    next_score = score + topk_probs[0, i].item()
                    next_seq = seq + [topk_ids[0, i].item()]
                    new_beams.append((next_score, next_seq))
            
            # 모든 후보 중 전체 점수가 높은 상위 beam_size개만 남김
            # 문장 길이에 따른 페널티를 주지 않으면 짧은 문장이 유리하므로 간단한 길이 정규화 적용 가능
            # 여기서는 단순히 점수 순으로 정렬
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
            
            # 모든 Beam이 EOS로 끝났다면 조기 종료
            if all(seq[-1] == eos_id for _, seq in beams):
                break
        
        # 최종적으로 점수가 가장 높은 시퀀스 선택
        best_seq = beams[0][1]
        result = tok.decode(best_seq, skip_special_tokens=True)
        return result

def inference(text, model, tok, max_len, device):
    model.eval()

    with torch.no_grad():
        bos_id = tok.bos_token_id
        eos_id = tok.eos_token_id
        src_enc = tok.encode(text, add_special_tokens=False)
        src = [bos_id] + src_enc[:max_len-2] + [eos_id]
        src = torch.tensor(src).unsqueeze(0).to(device)
        
        # encoding
        src_mask = make_pad_mask(src, tok.pad_token_id)
        enc_out = model.enc(src, src_mask)

        # decoding
        tgt = torch.tensor([[bos_id]]).to(device)

        for _ in range(max_len):
            tgt_mask = make_pad_mask(tgt, tok.pad_token_id)
            dec_out = model.dec(tgt, enc_out, src_mask, tgt_mask)
            last = model.fc(dec_out[:, -1, :])
            selected = last.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, selected], dim=1)
            if selected.item() == eos_id:
                break
        
        result = tok.decode(tgt.squeeze().tolist(), skip_special_tokens=True)
        return result

if __name__ == "__main__":
    BATCH_SIZE = 32
    _, _, test_raw = load_data()
    
    tok = load_tokenizer()
    special_tokens_dict = {'bos_token': '[BOS]', 'eos_token': '[EOS]', 'pad_token': '[PAD]'}
    num_added_toks = tok.add_special_tokens(special_tokens_dict)
    PAD_ID = tok.pad_token_id
    testset, testloader = set_and_loader(test_raw, tok, BATCH_SIZE)
    
    MAX_LEN = 256
    VOCAB_SIZE = len(tok)
    D_MODEL = 512
    NHEAD = 8
    NUM_LAYERS = 6
    D_FF = 2048
    DROPOUT = 0.1

    learning_rate = 0.5
    num_epochs = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(
        src_vocab = VOCAB_SIZE, tgt_vocab = VOCAB_SIZE, max_len = MAX_LEN, d_model = D_MODEL, nhead = NHEAD, num_layers = NUM_LAYERS, d_ff = D_FF, dropout = DROPOUT
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    model.load_state_dict(torch.load('best_model.pth')) 
    # evaluation
    test_loss = 0
    pred, target = [], []   
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction='sum') # 'sum'으로 변경
    total_test_loss = 0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for x, y_in, y_out in tqdm(testloader):
            x, y_in, y_out = x.to(device), y_in.to(device), y_out.to(device)
            
            src_mask = make_pad_mask(x, PAD_ID)
            tgt_mask = make_pad_mask(y_in, PAD_ID)
                
            outputs = model(x, y_in, src_mask, tgt_mask)
            outputs = outputs.view(-1, VOCAB_SIZE)
            y_out = y_out.view(-1)
            
            # 패딩이 아닌 토큰의 개수 계산
            num_tokens = (y_out != PAD_ID).sum().item()
            
            # reduction='sum'이므로 이 배치의 전체 손실 합이 나옴
            loss = criterion(outputs, y_out)

            total_test_loss += loss.item()
            total_tokens += num_tokens

    # 전체 토큰 대비 평균 손실 계산
    avg_test_loss = total_test_loss / total_tokens
    perplexity = math.exp(avg_test_loss)

    for item in tqdm(test_raw):
        src = item['translation']['de']
        tgt = item['translation']['en']
        model.eval()
        pred.append(inference_beam_search(src, model, tok, MAX_LEN, device))
        target.append(tgt)

    bleu = corpus_bleu(pred, [target]).score

    print(f"Training finished. Perplexity: {perplexity}, BLEU: {bleu}")

    print("\n====== Sample Translations ======")
    for i in range(min(5, len(pred))):
        print(f"[{i}] PRD: {pred[i]}")
        print(f"    REF: {target[i]}\n")
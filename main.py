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

class TransDataset(Dataset):
    def __init__(self, X, Y_input, Y_output):
        self.X = X
        self.Y_input = Y_input
        self.Y_output = Y_output
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_input[idx], self.Y_output[idx]

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

def load_data():
    train_raw = load_dataset("wmt19", "de-en", split="train[:1000000]")
    test = load_dataset("wmt19", "de-en", split="validation")
    split = test.train_test_split(test_size=0.5, seed=42)
    dev_raw = split["train"]
    test_raw  = split["test"]

    print("====Data succesfully loaded!====")

    return train_raw, dev_raw, test_raw

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    print("====Tokenizer succesfully loaded!====")
    return tok

def encode_data(data, tok, max_len=256):
    X, Y_input, Y_output = [], [], []
    EOS = tok.eos_token_id

    for item in tqdm(data):
        src = item['translation']['de']
        tgt = item['translation']['en']
        src_enc = tok.encode(src)[:max_len] 
        X.append(src_enc)
        tgt_enc = tok.encode(tgt, add_special_tokens=False)[:max_len-1] + [EOS]
        Y_input.append([EOS] + tgt_enc[:-1])
        Y_output.append(tgt_enc)
    return X, Y_input, Y_output

def collate_fn(batch):
    x, y_in, y_out = zip(*batch)

    x = pad_sequence([torch.tensor(s) for s in x], batch_first=True, padding_value=PAD_ID)
    y_in = pad_sequence([torch.tensor(s) for s in y_in], batch_first=True, padding_value=PAD_ID)
    y_out = pad_sequence([torch.tensor(s) for s in y_out], batch_first=True, padding_value=PAD_ID)

    return x, y_in, y_out

def make_pad_mask(seq, pad_id):
    # seq: batch x seq_len
    # return: batch x 1 x 1 x seq_len
    return (seq == pad_id).unsqueeze(1).unsqueeze(2)

def set_and_loader(data, tok, batch_size):
    X, Y_in, Y_out = encode_data(data, tok)
    dataset = TransDataset(X, Y_in, Y_out)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataset, dataloader

def inference(text, model, tok, max_len, device):
    model.eval()

    with torch.no_grad():
        eos_id = tok.eos_token_id
        src = tok.encode(text)[:max_len]
        src = torch.tensor(src).unsqueeze(0).to(device)
        
        # encoding
        src_mask = make_pad_mask(src, tok.pad_token_id)
        enc_out = model.enc(src, src_mask)

        # decoding
        tgt = torch.tensor([[tok.eos_token_id]]).to(device)

        for _ in range(max_len):
            tgt_mask = make_pad_mask(tgt, tok.pad_token_id)
            dec_out = model.dec(tgt, enc_out, src_mask, tgt_mask)
            last = dec_out[:, -1, :]
            selected = model.fc(last).argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, selected], dim=1)
            if selected.item() == eos_id:
                break
        
        result = tok.decode(tgt.squeeze().tolist(), skip_special_tokens=True)
        return result

if __name__ == "__main__":
    MAX_LEN = 256
    BATCH_SIZE = 32
    
    train_raw, dev_raw, test_raw = load_data()
    tok = load_tokenizer()
    PAD_ID = tok.pad_token_id
    # training set
    trainset, trainloader = set_and_loader(train_raw, tok, BATCH_SIZE)
    # dev set
    devset, devloader = set_and_loader(dev_raw, tok, BATCH_SIZE)
    # test set
    testset, testloader = set_and_loader(test_raw, tok, BATCH_SIZE)

    print("====Trainset, Devset loaded successfully!====")

    # src_vocab, tgt_vocab, max_len, d_model=512, nhead=8, num_layers=6, d_ff=2048, dropout=0.1
    VOCAB_SIZE = len(tok)
    D_MODEL = 512
    NHEAD = 8
    NUM_LAYERS = 6
    D_FF = 2048
    DROPOUT = 0.1

    learning_rate = 0.5
    num_epochs = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(
        src_vocab = VOCAB_SIZE, tgt_vocab = VOCAB_SIZE, max_len = MAX_LEN, d_model = D_MODEL, nhead = NHEAD, num_layers = NUM_LAYERS, d_ff = D_FF, dropout = DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = WarmupScheduler(optimizer, D_MODEL, warmup_steps=4000)

    print("Start Training...")
    min_loss = 1e9
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        # training loop
        for i, (x, y_in, y_out) in tqdm(enumerate(trainloader)):
            x = x.to(device)
            y_in = y_in.to(device)
            y_out = y_out.to(device)
            # Forward pass
            src_mask = make_pad_mask(x, PAD_ID)
            tgt_mask = make_pad_mask(y_in, PAD_ID)
            
            outputs = model(x, y_in, src_mask, tgt_mask) # (batch_size, seq_len, vocab_size)
            outputs = outputs.view(-1, VOCAB_SIZE) # (batch_size * seq_len, vocab_size)
            y_out = y_out.view(-1)
            loss = criterion(outputs, y_out)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f'Epoch {epoch+1}, step {i+1}, step loss: {(total_loss / (i+1)):.4f}')

        avg_loss = total_loss / len(trainloader) 
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # evaluate validation loss
        eval_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y_in, y_out in tqdm(devloader):
                x = x.to(device)
                y_in = y_in.to(device)
                y_out = y_out.to(device)

                # Forward pass
                src_mask = make_pad_mask(x, PAD_ID)
                tgt_mask = make_pad_mask(y_in, PAD_ID)
                
                outputs = model(x, y_in, src_mask, tgt_mask) # (batch_size, seq_len, vocab_size)
                outputs = outputs.view(-1, VOCAB_SIZE) # (batch_size * seq_len, vocab_size)
                y_out = y_out.view(-1)
                loss = criterion(outputs, y_out)

                eval_loss += loss.item()
        avg_eval_loss = eval_loss / len(devloader) 
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_eval_loss:.4f}')
        
        if avg_eval_loss < min_loss:
            min_loss = avg_eval_loss
            torch.save(model.state_dict(), 'weights.pth')
    # evaluation
    test_loss = 0
    pred, target = [], []   
    model.eval()
    with torch.no_grad():
        for x, y_in, y_out in tqdm(testloader):
            x = x.to(device)
            y_in = y_in.to(device)
            y_out = y_out.to(device)

            # Forward pass
            src_mask = make_pad_mask(x, PAD_ID)
            tgt_mask = make_pad_mask(y_in, PAD_ID)
                
            outputs = model(x, y_in, src_mask, tgt_mask) # (batch_size, seq_len, vocab_size)
            outputs = outputs.view(-1, VOCAB_SIZE) # (batch_size * seq_len, vocab_size)
            y_out = y_out.view(-1)
            loss = criterion(outputs, y_out)

            test_loss += loss.item()

    avg_test_loss = test_loss / len(testloader) 
    perplexity = math.exp(avg_test_loss)

    for item in tqdm(test_raw.select(range(10))):
        src = item['translation']['de']
        tgt = item['translation']['en']
        model.eval()
        pred.append(inference(src, model, tok, MAX_LEN, device))
        target.append(tgt)

    bleu = corpus_bleu(pred, [target]).score

    print(f"Training finished. Perplexity: {perplexity}, BLEU: {bleu}")

    print("\n====== Sample Translations ======")
    for i in range(min(5, len(pred))):
        print(f"[{i}] PRD: {pred[i]}")
        print(f"    REF: {target[i]}\n")
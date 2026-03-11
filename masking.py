import torch
import torch.nn as nn

def check_masking_logic():
    # 설정
    batch_size = 1
    max_len = 5
    seq_len = 5
    pad_id = 0
    
    # 1. 가상의 입력 시퀀스 생성 (0은 패딩)
    # 예: [BOS, 단어1, 단어2, PAD, PAD]
    tgt_seq = torch.tensor([[10, 11, 12, 0, 0]]) 
    
    # 2. 패딩 마스크 생성 (main.py의 make_pad_mask 방식)
    # 결과 크기: (batch, 1, 1, seq_len)
    tgt_pad_mask = (tgt_seq == pad_id).unsqueeze(1).unsqueeze(2)
    
    # 3. Causal Mask 생성 (model.py의 DecoderBlock 방식)
    # 결과 크기: (1, 1, max_len, max_len)
    causal_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool().unsqueeze(0).unsqueeze(0)
    
    # 4. 결합 (Broadcasting 발생)
    # causal_mask: (1, 1, 5, 5)
    # tgt_pad_mask: (1, 1, 1, 5) -> (1, 1, 5, 5)로 확장됨
    combined_mask = causal_mask | tgt_pad_mask

    print("--- 1. Target Sequence (0 is PAD) ---")
    print(tgt_seq)
    print("\n--- 2. Padding Mask (True means 'Masked') ---")
    print(tgt_pad_mask.int()) # 1: Masked, 0: Visible
    
    print("\n--- 3. Causal Mask (Upper triangle is True) ---")
    print(causal_mask[:, :, :seq_len, :seq_len].int())
    
    print("\n--- 4. Combined Mask (Final Mask to be applied) ---")
    # 시각적으로 보기 좋게 행렬 형태로 출력
    mask_to_show = combined_mask[0, 0].int()
    print(mask_to_show)
    
    print("\n--- Interpretation ---")
    print("Row (i) is the position we are calculating the hidden state for.")
    print("Col (j) is the position we are 'attending' to.")
    print("1 means it is BLOCKED, 0 means it is VISIBLE.")

if __name__ == "__main__":
    check_masking_logic()

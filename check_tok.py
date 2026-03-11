
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')
print('eos_token:', repr(tok.eos_token), 'id:', tok.eos_token_id)
print('pad_token:', repr(tok.pad_token), 'id:', tok.pad_token_id)
print('bos_token:', repr(tok.bos_token), 'id:', tok.bos_token_id)
print('vocab_size:', len(tok))

# Test encoding behavior
test = 'Hello world'
enc_default = tok.encode(test)
enc_no_special = tok.encode(test, add_special_tokens=False)
print('encode(default):', enc_default)
print('encode(no_special):', enc_no_special)
print('decode default:', tok.decode(enc_default))
print('decode no_special:', tok.decode(enc_no_special))

# Check if eos is added by default
print('Last token of default encode:', enc_default[-1], '== eos_id?', enc_default[-1] == tok.eos_token_id)

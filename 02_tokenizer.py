from transformers import AutoTokenizer

for name in ["klue/bert-base", "gpt2"]:
    tok = AutoTokenizer.from_pretrained(name)
    s = "헤이트슬롭 사내 교육에 오신 것을 환영합니다"
    ids = tok(s)["input_ids"]
    print(f"{name:<16} {len(ids):>3} tokens", tok.convert_ids_to_tokens(ids)[:12])

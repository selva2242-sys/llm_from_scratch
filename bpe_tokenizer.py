import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "I love supercalifragilisticexpialidocious"


tokens = tokenizer.encode(text);

print(tokens)

decoded = tokenizer.decode(tokens);

print(decoded)

for token_id in tokens:
    token_bytes = tokenizer.decode_single_token_bytes(token_id)
    print(token_id, "->", token_bytes)


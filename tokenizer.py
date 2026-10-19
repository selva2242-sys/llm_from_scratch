from pathlib import Path
import re 

file_path = Path("./data/the-verdict.txt")

with open(file_path, "r" , encoding = "utf-8" ) as f:
	raw_text = f.read();

tokens = set(re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)) 

vocab = {
    token: index
    for index, token in enumerate(sorted(set(tokens)))
}
vocab["<|unk|>"] = len(vocab)


class Tokenizer:
	def __init__(self, vocab):
        	self.str_to_int = vocab
        	self.int_to_str = {
            		token_id: token
            		for token, token_id in vocab.items()
        	}
	def encode(self, text):
		tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
		tokens = [token.strip() for token in tokens if token.strip()]
		ids = [self.str_to_int.get(token, self.str_to_int["<|unk|>"])for token in tokens]   		
		return ids
	def decode(self, ids):
        	text = " ".join(self.int_to_str[token_id] for token_id in ids)

        	# Remove unnecessary spaces before punctuation
        	text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        	return text


tokenizer = Tokenizer(vocab)

encoded = tokenizer.encode("Java JavaScript.")
print("Encoded:", encoded)

decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)

print("Total tokens", len(tokens))

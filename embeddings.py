
import torch
import tiktoken


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path


file_path = Path("./data/the-verdict.txt")

with open(file_path, "r" , encoding = "utf-8" ) as f:
	raw_text = f.read();


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Convert text into token IDs
        token_ids = tokenizer.encode(txt)

        # Create sliding-window samples
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


tokenizer = tiktoken.get_encoding("gpt2")

text = "I love AI. I love Python. I love machine learning."

dataset = GPTDatasetV1(
    txt=raw_text,
    tokenizer=tokenizer,
    max_length=4,
    stride=2
)


# for i in range(len(dataset)):
#     x, y = dataset[i]

#     print(f"Sample {i + 1}")
#     print("Input IDs:", x.tolist())
#     print("Target IDs:", y.tolist())

#     print("Input text:", tokenizer.decode(x.tolist()))
#     print("Target text:", tokenizer.decode(y.tolist()))
#     print("---")

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    drop_last=True
)
print("Total training samples:", len(dataset))
print("Total batches:", len(dataloader))

# for batch_number, (inputs, targets) in enumerate(dataloader):
#     print(f"\nBatch {batch_number + 1}")
#     print("Input shape:", inputs.shape)
#     print("Target shape:", targets.shape)
#     print("Inputs:", inputs)
#     print("Targets:", targets)




# Vocabulary size comes from the tokenizer
vocab_size = tokenizer.n_vocab

# Our chosen embedding dimension
embedding_dim = 256

# Create the trainable token embedding table
token_embedding_layer = torch.nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

# Get the first batch from DataLoader
inputs, targets = next(iter(dataloader))

print("Input token IDs:")
print(inputs)

print("Input shape:", inputs.shape)

# Convert token IDs into embedding vectors
token_embeddings = token_embedding_layer(inputs)

print("Embedding table shape:", token_embedding_layer.weight.shape)
print("Token embeddings shape:", token_embeddings.shape)


pos_embedding_layer = torch.nn.Embedding(4, 256)

positions = torch.arange(4)
pos_embeddings = pos_embedding_layer(positions)
input_embeddings = token_embeddings + pos_embeddings
print("Positiong embeddings shape:", pos_embeddings.shape)
print("Input embeddings shape:", input_embeddings.shape)


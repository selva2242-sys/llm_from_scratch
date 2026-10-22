token_ids = [10, 20, 30, 40, 50, 60]

context_length = 3
stride = 1

for i in range(0, len(token_ids) - context_length, stride):
    input_chunk = token_ids[i:i + context_length]
    target_chunk = token_ids[i + 1:i + context_length + 1]

    print(f"X: {input_chunk}")
    print(f"Y: {target_chunk}")
    print("---")
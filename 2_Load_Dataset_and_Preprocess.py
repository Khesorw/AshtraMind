from datasets import load_dataset

train_dataset = load_dataset("rahular/itihasa", split="train")
valid_dataset = load_dataset("rahular/itihasa", split="validation")
test_dataset  = load_dataset("rahular/itihasa", split="test")
print("Datasets loaded successfully.")

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(valid_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
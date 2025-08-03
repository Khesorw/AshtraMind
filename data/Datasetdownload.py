from datasets import load_dataset

ds = load_dataset("rahular/itihasa")

# Save the dataset to a local directory
ds.save_to_disk("./itihasa_dataset")
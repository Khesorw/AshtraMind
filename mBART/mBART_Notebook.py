# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: ashtra_mind
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1_Setup_and_Testing

# %%
# All imports
import sys
import pip
import torch
from datasets import get_dataset_split_names, load_dataset, load_dataset_builder, get_dataset_config_names,  load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBart50Tokenizer, MBartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, BitsAndBytesConfig
import evaluate
import numpy as np
import gc
from accelerate import Accelerator

print("All imports are successful ✅")

print("--" * 50)

#---------------------------------------------------------------
# Check Python, pip, and pytorch versions and cuda compatibility
#---------------------------------------------------------------
print("Python version:", sys.version)
# Print pip version
print("Pip version:", pip.__version__)
# Print pytorch version
print("Pytorch version:", torch.__version__)
# Print CUDA version
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
else:
    print("CUDA is not available.")

# Print GPU information
if torch.cuda.is_available():
    print("GPU is available.")
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available.")

# Check if pytorch can use CUDA
if torch.cuda.is_available():
    x = torch.rand(5, 3).cuda()
    if x.is_cuda:
        print("Pytorch can use CUDA ✅Tensor on GPU")
else:
    print("Pytorch is not using CUDA.")

print("--" * 50)
# Check if evaluate is working
# !python -c "import evaluate; print(evaluate.load('exact_match').compute(references=['hello'], predictions=['hello']))"
print("Evaluate is working ✅")



# %% [markdown]
# # 2_Load_Dataset

# %%
# https://huggingface.co/docs/datasets/load_hub
splits = get_dataset_split_names("rahular/itihasa")
print("Available dataset splits:", splits)
configs = get_dataset_config_names("rahular/itihasa")
print("Available dataset configurations:", configs)

# %%
ds_builder = load_dataset_builder("rahular/itihasa")

# Inspect dataset description
ds_builder.info.description

# Inspect dataset features
ds_builder.info.features

# %%
from datasets import load_dataset

train_dataset = load_dataset("rahular/itihasa", split="train")
valid_dataset = load_dataset("rahular/itihasa", split="validation")
test_dataset  = load_dataset("rahular/itihasa", split="test")
print("Datasets loaded successfully ✅.")

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(valid_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# %%
train_dataset[0]  # Inspect the first example in the train dataset

# %%
test_dataset[0]  # Inspect the first example in the test dataset

# %%
valid_dataset[0] # Inspect the first example in the validation dataset

# %%
# Indexing the datasets
# print(train_dataset[0])  # To see the full content of the first example
# print("--" * 50)
# print(train_dataset[0]["translation"])  # To see the root of the nested dictionary
# print("--" * 50)
# print(train_dataset[0]["translation"]["en"])  # To see the English translation of the first example
# print("--" * 50)
# print(train_dataset[0]["translation"]["sn"])  # To see the Sanskrit translation of the first example
# print("--" * 50)
# for i in range(3):
#     print(f"Example {i}: (English: {train_dataset[i]['translation']['en']}) (Sanskrit: {train_dataset[i]['translation']['sn']})")

# %% [markdown]
# # 3_Modelling

# %%
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_threshold=6.0,
#     llm_int8_has_fp16_weight=True
# )

# MODEL = MBartForConditionalGeneration.from_pretrained(
#     "facebook/mbart-large-50-many-to-many-mmt",
#     quantization_config=bnb_config, 
#     device_map="auto"
# )
TOKENIZER = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
TOKENIZER.src_lang = "en_XX"
TOKENIZER.tgt_lang = "hi_IN"  # Setting Hindi token id as a proxy for Sanskrit

MODEL = MBartForConditionalGeneration.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt",
    torch_dtype=torch.float16,  # Use float16 for better performance on GPUs
    device_map="auto",
)
TEXT_TO_TRANSLATE = "For one who has conquered the mind, the mind is the best of friends; but for one who has failed to do so, his very mind will be the greatest enemy."


# %%
def translate_text(text, model=MODEL, tokenizer=TOKENIZER, src_lang=TOKENIZER.src_lang, tgt_lang=TOKENIZER.tgt_lang, skip_special_tokens=True):

    inputs = tokenizer(text, return_tensors="pt")

    # Force decoder to use target language
    output_ids = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=skip_special_tokens)

try:
    translated_text = translate_text(TEXT_TO_TRANSLATE)
    print(f"Model and Tokenizer intialized successfully ✅")
    print(f"Translated text: {translated_text}")
except Exception as e:
    print(f"Error occurred during translation: {e}")

# %%
# Human-readable language names mapped to mBART-50 language codes
# lang_code_to_name = {
#     "ar_AR": "Arabic", "cs_CZ": "Czech", "de_DE": "German", "en_XX": "English", "es_XX": "Spanish",
#     "et_EE": "Estonian", "fi_FI": "Finnish", "fr_XX": "French", "gu_IN": "Gujarati", "hi_IN": "Hindi",
#     "it_IT": "Italian", "ja_XX": "Japanese", "kk_KZ": "Kazakh", "ko_KR": "Korean", "lt_LT": "Lithuanian",
#     "lv_LV": "Latvian", "my_MM": "Burmese", "ne_NP": "Nepali", "nl_XX": "Dutch", "ro_RO": "Romanian",
#     "ru_RU": "Russian", "si_LK": "Sinhala", "tr_TR": "Turkish", "vi_VN": "Vietnamese", "zh_CN": "Chinese (Simplified)",
#     "af_ZA": "Afrikaans", "az_AZ": "Azerbaijani", "bn_IN": "Bengali", "fa_IR": "Persian", "he_IL": "Hebrew",
#     "hr_HR": "Croatian", "id_ID": "Indonesian", "ka_GE": "Georgian", "km_KH": "Khmer", "mk_MK": "Macedonian",
#     "ml_IN": "Malayalam", "mn_MN": "Mongolian", "mr_IN": "Marathi", "pl_PL": "Polish", "ps_AF": "Pashto",
#     "pt_XX": "Portuguese", "sr_XX": "Serbian", "ta_IN": "Tamil", "te_IN": "Telugu", "th_TH": "Thai",
#     "tl_XX": "Tagalog", "uk_UA": "Ukrainian", "ur_PK": "Urdu", "xh_ZA": "Xhosa", "gl_ES": "Galician",
#     "sl_SI": "Slovenian"
# }

# # Print total number of languages
# print("Total languages supported by the tokenizer:", len(TOKENIZER.lang_code_to_id))

# # Print human-readable name for each language code
# for lang_code, token_id in TOKENIZER.lang_code_to_id.items():
#     name = lang_code_to_name.get(lang_code, "Unknown")
#     print(f"Language Code: {lang_code}, Human Name: {name}, Token ID: {token_id}")


# %% [markdown]
# # 4_Preprocessing

# %%
# Calculate the length of input IDs for each Sanskrit translation in the training dataset
# This will help to select max length for model inputs in the preprocess function
# Extract list of Sanskrit texts
# Sanskrit contains lot of samasa (compound words) which can be long therefore appropriate to check token lengths
sanskrit_texts = [item["translation"]["sn"] for item in train_dataset]

# Now calculate token lengths
token_lens = [len(TOKENIZER(text)["input_ids"]) for text in sanskrit_texts]

# Check maximum and top 10 longest
print("Max length:", max(token_lens))
print("Top 10 longest:", sorted(token_lens)[-10:])


# %%
def preprocess_function(examples):
    inputs = [t["en"] for t in examples["translation"]]
    targets = [t["sn"] for t in examples["translation"]]  # Sanskrit texts

    model_inputs = TOKENIZER(inputs, max_length=8, truncation=True, padding="longest")

    # tokenize targets, set padding as longest which saves memory and pads only to the longest target in the batch
    labels = TOKENIZER(targets, max_length=8, truncation=True, padding="longest")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)

# Tokenize the validation dataset
tokenized_valid = valid_dataset.map(preprocess_function, batched=True)

# Tokenize the test dataset
tokenized_test = test_dataset.map(preprocess_function, batched=True)

print("Tokenization complete for train, validation, and test datasets ✅")


# %% [markdown]
# # 5_Training

# %%
# Data collator for Seq2Seq models used for padding and creating attention masks
data_collator = DataCollatorForSeq2Seq(tokenizer=TOKENIZER, model=MODEL,padding="longest")
print("Data collator created successfully ✅")

# %%
bleu = evaluate.load("bleu")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # decode predictions and labels
    decoded_preds = TOKENIZER.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, TOKENIZER.pad_token_id)
    decoded_labels = TOKENIZER.batch_decode(labels, skip_special_tokens=True)
    
    # BLEU expects list of references for each prediction (hence [[ref1], [ref2], ...])
    decoded_labels = [[label.split()] for label in decoded_labels]
    decoded_preds = [pred.split() for pred in decoded_preds]
    
    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["bleu"]}

print("Metrics function created successfully ✅")

# %%
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=3e-5,
    num_train_epochs=3,
    fp16=True,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    # optim="paged_adamw_8bit",  # Using 8-bit AdamW optimizer for memory efficiency
    predict_with_generate=True,  # important for seq2seq tasks
)
print("Training arguments set successfully ✅")

# %%
trainer = Seq2SeqTrainer(
    model=MODEL,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    processing_class=TOKENIZER,
    data_collator=data_collator,
    
    compute_metrics=compute_metrics,
)

# %%
# Clear CUDA cache to free up memory
gc.collect()
torch.cuda.empty_cache()
print("CUDA cache cleared ✅")

# %%
accelerator = Accelerator()
print(f"Mixed precision: {accelerator.mixed_precision}")


# %%
# Train the model
trainer.train()

# %% [markdown]
# # 6_Evaluation

# %%
# Evaluate the model on the test dataset

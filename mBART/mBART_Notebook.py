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
from peft import get_peft_model, LoraConfig, TaskType

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
QUANT_CONFIG = BitsAndBytesConfig(load_in_8bit=True)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,               # rank of LoRA matrices
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj"],
)

model = MBartForConditionalGeneration.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt",
    quantization_config=QUANT_CONFIG,
)

MODEL = get_peft_model(model, peft_config)
print("Quantize and LoRA Model loaded successfully ✅")
trainable_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad) 
total_params = sum(p.numel() for p in MODEL.parameters())
print(f"Total parameters: {total_params:,}")
print(f"✅ Trainable parameters: {trainable_params:,}")
print(f"💡 Trainable ratio: {100 * trainable_params / total_params:.4f}%")
for name, param in MODEL.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

TOKENIZER = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
TOKENIZER.src_lang = "en_XX"
TOKENIZER.tgt_lang = "hi_IN"  # Setting Hindi token id as a proxy for Sanskrit

TEXT_TO_TRANSLATE = "For one who has conquered the mind, the mind is the best of friends; but for one who has failed to do so, his very mind will be the greatest enemy."

# %%
def translate_text(text, model=MODEL, tokenizer=TOKENIZER, src_lang=TOKENIZER.src_lang, tgt_lang=TOKENIZER.tgt_lang, skip_special_tokens=True):
    # Get model device
    device = next(model.parameters()).device
    
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Force decoder to use target language
    with torch.no_grad():  # Add no_grad for inference
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=128,  # Add explicit max_length
            num_beams=4,     # Match training beam size
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=skip_special_tokens)

try:
    translated_text = translate_text(TEXT_TO_TRANSLATE)
    print(f"Model and Tokenizer intialized successfully ✅")
    print(f"Original text: {TEXT_TO_TRANSLATE}")
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

    model_inputs = TOKENIZER(inputs, max_length=128, truncation=True, padding="longest")

    # tokenize targets, set padding as longest which saves memory and pads only to the longest target in the batch
    labels = TOKENIZER(targets, max_length=128, truncation=True, padding="longest")

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
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode predictions
    decoded_preds = TOKENIZER.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels with pad_token_id, then decode
    labels = np.where(labels != -100, labels, TOKENIZER.pad_token_id)
    decoded_labels = TOKENIZER.batch_decode(labels, skip_special_tokens=True)

    # Clean up spacing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]  # wrap each label in a list

    # Compute BLEU
    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Compute ROUGE for additional metrics
    rouge_result = rouge.compute(
        predictions=decoded_preds, 
        references=[ref[0] for ref in decoded_labels]
    )

    return {
        "bleu": bleu_result["bleu"],
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"]
    }


print("Metrics function created successfully ✅")

# %%
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    num_train_epochs=3,
    fp16=True,
    weight_decay=0.01,
    save_total_limit=3,
    save_steps=200,
    logging_dir="./logs",
    logging_steps=100,
    optim="adamw_torch_fused",
    predict_with_generate=True, # Important for seq2seq tasks
    generation_max_length=128,  # Max length for generated sequences during eval
    generation_num_beams=4,     # Use beam search for better quality
    warmup_steps=100,           # Add warmup for stable training
    load_best_model_at_end=True, # Load best model at end
    metric_for_best_model="bleu", # Use BLEU to determine best model
    greater_is_better=True,      # Higher BLEU is better
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
print("Starting model evaluation...")

# Evaluate on test set
test_results = trainer.evaluate(eval_dataset=tokenized_test)
print("Test Results:", test_results)

# %%
# Sample translations for qualitative analysis
def evaluate_sample_translations():
    sample_texts = [
        "The mind is everything. What you think you become.",
        "Happiness comes from within.",
        "Knowledge is power.",
        "Truth alone triumphs.",
        "The whole world is one family."
    ]
    
    print("Sample Translations:")
    print("=" * 80)
    
    for i, text in enumerate(sample_texts, 1):
        try:
            translation = translate_text(text)
            print(f"{i}. English: {text}")
            print(f"   Sanskrit: {translation}")
            print("-" * 60)
        except Exception as e:
            print(f"Error translating '{text}': {e}")
            print("-" * 60)

evaluate_sample_translations()

# %%
# Save the fine-tuned model
MODEL.save_pretrained("./fine_tuned_mbart_sanskrit")
TOKENIZER.save_pretrained("./fine_tuned_mbart_sanskrit")
print("Model and tokenizer saved successfully ✅")

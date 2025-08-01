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
from datasets import get_dataset_split_names, load_dataset, load_dataset_builder, get_dataset_config_names
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBart50Tokenizer, MBartForConditionalGeneration

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


# %% [markdown]
# # 2_Load_Dataset_and_Preprocess

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
print(train_dataset[0])  # To see the full content of the first example
print("--" * 50)
print(train_dataset[0]["translation"])  # To see the root of the nested dictionary
print("--" * 50)
print(train_dataset[0]["translation"]["en"])  # To see the English translation of the first example
print("--" * 50)
print(train_dataset[0]["translation"]["sn"])  # To see the Sanskrit translation of the first example
print("--" * 50)
for i in range(3):
    print(f"Example {i}: (English: {train_dataset[i]['translation']['en']}) (Sanskrit: {train_dataset[i]['translation']['sn']})")

# %% [markdown]
# # 3_Modelling_and_Training

# %%
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

def translate_text(model, tokenizer, text, src_lang="en_XX", tgt_lang="hi_IN", skip_special_tokens=True):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(text, return_tensors="pt")
    
    # Force decoder to use target language
    output_ids = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=skip_special_tokens)



# %%
# Human-readable language names mapped to mBART-50 language codes
lang_code_to_name = {
    "ar_AR": "Arabic", "cs_CZ": "Czech", "de_DE": "German", "en_XX": "English", "es_XX": "Spanish",
    "et_EE": "Estonian", "fi_FI": "Finnish", "fr_XX": "French", "gu_IN": "Gujarati", "hi_IN": "Hindi",
    "it_IT": "Italian", "ja_XX": "Japanese", "kk_KZ": "Kazakh", "ko_KR": "Korean", "lt_LT": "Lithuanian",
    "lv_LV": "Latvian", "my_MM": "Burmese", "ne_NP": "Nepali", "nl_XX": "Dutch", "ro_RO": "Romanian",
    "ru_RU": "Russian", "si_LK": "Sinhala", "tr_TR": "Turkish", "vi_VN": "Vietnamese", "zh_CN": "Chinese (Simplified)",
    "af_ZA": "Afrikaans", "az_AZ": "Azerbaijani", "bn_IN": "Bengali", "fa_IR": "Persian", "he_IL": "Hebrew",
    "hr_HR": "Croatian", "id_ID": "Indonesian", "ka_GE": "Georgian", "km_KH": "Khmer", "mk_MK": "Macedonian",
    "ml_IN": "Malayalam", "mn_MN": "Mongolian", "mr_IN": "Marathi", "pl_PL": "Polish", "ps_AF": "Pashto",
    "pt_XX": "Portuguese", "sr_XX": "Serbian", "ta_IN": "Tamil", "te_IN": "Telugu", "th_TH": "Thai",
    "tl_XX": "Tagalog", "uk_UA": "Ukrainian", "ur_PK": "Urdu", "xh_ZA": "Xhosa", "gl_ES": "Galician",
    "sl_SI": "Slovenian"
}

# Print total number of languages
print("Total languages supported by the tokenizer:", len(tokenizer.lang_code_to_id))

# Print human-readable name for each language code
for lang_code, token_id in tokenizer.lang_code_to_id.items():
    name = lang_code_to_name.get(lang_code, "Unknown")
    print(f"Language Code: {lang_code}, Human Name: {name}, Token ID: {token_id}")


# %%

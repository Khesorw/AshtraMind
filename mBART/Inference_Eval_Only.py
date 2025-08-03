import torch
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig
import evaluate

def preprocess_function(examples, tokenizer):
    inputs = [t["en"] for t in examples["translation"]]
    targets = [t["sn"] for t in examples["translation"]]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="longest")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="longest")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    checkpoint_dir = "results/checkpoint-4500"
    base_model_name = "facebook/mbart-large-50-many-to-many-mmt"

    QUANT_CONFIG = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = MBart50Tokenizer.from_pretrained(checkpoint_dir)
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "hi_IN"  # Sanskrit proxy

    base_model = MBartForConditionalGeneration.from_pretrained(
        base_model_name,
        quantization_config=QUANT_CONFIG,
    )

    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load dataset
    test_dataset = load_dataset("rahular/itihasa", split="test")

    # Tokenize but KEEP original fields
    tokenized_test = test_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Load metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    chrf_metric = evaluate.load("chrf")

    predictions = []
    references = []
    batch_size = 16

    print("Starting evaluation...")
    for i in range(0, len(tokenized_test), batch_size):
        batch = tokenized_test[i : i + batch_size]

        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
            )

        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # Use original test_dataset for references
        refs = [test_dataset[i + j]["translation"]["sn"] for j in range(len(preds))]

        predictions.extend(preds)
        references.extend(refs)

    # Format references for BLEU (list of list)
    references_for_metric = [[r] for r in references]

    bleu_res = bleu_metric.compute(predictions=predictions, references=references_for_metric)
    rouge_res = rouge_metric.compute(predictions=predictions, references=references)
    chrf_res = chrf_metric.compute(predictions=predictions, references=references)

    print(f"\n--- Evaluation Results ---")
    print(f"BLEU:     {bleu_res['bleu']:.4f}")
    print(f"ROUGE-1:  {rouge_res['rouge1']:.4f}")
    print(f"ROUGE-2:  {rouge_res['rouge2']:.4f}")
    print(f"ROUGE-L:  {rouge_res['rougeL']:.4f}")
    print(f"chrF++:   {chrf_res['score']:.4f}")

    print("\nSample Predictions:\n")
    for ref, pred in list(zip(references, predictions))[:5]:
        print(f"REF : {ref}")
        print(f"PRED: {pred}")
        print()

if __name__ == "__main__":
    main()

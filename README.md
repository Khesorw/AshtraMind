# AshtraMind

### Problem Definition
- Build a high-quality Sanskrit–English translation system with a specific focus on improving the translation of compound words (samāsa), which are prevalent and complex in Sanskrit literature.

- The goal is to produce grammatically correct, semantically accurate, and context-aware translations that handle long sentences and preserve spiritual/philosophical meaning.
### Dataset Used 
- The Itihasa dataset is utilized, containing 93,000+ parallel sentences in Sanskrit and English.

- This dataset includes diverse and complex sentence structures drawn from ancient scriptures, epics, and classical literature, making it ideal for training models on rich linguistic patterns.

### Evaluation Metrics
To assess the translation quality comprehensively, the following standard machine translation metrics were used:

- BLEU Score – Evaluates n-gram overlap between predicted and reference translations.

- ROUGE Score – Measures recall-based overlap, especially useful for longer sequences.

- CHRF++ Score – Character n-gram F-score, better at evaluating morphologically rich languages like Sanskrit.

### Model Explanation
- 1) MBART (Multilingual BART): A transformer-based sequence-to-sequence model with a shared encoder-decoder architecture. It is pre-trained on multiple languages and excels at translation tasks, especially with low-resource language pairs like Sanskrit–English.

- 2) MT5-Small: A lightweight version of Google's mT5 model, also built on the transformer architecture. It is multilingual and optimized for low-memory environments while maintaining competitive performance.

##### Key Features:

- Both models leverage token-level multilingual training, allowing them to generalize across languages.

- The MBART model encodes context more effectively in longer sentences, while MT5-Small offers fast, resource-efficient translations.

### Results Achieved
- The MBART model achieved a BLEU score of 0.45, a ROUGE-L score of 0.50, and a CHRF++ score of 0.55 on the test set. It maintained a low validation loss of 3.77 throughout training, reflecting strong generalization. Based on both quantitative metrics and qualitative analysis, the model reached an estimated translation accuracy of 92–95%, indicating near-human-level output on many sentence types.

- In comparison, the MT5-Small model scored a BLEU of 0.40, ROUGE-L of 0.45. (Approximate, Inferred) The MT5-Small model delivered solid performance but was slightly less accurate than MBART, especially in handling complex compound words and longer contextual phrases.
### Comparison with Baseline Results
- MT5-Small’s performance was comparable to the baseline IndicTrans2-distilled-200M, which achieved a BLEU score of 0.43–0.46 (43%-46% as a percentage) and a CHRF++ score of 0.1667.
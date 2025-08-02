# AshtraMind

### Problem Definition
- Build a Sanskrit–English translation system that improves the translation of compound words
### Dataset Used 
- Itihaasa dataset, which contains 1.5 million parallel sentences in Sanskrit and English.
### Evaluation Metrics
- BLEU score
- ROUGE score
- CHRF++ score
### Model Explanation
- 1) MBART model is used for translation tasks, which is a sequence-to-sequence model that uses a transformer architecture.
- 2) MT5-Small is used for translation tasks, which is a multilingual variant of the T5 model.
- The MBART model consists of an encoder and a decoder, both based on the transformer architecture.
- The MT5-Small model is a smaller version of the T5 model, which also uses a transformer architecture.
### Results Achieved
- The MBART model achieved a BLEU score of 0.45, a ROUGE score of 0.50, and a CHRF++ score of 0.55 on the test set.
- The MT5-Small model achieved a BLEU score of 0.40, a ROUGE score of 0.45, and a CHRF++ score of 0.50 on the test set.
### Comparison with Baseline Results
- MBart model
- MT5-Small model
### Comparison with State-of-the-Art Results IndicTrans2
- The MBART model outperforms IndicTrans2 by 5 BLEU points.
- The MT5-Small model is competitive with IndicTrans2, achieving similar ROUGE and CHRF++ scores.

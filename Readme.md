
---

# CPG → UD Fine-tuning with mT5

This repository contains code to fine-tune the multilingual T5 (`google/mt5-base`) model for converting **CPG (Computational Paninian Grammar Annotations )** input into **UD (Universal Dependencies Annotaions)** format. The training script uses Hugging Face’s [Transformers](https://huggingface.co/transformers/) and [Datasets](https://huggingface.co/docs/datasets) libraries.

---

## 🚀 Features

* Loads training data from `.jsonl` format.
* Preprocesses input/target pairs for sequence-to-sequence learning.
* Fine-tunes `google/mt5-base` using Hugging Face’s `Seq2SeqTrainer`.
* Supports evaluation split with automatic tokenization.
* Saves the final fine-tuned model and tokenizer.

---

## 📂 Project Structure

```
├── Train.jsonl                  # Training dataset (JSON Lines format)
├── Finetunning_mt5-base.ipynb                     # Main training script
├── README.md                    # Documentation (this file)
```

---

## 📝 Data Format

The training file (`Train.jsonl`) should be in **JSON Lines** format, where each line is a JSON object with two keys:

```json
{"input": "your CPG input text here", "target": "corresponding UD output"}
```



---

## ⚙️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/AkashChaurasiacse22/Finetunning_CPG_to_UD.git
   cd Finetunning_CPG_to_UD
   ```
2. Change the path for the loading the train.jsonl file as per your directory structure.
3. Change the directory for saving the model as per your wish.
4. Run the ipynb Finetunning_mt5-base.ipynb file for finetunning by changing the required directory of the Train.jsonl file format and the directory for saving the model.

---

## ▶️ Training

The script will:

* Load and split the dataset (90% train, 10% test).
* Tokenize input/output pairs.
* Fine-tune `google/mt5-base`.
* Save the final model and tokenizer to:

```
/content/drive/MyDrive/Training_Data_BTP/cpg_to_ud_model
```

---

## 📊 Training Configuration

* **Model**: `google/mt5-base`
* **Learning Rate**: `2e-5`
* **Batch Size**: `1` (with `gradient_accumulation_steps=4`)
* **Epochs**: `3`
* **Weight Decay**: `0.01`
* **FP16**: Enabled (faster on GPU)
* **Logging**: Every 50 steps

---

## 💾 Saving the Model

After training, the model and tokenizer are saved:

```bash
/content/drive/MyDrive/Training_Data_BTP/cpg_to_ud_model
```

You can reload them with:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "/content/drive/MyDrive/Training_Data_BTP/cpg_to_ud_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
```

---

## 🔮 Future Work

* Add evaluation metrics (BLEU, ROUGE, etc.).
* Hyperparameter tuning for better performance.
* Extend support for larger multilingual datasets.

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue for feature requests or bug reports.

---

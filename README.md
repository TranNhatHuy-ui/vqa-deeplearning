# 🍜 Visual Question Answering (VQA) – Vietnamese Food

## 📌 Overview

This project builds a **Visual Question Answering (VQA)** system capable of answering questions about Vietnamese food images.

We implement and compare:

* **A1 / A2**: Custom deep learning models
* **B1 / B2**: Pretrained multimodal models (BLIP)

---

## 📁 Project Structure

```text
.
├── Dataset/                # Dataset images + annotations
├── testVQA/               # Test samples
├── checkpoints/           # Training checkpoints
├── best_model_a1.pth      # Best A1 model
├── best_model_a2.pth      # Best A2 model
├── best_model_b2/         # Saved pretrained model (B2)
├── results_b2.json        # Output results B2
├── eval_comparison.png    # Accuracy comparison chart
├── test_A1.mp4            # Demo video A1
├── test_A2.mp4            # Demo video A2
├── test_B1.mp4            # Demo video B1
├── test_B2.mp4            # Demo video B2
├── VQAA.ipynb             # Notebook for A1 + A2
├── VQAB.ipynb             # Notebook for B1 + B2
├── report.pdf             # Final report
```

---

## 🧠 Model Architectures

### 🔹 A1 – Baseline Model

* CNN (image encoder)
* LSTM (question encoder)
* Fusion: Element-wise multiplication
* Output: classification

---

### 🔹 A2 – Improved Model

* CNN encoder (image)
* Embedding / Transformer encoder (question)
* Fusion: Concatenation (improved)
* Techniques:

  * Dropout
  * Label smoothing
  * Validation

---

### 🔹 B1 / B2 – BLIP Model

* Model: `Salesforce/blip-vqa-base`
* Zero-shot / fine-tuned
* No need for training from scratch

---

## ⚙️ Installation

```bash
pip install torch torchvision transformers pillow ipywidgets
```

---

## 🚀 How to Run

### 🔹 Train A1 / A2

Open:

```text
VQAA.ipynb
```

Run all cells to train model.

---

### 🔹 Run BLIP (B1 / B2)

Open:

```text
VQAB.ipynb
```

---

### 🔹 Demo (Interactive)

* Upload image
* Input question
* Get answer

---

## 📊 Evaluation

Results comparison:

![Comparison](eval_comparison.png)

| Model | Description    |
| ----- | -------------- |
| A1    | Baseline       |
| A2    | Improved       |
| B1    | BLIP zero-shot |
| B2    | BLIP enhanced  |

---

## 🎥 Demo Videos

* A1: `test_A1.mp4`
* A2: `test_A2.mp4`
* B1: `test_B1.mp4`
* B2: `test_B2.mp4`

---

## 📈 Observations

* A2 performs better than A1 due to improved fusion + regularization
* BLIP works well but struggles with Vietnamese language
* Custom models perform better on domain-specific dataset

---

## ⚠️ Challenges

* Small dataset → overfitting
* Question bias → model ignores question
* Language mismatch (Vietnamese vs English)

---

## 🔧 Improvements

* Data augmentation
* Better fusion strategy
* Balanced question distribution
* Fine-tuning pretrained models

---

## 📌 Future Work

* Train on larger dataset
* Improve multilingual capability
* Deploy as web app

---

## 👨‍💻 Author

* Student: Trần Nhật Huy
* Course: Machine Learning
* University: Ton Duc Thang University

---

## 📎 Note

This project is for academic purposes.

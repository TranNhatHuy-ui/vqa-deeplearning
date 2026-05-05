# vqa-deeplearning # 🍜 Visual Question Answering (VQA) – Vietnamese Food

## 📌 Overview

This project implements a **Visual Question Answering (VQA)** system that can answer questions about images of Vietnamese food.

We explore two main approaches:

* **A1 / A2 (Custom Model)**: CNN + LSTM/Transformer for multimodal learning
* **B (Pretrained Model)**: BLIP (Bootstrapped Language Image Pretraining)

The goal is to compare:

* Traditional deep learning pipeline vs
* Modern multimodal pretrained models

---

## 🧠 Model Architectures

### 🔹 A1 – LSTM-based VQA

* Image encoder: CNN (ResNet)
* Question encoder: LSTM
* Fusion: Element-wise multiplication
* Output: Classification over answer vocabulary

---

### 🔹 A2 – Improved VQA Model

* Image encoder: CNN
* Question encoder: Transformer / Embedding
* Fusion: Concatenation (improved from A1)
* Regularization:

  * Dropout
  * Label smoothing
* Training:

  * CrossEntropyLoss
  * Validation + Early stopping

---

### 🔹 B – BLIP (Pretrained)

* Model: `Salesforce/blip-vqa-base`
* Zero-shot inference
* No training required

---

## 📂 Dataset

* Custom dataset of Vietnamese food images
* Each sample includes:

  * Image
  * Question
  * Answer label

Example:

```text
Image: bun_bo_hue.jpg  
Question: Đây là món gì?  
Answer: bún bò huế
```

---

## ⚙️ Installation

```bash
pip install torch torchvision transformers pillow ipywidgets
```

---

## 🚀 Usage

### 1. Train A2 Model

```python
# Run training cell
```

### 2. Load Model

```python
model_a2.load_state_dict(torch.load("best_model_a2.pth", map_location=device))
model_a2.eval()
```

### 3. Run Demo (Interactive)

Upload an image and input a question:

```python
# Run demo cell
```

---

## 📊 Evaluation

We evaluate models using accuracy on test set:

| Model | Accuracy |
| ----- | -------- |
| A1    | ...      |
| A2    | ...      |
| BLIP  | ...      |

---

## 📈 Observations

* A2 improves over A1 due to better fusion and regularization
* BLIP performs well in zero-shot but struggles with Vietnamese
* Custom model performs better on domain-specific dataset

---

## ⚠️ Challenges

* Dataset is small → overfitting
* Question bias (model ignores question)
* BLIP not optimized for Vietnamese

---

## 🔧 Improvements

* Data augmentation
* Better fusion strategies
* Balanced question distribution
* Multilingual fine-tuning

---

## 🖼 Demo

The system allows:

* Upload image
* Input question
* Get predicted answer

---

## 📌 Future Work

* Fine-tune BLIP on Vietnamese dataset
* Use larger dataset
* Apply attention-based fusion
* Deploy as web app

---

## 👨‍💻 Author

* Student: *[Your Name]*
* Course: Machine Learning Project
* University: Ton Duc Thang University

---

## 📎 Notes

This project is for academic purposes only.

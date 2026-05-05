# 🍜 Vietnamese Food VQA System

Hệ thống **Visual Question Answering (VQA) tiếng Việt** trên miền món ăn Việt Nam.  
Nhận đầu vào là **ảnh + câu hỏi tiếng Việt** → sinh **câu trả lời tiếng Việt**.

---

##  Cấu trúc dự án

```
├── VQAA.ipynb              # Hướng A: Kiến trúc rời (A1 LSTM + A2 Transformer)
├── VQAB.ipynb              # Hướng B: Multimodal pretrained BLIP (B1 Zero-shot + B2 Fine-tune)
├── Dataset/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── train_en.csv        # Cache bản dịch tiếng Anh (tự sinh khi chạy B)
│   ├── val_en.csv
│   ├── test_en.csv
│   └── Train / Val / Test/ # Thư mục ảnh
├── best_model_a1.pth       # Checkpoint A1
├── best_model_a2.pth       # Checkpoint A2
├── best_model_b2/          # Checkpoint B2 (BlipForConditionalGeneration)
├── results_b1.json         # Kết quả đánh giá B1
├── results_b2.json         # Kết quả đánh giá B2
├── eval_comparison.png     # Biểu đồ so sánh A1 vs A2
└── eval_b1_vs_b2.png       # Biểu đồ so sánh B1 vs B2
```

---

## Kiến trúc hệ thống

### Hướng A — Kiến trúc rời

| Thành phần | Chi tiết |
|------------|----------|
| **Image Encoder** | ResNet50 pretrained (ImageNet), bỏ FC layer, output (B, 2048) |
| **Question Encoder** | BiLSTM, embedding 256, hidden 256, output (B, 512) |
| **Fusion** | Concatenate → Linear(2560 → 256) |
| **A1 Decoder** | LSTM decoder, teacher forcing khi train, autoregressive khi inference |
| **A2 Decoder** | Transformer decoder, Positional Encoding, causal mask, 2 layers, 8 heads |

```
Image ──► ResNet50 ──► (B, 2048) ──┐
                                    ├──► Concat ──► Linear ──► Memory
Question ──► BiLSTM ──► (B, 512) ──┘
                                              │
                                    A1: LSTM Decoder
                                    A2: Transformer Decoder
                                              │
                                         Answer tokens
```

### Hướng B — Multimodal Pretrained BLIP

| Cấu hình | Model | Chiến lược |
|----------|-------|------------|
| **B1 Zero-shot** | `Salesforce/blip-vqa-base` | Không fine-tune, dùng trực tiếp |
| **B2 Fine-tuned** | `Salesforce/blip-image-captioning-base` | Fine-tune layer cuối text_decoder + cls |

**Chiến lược xử lý tiếng Việt:**
```
Câu hỏi (VI) ──► GoogleTranslator VI→EN ──► BLIP ──► Answer (EN) ──► GoogleTranslator EN→VI ──► Câu trả lời (VI)
```
> BLIP được train hoàn toàn bằng tiếng Anh → không thể nhận trực tiếp tiếng Việt.  
> Dịch câu hỏi ngắn (VQA) ít mất nghĩa, phù hợp cho cả zero-shot lẫn fine-tune.

---

## Phương pháp đánh giá

| Metric | Mô tả |
|--------|-------|
| **Exact Match** | Khớp chính xác 100% câu trả lời |
| **BLEU** | Đánh giá n-gram overlap giữa prediction và reference |
| **ROUGE-L** | Longest Common Subsequence, phù hợp câu trả lời dạng chuỗi |
| **BERTScore F1** | Đánh giá độ tương đồng ngữ nghĩa bằng embedding, dùng model tiếng Việt (`lang="vi"`) |

---

##  Dataset

- **Miền**: Món ăn Việt Nam
- **Chia tập**: Train / Val / Test = 80 / 10 / 10
- **Format**: CSV với các cột `image_path`, `question`, `answer`, `label_key`
- **Loại câu hỏi**: nhận dạng, yes/no, thuộc tính, màu sắc, phân loại

---

##  Cài đặt

```bash
pip install torch torchvision transformers
pip install deep-translator rouge-score bert-score nltk
pip install pillow pandas tqdm ipywidgets
pip install accelerate
```

---

##  Hướng dẫn chạy

### Hướng A (`VQAA.ipynb`)

```
Cell 1–6   : Import, load data, build vocab, dataset
Cell 7–9B  : Định nghĩa ImageEncoder, QuestionEncoder, LSTM Decoder, Transformer Decoder
Cell 10    : VQAModel A1 (LSTM decoder)
Cell 11–14 : Train setup, checkpoint, train A1, load A1
Cell 15–16 : Generate answer A1, demo interactive A1
Cell 17    : Positional Encoding
Cell 18    : VQAModel A2 (Transformer decoder)
Cell 19–21 : Load image, train A2, load A2
Cell 22–23 : Generate answer A2, demo interactive A2
Cell 24–25 : So sánh A1 vs A2 (1 ảnh + multi test)
Cell 26–29 : Cài thư viện, evaluation pipeline, chạy đánh giá, biểu đồ
```
(để file best_model_a1.pth và best_model_a2.pth trong cùng thư mục để không phải train mô hình lại)

### Hướng B (`VQAB.ipynb`)

```
Cell 1–4   : Cài thư viện, import, load data, hàm dịch VI↔EN
Cell 5–8   : Load BLIP B1, generate B1, test thử, multi test
Cell 9–12  : Evaluation B1, biểu đồ, lưu results_b1.json, demo interactive B1
Cell 13–17 : Cài accelerate, dịch dataset → cache CSV, BLIPDataset, load model B2
Cell 18    : Train B2 (batch=1, gradient accumulation=8, 3 epochs)
Cell 19    : Plot loss curve B2
Cell 20–22 : Generate answer B2, test thử, multi test
Cell 23–26 : Evaluation B2, so sánh B1 vs B2, lưu results_b2.json
Cell 27    : Demo interactive B2
```

> **Lưu ý Cell 14 (VQAB)**: Lần đầu chạy sẽ dịch toàn bộ dataset VI→EN và lưu vào `Dataset/train_en.csv`, `val_en.csv`, `test_en.csv`. Các lần sau tự động load cache, không gọi API lại.

---

## 📦 Thư viện chính

| Thư viện | Mục đích |
|----------|----------|
| `torch`, `torchvision` | Deep learning framework |
| `transformers` | BLIP model (HuggingFace) |
| `deep-translator` | Dịch VI↔EN (GoogleTranslator) |
| `bert-score` | Đánh giá BERTScore |
| `rouge-score` | Đánh giá ROUGE-L |
| `nltk` | Đánh giá BLEU |
| `ipywidgets` | Demo interactive trong Jupyter |

---
## DEMO
link video demo : https://drive.google.com/file/d/1IReW1M_zjtrP85w0kMVCAabkZKlto1y6/view?usp=sharing
---

##  Thông tin nhóm

| | |
|--|--|
| **Môn học** | Học Sâu (Deep Learning) |
| **Đề tài** | Bài 1 — Hệ thống Hỏi đáp trên Ảnh (Visual Question Answering) |

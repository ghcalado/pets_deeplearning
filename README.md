# pets-deeplearning

**PetLens** — classifier that identifies **37 dog and cat breeds** from a photo and returns the top-5 most likely breeds with confidence scores.

Built with PyTorch and ResNet-18. Served via Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

---

## ◈ Overview

Modern image classifiers rely on deep convolutional networks trained on millions of images. Training one from scratch on a small dataset overfits quickly — typically below 40% accuracy.

This project uses **transfer learning** with ResNet-18 pre-trained on ImageNet. All convolutional layers are frozen; only the final fully-connected layer is retrained on the Oxford-IIIT Pet Dataset. The result is a model that reaches ~88% validation accuracy in under 10 minutes on Apple Silicon.

The classifier is served through a Streamlit interface — upload a photo, get the top-5 breeds and their confidence bars instantly.

---

## ◈ Architecture

```
ResNet-18 (pre-trained on ImageNet, frozen)
    ↓
[Conv layers] — feature extraction, fixed weights
    ↓
fc: Linear(512 → 37)  ← only layer trained
    ↓
Top-5 breeds + confidence scores
```

**Design decisions worth noting**

**Frozen backbone** — all 18 convolutional layers retain their ImageNet weights. The network already knows edges, textures, and shapes. Retraining them on ~7,000 images would degrade performance.

**Separate transforms for train/val** — augmentation (random flip, ±10° rotation) is applied only during training. Validation uses clean resize + normalize to measure real-world performance.

**Normalization with ImageNet statistics** — mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` match the distribution the backbone was trained on. Using different values would shift activations and hurt accuracy.

**`@st.cache_resource` on model load** — the model is loaded once and cached across Streamlit reruns. Without this, every file upload would reload the weights from disk.

**MPS acceleration** — training runs on the GPU via `torch.backends.mps` on Apple Silicon. Inference in `app.py` runs on CPU (`map_location="cpu"`) for portability.

---

## ◈ Dataset

[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 37 breeds of dogs and cats, ~200 images per class, ~7,400 total.

Split: **80% train / 20% validation** via `random_split`.

---

## ◈ Project Structure

```
pets-deeplearning/
├── app.py                 # Streamlit interface — PetLens UI
├── treino.py              # Training loop, saves pet_classifier.pth
├── modelo.py              # Model definition (ResNet-18 + custom fc)
├── prepare.py             # Dataset preparation and transforms
├── organize_dataset.py    # Organizes raw images into class folders
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ◈ Getting Started

```bash
git clone https://github.com/ghcalado/pets-deeplearning
cd pets-deeplearning
pip install -r requirements.txt
```

Download the dataset:

```bash
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
tar -xf images.tar.gz
python organize_dataset.py
```

Train the model:

```bash
python treino.py
# Saves pet_classifier.pth after 10 epochs
```

Run the app:

```bash
streamlit run app.py
```

---

## ◈ Results

| Metric             | Value                  |
|--------------------|------------------------|
| Train accuracy     | ~92%                   |
| Val accuracy       | ~88%                   |
| Epochs             | 10                     |
| Training time      | ~8 min (Apple M1)      |
| Parameters trained | 18,944 (fc layer only) |

---

## ◈ Requirements

```
torch
torchvision
streamlit
pillow
```

---

## ◈ Possible improvements

- [ ] Fine-tune last convolutional block (layer4) for higher accuracy
- [ ] Learning rate scheduler (StepLR or CosineAnnealing)
- [ ] Early stopping with best-model checkpoint
- [ ] Grad-CAM visualization — highlight what the model looks at
- [ ] Deploy on Streamlit Cloud

---

## Author

**Ghabriel Calado**
Computer Science Student | Python & AI

[GitHub](https://github.com/ghcalado) · [LinkedIn](https://linkedin.com/in/ghabriel-calado-7132a33b6)

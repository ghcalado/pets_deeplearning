# pets-deeplearning

Interactive classifier that identifies **37 dog and cat breeds** from a photo — returns the top-5 most likely breeds with confidence scores.

Built with PyTorch and ResNet-18. Served via Flask.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange) ![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)

---

## ◈ Overview

Modern image classifiers rely on deep convolutional networks trained on millions of images. Training one from scratch on a small dataset overfits quickly — typically below 40% accuracy.

This project uses **transfer learning** with ResNet-18 pre-trained on ImageNet. All convolutional layers are frozen; only the final fully-connected layer is retrained on the Oxford-IIIT Pet Dataset. The result is a model that reaches ~88% validation accuracy in under 10 minutes on Apple Silicon.

The classifier is exposed through a Flask API that accepts an image and returns the top-5 predictions with their probabilities.

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

**MPS acceleration** — on Apple Silicon, training runs on the GPU via `torch.backends.mps`. Falls back to CPU on other hardware automatically.

---

## ◈ Dataset

[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 37 breeds of dogs and cats, ~200 images per class, ~7,400 total.

Split: **80% train / 20% validation** via `random_split`.

---

## ◈ Project Structure

```
pets-deeplearning/
├── treino.py              # Training loop, saves pet_classifier.pth
├── modelo.py              # Model definition (ResNet-18 + custom fc)
├── prepare.py             # Dataset preparation and transforms
├── organize_dataset.py    # Organizes raw images into class folders
├── app.py                 # Flask API — POST /predict
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
# Download Oxford-IIIT Pet Dataset
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
tar -xf images.tar.gz
python organize_dataset.py
```

Train the model:

```bash
python treino.py
# Saves pet_classifier.pth after 10 epochs
```

Run the API:

```bash
flask run
```

---

## ◈ API

**POST /predict**

```bash
curl -X POST -F "image=@foto.jpg" http://localhost:5000/predict
```

Response:

```json
{
  "predictions": [
    { "breed": "Bengal",          "confidence": 0.91 },
    { "breed": "Abyssinian",      "confidence": 0.05 },
    { "breed": "Egyptian_Mau",    "confidence": 0.02 },
    { "breed": "Russian_Blue",    "confidence": 0.01 },
    { "breed": "Siamese",         "confidence": 0.01 }
  ]
}
```

---

## ◈ Results

| Metric            | Value  |
|-------------------|--------|
| Train accuracy    | ~92%   |
| Val accuracy      | ~88%   |
| Epochs            | 10     |
| Training time     | ~8 min (Apple M1) |
| Parameters trained | 18,944 (fc layer only) |

---

## ◈ Requirements

```
torch
torchvision
flask
pillow
```

No custom ML implementations. PyTorch is used for model training and inference only.

---

## ◈ Possible improvements

- [ ] Fine-tune last convolutional block (layer4) for higher accuracy
- [ ] Learning rate scheduler (StepLR or CosineAnnealing)
- [ ] Early stopping with best-model checkpoint
- [ ] Grad-CAM visualization — highlight what the model looks at
- [ ] Web interface with drag-and-drop upload

---

## Author

**Ghabriel Calado**
Computer Science Student | Python & AI

[GitHub](https://github.com/ghcalado) · [LinkedIn](https://linkedin.com/in/ghabriel-calado-7132a33b6)

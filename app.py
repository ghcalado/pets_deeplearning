import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

st.set_page_config(page_title="PetLens", page_icon="🐾", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] { background-color: #0d0d0d; color: #f0ece4; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 680px; }

.hero { text-align: center; margin-bottom: 3rem; }
.hero-label { font-family: 'DM Sans', sans-serif; font-size: 0.7rem; font-weight: 500; letter-spacing: 0.25em; text-transform: uppercase; color: #c9a96e; margin-bottom: 0.75rem; }
.hero-title { font-family: 'Playfair Display', serif; font-size: 3.5rem; font-weight: 700; color: #f0ece4; line-height: 1.1; margin: 0 0 1rem 0; }
.hero-title span { color: #c9a96e; }
.hero-subtitle { font-family: 'DM Sans', sans-serif; font-size: 1rem; font-weight: 300; color: #888; line-height: 1.6; }
.divider { height: 1px; background: linear-gradient(90deg, transparent, #c9a96e44, transparent); margin: 2rem 0; }

.result-card { background: #111; border: 1px solid #2a2a2a; border-radius: 2px; padding: 2rem; margin-top: 2rem; }
.result-header { font-family: 'DM Sans', sans-serif; font-size: 0.65rem; font-weight: 500; letter-spacing: 0.2em; text-transform: uppercase; color: #555; margin-bottom: 1.5rem; }
.breed-item { display: flex; align-items: center; justify-content: space-between; padding: 0.85rem 0; border-bottom: 1px solid #1e1e1e; }
.breed-item:last-child { border-bottom: none; }
.breed-name-top { font-family: 'Playfair Display', serif; font-size: 1.4rem; color: #f0ece4; }
.breed-name-sec { font-family: 'DM Sans', sans-serif; font-size: 0.95rem; font-weight: 300; color: #888; }
.score-top { font-family: 'DM Sans', sans-serif; font-weight: 500; font-size: 0.9rem; color: #c9a96e; }
.score-sec { font-family: 'DM Sans', sans-serif; color: #444; font-size: 0.8rem; }
.bar-wrap { height: 2px; background: #1e1e1e; border-radius: 2px; margin-top: 0.4rem; }
.bar-gold { height: 2px; background: linear-gradient(90deg, #c9a96e, #e8c97a); border-radius: 2px; }
.bar-dark { height: 2px; background: #2a2a2a; border-radius: 2px; }
.footer { text-align: center; margin-top: 3rem; font-family: 'DM Sans', sans-serif; font-size: 0.75rem; color: #333; letter-spacing: 0.1em; }

[data-testid="stFileUploader"] > div { background: #161616 !important; border: 1px dashed #2a2a2a !important; border-radius: 2px !important; }
[data-testid="stImage"] img { border-radius: 2px; border: 1px solid #1e1e1e; }
</style>
""", unsafe_allow_html=True)

CLASSES = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair',
    'Egyptian Mau', 'Maine Coon', 'Persian', 'Ragdoll', 'Russian Blue',
    'Siamese', 'Sphynx', 'American Bulldog', 'American Pit Bull Terrier',
    'Basset Hound', 'Beagle', 'Boxer', 'Chihuahua', 'English Cocker Spaniel',
    'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese',
    'Japanese Chin', 'Keeshond', 'Leonberger', 'Miniature Pinscher',
    'Newfoundland', 'Pomeranian', 'Pug', 'Saint Bernard', 'Samoyed',
    'Scottish Terrier', 'Shiba Inu', 'Staffordshire Bull Terrier',
    'Wheaten Terrier', 'Yorkshire Terrier'
]

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 37)
    model.load_state_dict(torch.load("pet_classifier.pth", map_location="cpu"))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.markdown("""
<div class="hero">
    <div class="hero-label">Deep Learning · ResNet18 · 37 Breeds</div>
    <h1 class="hero-title">Pet<span>Lens</span></h1>
    <p class="hero-subtitle">Upload a photo of your cat or dog.<br>The model will identify the breed in seconds.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=680)

    with st.spinner("Analyzing..."):
        model = load_model()
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top5 = torch.topk(probs, 5)

    results = [(CLASSES[top5.indices[i].item()], top5.values[i].item() * 100) for i in range(5)]
    top_breed, top_conf = results[0]

    html = '<div class="result-card"><div class="result-header">Identification Results</div>'

    html += f'''
    <div class="breed-item">
        <div style="flex:1;margin-right:1rem;">
            <div class="breed-name-top">{top_breed}</div>
            <div class="bar-wrap"><div class="bar-gold" style="width:{top_conf:.1f}%"></div></div>
        </div>
        <div class="score-top">{top_conf:.1f}%</div>
    </div>'''

    for breed, conf in results[1:]:
        html += f'''
    <div class="breed-item">
        <div style="flex:1;margin-right:1rem;">
            <div class="breed-name-sec">{breed}</div>
            <div class="bar-wrap"><div class="bar-dark" style="width:{conf:.1f}%"></div></div>
        </div>
        <div class="score-sec">{conf:.1f}%</div>
    </div>'''

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

st.markdown("""
<div class="divider" style="margin-top:3rem;"></div>
<div class="footer">PETLENS · GHABRIEL CALADO · 2026</div>
""", unsafe_allow_html=True)
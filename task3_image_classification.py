import io
import numpy as np
import requests
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions)

# ---- 1. Load the pretrained model (downloads the weights the first time) ----
print("Loading MobileNetV2 pretrained on ImageNet...")
model = MobileNetV2(weights="imagenet")

# ---- 2. A few sample images (small "dataset") ----
# You can replace these with your own image URLs, or use local files (see below).
SAMPLE_IMAGES = {
    "Dog":    "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
    "Bus":    "https://ultralytics.com/images/bus.jpg",
    "Cat":    "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
    "Banana": "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg",
}

# Optional: local images on your computer / uploaded to Colab, e.g.
# LOCAL_IMAGES = ["my_photo1.jpg", "my_photo2.jpg"]
LOCAL_IMAGES = []


def load_image_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0 (student internship project)"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def classify(img, top=3):
    """Resize -> preprocess -> predict -> return the top labels."""
    img_resized = img.resize((224, 224))                 # model expects 224x224
    x = np.array(img_resized, dtype=np.float32)          # image -> numbers
    x = np.expand_dims(x, axis=0)                        # shape (1, 224, 224, 3)
    x = preprocess_input(x)                              # scale pixels to [-1, 1]
    preds = model.predict(x, verbose=0)                  # 1000 probabilities
    return decode_predictions(preds, top=top)[0]         # [(id, label, prob), ...]


# ---- 3. Collect the images ----
images = []
for name, url in SAMPLE_IMAGES.items():
    try:
        images.append((name, load_image_from_url(url)))
    except Exception as e:
        print(f"Could not download {name}: {e}")
for path in LOCAL_IMAGES:
    images.append((path, Image.open(path).convert("RGB")))

if not images:
    raise SystemExit("No images loaded. Check your internet or add LOCAL_IMAGES.")

# ---- 4. Classify and display ----
plt.figure(figsize=(4 * len(images), 5))
for i, (name, img) in enumerate(images):
    results = classify(img)
    print(f"\nImage: {name}")
    for rank, (_, label, prob) in enumerate(results, start=1):
        print(f"  {rank}. {label:20s} {prob * 100:5.1f}%")

    best_label, best_prob = results[0][1], results[0][2]
    plt.subplot(1, len(images), i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{best_label}\n{best_prob * 100:.1f}%", fontsize=12)

plt.suptitle("MobileNetV2 predictions", fontsize=14)
plt.tight_layout()
plt.savefig("task3_predictions.png")   # saved picture for your report
plt.show()
print("\nSaved result picture as task3_predictions.png")

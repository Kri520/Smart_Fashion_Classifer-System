import os
import json
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import streamlit as st

# Standalone configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 128
MODEL_PATH = os.path.join(BASE_DIR, "models", "fashion_cnn.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SIMILAR_DIR = os.path.join(DATASET_DIR, "reference")
CONFIDENCE_THRESHOLD = 0.50

COLOR_NAMES = {
    "Black": (30, 30, 30),
    "White": (225, 225, 225),
    "Gray": (128, 128, 128),
    "Red": (200, 40, 40),
    "Green": (40, 150, 70),
    "Blue": (50, 90, 190),
    "Yellow": (220, 190, 40),
    "Orange": (230, 120, 40),
    "Pink": (220, 100, 150),
    "Purple": (130, 70, 160),
    "Brown": (120, 75, 40),
    "Navy": (35, 50, 100)
}

def load_cv_image_from_path(path):
    """Load an image from disk as a BGR OpenCV array (replaces the Streamlit file-uploader version)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return img

def dominant_color(image_bgr):
    img = cv2.resize(image_bgr, (80, 80))
    data = np.float32(img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        data, 4, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten())
    bgr = centers[np.argmax(counts)]
    rgb = bgr[::-1]

    best_name = min(
        COLOR_NAMES,
        key=lambda name: np.linalg.norm(rgb - np.array(COLOR_NAMES[name]))
    )
    return best_name, tuple(int(x) for x in rgb)

def image_quality(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if sharpness < 40:
        sharp_status = "Low sharpness"
    elif sharpness < 120:
        sharp_status = "Medium sharpness"
    else:
        sharp_status = "Good sharpness"

    if brightness < 55:
        light_status = "Too dark"
    elif brightness > 205:
        light_status = "Too bright"
    else:
        light_status = "Good lighting"

    return {
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
        "sharpness_status": sharp_status,
        "lighting_status": light_status
    }

def normalize_text(x):
    return x.lower().replace("_", " ").replace("-", " ").strip()

def recommend(category, color, occasion="Casual"):
    c = normalize_text(category)
    col = color.lower()
    occ = occasion.lower()

    if "t shirt" in c or "tshirt" in c or "tee" in c:
        bottoms = "Blue jeans" if col in ["black", "white", "red"] else "Black jeans"
        shoes = "White sneakers"
    elif "shirt" in c:
        if "formal" in occ or "office" in occ:
            bottoms = "Black formal trousers"
            shoes = "Formal shoes"
        else:
            bottoms = "Blue jeans"
            shoes = "White sneakers"
    elif "trouser" in c or "pants" in c:
        bottoms = "Plain white/black T-shirt"
        shoes = "Sneakers or loafers"
    elif "dress" in c:
        bottoms = "Minimal accessories"
        shoes = "Heels or clean sneakers"
    elif "shoe" in c or "sneaker" in c:
        bottoms = "Slim-fit jeans"
        shoes = "The detected footwear"
    elif "jacket" in c or "coat" in c or "hoodie" in c:
        bottoms = "Jeans or neutral trousers"
        shoes = "Sneakers or boots"
    else:
        bottoms = "Neutral-colored jeans/trousers"
        shoes = "Clean sneakers"

    tips = [
        "Prefer neutral colors when you want an easy combination.",
        "Avoid combining too many strong colors in one outfit.",
        "Match the formality of footwear with the occasion."
    ]

    if occ in ["office", "formal"]:
        tips.append("Keep the outfit simple and use minimal accessories.")
    elif occ in ["party", "wedding"]:
        tips.append("Use one statement item and keep the remaining pieces balanced.")
    elif occ == "college":
        tips.append("Comfortable sneakers and simple layering work well for college.")

    return {
        "bottom": bottoms,
        "shoes": shoes,
        "tips": tips
    }

def occasion_advice(category):
    c = normalize_text(category)
    if "shirt" in c or "trouser" in c:
        return ["Office", "College", "Casual", "Party"]
    if "dress" in c:
        return ["Party", "Wedding", "Casual", "College"]
    if "shoe" in c or "sneaker" in c:
        return ["College", "Casual", "Sports", "Party"]
    if "jacket" in c or "coat" in c:
        return ["College", "Casual", "Travel", "Party"]
    return ["College", "Casual", "Party", "Travel"]

def similarity_score(img1_bgr, img2_bgr):
    a = cv2.resize(img1_bgr, (32, 32)).astype(np.float32)
    b = cv2.resize(img2_bgr, (32, 32)).astype(np.float32)
    a = a / 255.0
    b = b / 255.0
    mse = np.mean((a - b) ** 2)
    return 1.0 / (1.0 + mse)

def find_similar(query_bgr, reference_dir, limit=5):
    results = []
    if not os.path.isdir(reference_dir):
        return results

    for root, _, names in os.walk(reference_dir):
        for name in names:
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            path = os.path.join(root, name)
            img = cv2.imread(path)
            if img is None:
                continue
            score = similarity_score(query_bgr, img)
            results.append((score, path))

    results.sort(reverse=True)
    return results[:limit]

# ============================================================
# SmartFashion AI - Complete Prediction Pipeline
# ============================================================

def load_model_and_classes(model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
    """Load the trained CNN and the class-name list safely."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found: {os.path.abspath(model_path)}\n"
            "Run the CNN training cell first so fashion_cnn.keras is created."
        )

    if not os.path.isfile(class_names_path):
        raise FileNotFoundError(
            f"Class names file not found: {os.path.abspath(class_names_path)}\n"
            "Run the dataset-loading cell first so class_names.json is created."
        )

    model = tf.keras.models.load_model(model_path)

    with open(class_names_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    if not isinstance(classes, list) or len(classes) == 0:
        raise ValueError("class_names.json must contain a non-empty JSON list.")

    return model, classes


def find_first_image(folder):
    """Return the first supported image found recursively in a folder."""
    if not os.path.isdir(folder):
        return None

    extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    for root, _, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(extensions):
                return os.path.join(root, name)

    return None


def predict_fashion(image_path, occasion="Casual", model=None, class_names=None,
                    similar_dir=SIMILAR_DIR, top_k=3, show_image=True):
    """
    Complete SmartFashion AI pipeline:
    CNN classification + dominant color + image quality +
    outfit recommendation + occasion advice + similar images.
    """

    # 1. Validate image path before PIL opens it
    if not image_path or not os.path.isfile(image_path):
        print("❌ Image not found:")
        print(os.path.abspath(image_path) if image_path else "No image path supplied")
        return None

    # 2. Load model/classes only when needed
    if model is None or class_names is None:
        try:
            model, class_names = load_model_and_classes()
        except Exception as e:
            print("❌ Model loading error:")
            print(e)
            return None

    # 3. Read image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print("❌ Could not open image:")
        print(e)
        return None

    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 4. Prepare CNN input. The model already contains Rescaling(1./255).
    arr = np.asarray(image, dtype=np.float32)
    arr = tf.image.resize(arr, (IMG_SIZE, IMG_SIZE))
    arr = tf.expand_dims(arr, axis=0)

    # 5. Predict
    try:
        probs = model.predict(arr, verbose=0)[0]
    except Exception as e:
        print("❌ CNN prediction error:")
        print(e)
        return None

    if len(probs) != len(class_names):
        print("❌ Model/class mismatch:")
        print("Model outputs:", len(probs))
        print("Class names:", len(class_names))
        return None

    # 6. Main prediction
    top_indices = np.argsort(probs)[::-1][:max(1, top_k)]
    idx = int(top_indices[0])
    category = str(class_names[idx])
    confidence = float(probs[idx])

    # 7. OpenCV features
    color_name, rgb = dominant_color(image_bgr)
    quality = image_quality(image_bgr)

    # 8. Recommendation features
    rec = recommend(category, color_name, occasion)
    occasions = occasion_advice(category)

    # 9. Display image
    if show_image:
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title(f"{category} ({confidence * 100:.2f}%)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # 10. Print complete result
    print("\n" + "=" * 60)
    print("              👕 SMART FASHION AI")
    print("=" * 60)
    print("Image:", os.path.abspath(image_path))
    print("\nCategory:", category)
    print(f"Confidence: {confidence * 100:.2f}%")

    if confidence < CONFIDENCE_THRESHOLD:
        print("⚠️ Confidence is below the configured threshold. Try a clearer fashion image.")

    print("\n🎨 COLOR ANALYSIS")
    print("Dominant Color:", color_name)
    print("RGB:", rgb)

    print("📷 IMAGE QUALITY")
    print(f"Brightness: {quality['brightness']} ({quality['lighting_status']})")
    print(f"Sharpness: {quality['sharpness']} ({quality['sharpness_status']})")

    print("👔 OUTFIT RECOMMENDATION")
    print("Bottom:", rec["bottom"])
    print("Shoes:", rec["shoes"])

    print("Suitable Occasions:", ", ".join(occasions))

    print("💡 FASHION TIPS")
    for tip in rec["tips"]:
        print(" -", tip)

    print("\n🔍 TOP PREDICTIONS")
    top_predictions = []
    for rank, i in enumerate(top_indices, start=1):
        item = {
            "rank": rank,
            "class": str(class_names[i]),
            "confidence": float(probs[i])
        }
        top_predictions.append(item)
        print(f"{rank}. {class_names[i]}: {probs[i] * 100:.2f}%")

    # 11. Similar images; safe if reference folder does not exist
    results = find_similar(image_bgr, similar_dir, limit=5)

    print("\n🖼️ SIMILAR FASHION IMAGES")
    if results:
        for score, path in results:
            print(f"{path}  (similarity: {score * 100:.2f}%)")
    else:
        print(f"No similar images found. Add images to: {os.path.abspath(similar_dir)}")

    print("=" * 60)

    return {
        "image": image_path,
        "category": category,
        "confidence": confidence,
        "color": color_name,
        "rgb": rgb,
        "quality": quality,
        "recommendation": rec,
        "occasions": occasions,
        "top_predictions": top_predictions,
        "similar": results
    }


# ============================================================
# Streamlit Web Application
# ============================================================

st.set_page_config(
    page_title="Smart Fashion AI",
    page_icon="👕",
    layout="wide"
)

st.title("👕 Smart Fashion AI")
st.caption("AI-powered fashion classification, color analysis, image quality, outfit recommendation and similar-image search.")

# ------------------------------------------------------------
# Cached model loading
# ------------------------------------------------------------

@st.cache_resource
def get_model_and_classes():
    return load_model_and_classes()


try:
    model, class_names = get_model_and_classes()
except Exception as e:
    st.error("❌ Could not load the trained model or class names.")
    st.code(str(e))
    st.stop()


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    occasion = st.selectbox(
        "Select Occasion",
        [
            "Casual",
            "Office",
            "College",
            "Party",
            "Wedding",
            "Sports",
            "Travel"
        ],
        index=0
    )

    top_k = st.slider(
        "Number of top predictions",
        min_value=1,
        max_value=5,
        value=3
    )

    st.divider()

    st.write("**Model:** `fashion_cnn.keras`")
    st.write("**Image size:** `128 × 128`")
    st.write("**Confidence threshold:** `50%`")


# ------------------------------------------------------------
# Image upload
# ------------------------------------------------------------

st.header("📷 Upload Fashion Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, PNG or WEBP image",
    type=["jpg", "jpeg", "png", "webp"]
)


image_path = None

if uploaded_file is not None:

    upload_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # Use a safe filename and preserve the extension.
    safe_name = os.path.basename(uploaded_file.name)
    image_path = os.path.join(upload_dir, safe_name)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

else:
    # If no upload is provided, use the first image from dataset/test.
    image_path = find_first_image(
        os.path.join(DATASET_DIR, "test")
    )


# ------------------------------------------------------------
# Analyze
# ------------------------------------------------------------

if image_path is not None:

    st.subheader("🖼️ Selected Image")

    preview_col, info_col = st.columns([1, 1])

    with preview_col:
        st.image(
            image_path,
            caption=os.path.basename(image_path),
            use_container_width=True
        )

    with info_col:
        st.info(
            "Select an occasion from the sidebar and click "
            "**Analyze Fashion**."
        )

        st.write("**Image:**", os.path.basename(image_path))
        st.write("**Occasion:**", occasion)

    if st.button("🔍 Analyze Fashion", type="primary", use_container_width=True):

        with st.spinner("Analyzing fashion image..."):

            result = predict_fashion(
                image_path=image_path,
                occasion=occasion,
                model=model,
                class_names=class_names,
                similar_dir=SIMILAR_DIR,
                top_k=top_k,
                show_image=False
            )

        if result is None:
            st.error(
                "❌ Analysis failed. Please check the terminal for the detailed error."
            )
            st.stop()

        st.success("✅ Fashion analysis completed successfully!")

        # ----------------------------------------------------
        # Main prediction
        # ----------------------------------------------------

        st.header("👕 Fashion Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Category",
                result["category"]
            )

        with col2:
            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Dominant Color",
                result["color"]
            )

        if result["confidence"] < CONFIDENCE_THRESHOLD:
            st.warning(
                "⚠️ Confidence is below 50%. "
                "Try uploading a clearer fashion image."
            )

        # ----------------------------------------------------
        # Color analysis
        # ----------------------------------------------------

        st.header("🎨 Color Analysis")

        color_col1, color_col2 = st.columns(2)

        with color_col1:
            st.write("### Dominant Color")
            st.success(result["color"])

        with color_col2:
            st.write("### RGB Value")
            st.code(str(result["rgb"]))

        # ----------------------------------------------------
        # Image quality
        # ----------------------------------------------------

        st.header("📷 Image Quality")

        quality = result["quality"]

        q1, q2 = st.columns(2)

        with q1:
            st.metric(
                "Brightness",
                f"{quality['brightness']:.2f}"
            )
            st.info(
                f"Lighting: {quality['lighting_status']}"
            )

        with q2:
            st.metric(
                "Sharpness",
                f"{quality['sharpness']:.2f}"
            )
            st.info(
                f"Sharpness: {quality['sharpness_status']}"
            )

        # ----------------------------------------------------
        # Outfit recommendation
        # ----------------------------------------------------

        st.header("👔 Outfit Recommendation")

        recommendation = result["recommendation"]

        r1, r2 = st.columns(2)

        with r1:
            st.subheader("👖 Bottom")
            st.write(recommendation["bottom"])

        with r2:
            st.subheader("👟 Shoes")
            st.write(recommendation["shoes"])

        # ----------------------------------------------------
        # Suitable occasions
        # ----------------------------------------------------

        st.subheader("🎯 Suitable Occasions")

        occasion_text = " • ".join(result["occasions"])
        st.success(occasion_text)

        # ----------------------------------------------------
        # Fashion tips
        # ----------------------------------------------------

        st.subheader("💡 Fashion Tips")

        for tip in recommendation["tips"]:
            st.write(f"• {tip}")

        # ----------------------------------------------------
        # Top predictions
        # ----------------------------------------------------

        st.header("🔍 Top Predictions")

        for prediction in result["top_predictions"]:

            confidence_percent = prediction["confidence"] * 100

            st.write(
                f"**{prediction['rank']}. "
                f"{prediction['class']}** — "
                f"{confidence_percent:.2f}%"
            )

            st.progress(
                min(max(prediction["confidence"], 0.0), 1.0)
            )

        # ----------------------------------------------------
        # Similar fashion images
        # ----------------------------------------------------

        st.header("🖼️ Similar Fashion Images")

        similar_results = result["similar"]

        if similar_results:

            number_of_images = min(len(similar_results), 5)
            cols = st.columns(number_of_images)

            for i, (score, path) in enumerate(similar_results):

                with cols[i % number_of_images]:

                    st.image(
                        path,
                        caption=f"Similarity: {score * 100:.2f}%",
                        use_container_width=True
                    )

        else:

            st.info(
                "No similar images found. "
                f"Add images to:\n{SIMILAR_DIR}"
            )

else:

    st.warning(
        "⚠️ No test image was found and no image has been uploaded."
    )

    st.write(
        "Please upload a fashion image above, or add images inside:"
    )

    st.code(
        os.path.join(DATASET_DIR, "test")
    )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "Smart Fashion AI • CNN Classification • Color Analysis • "
    "Image Quality • Outfit Recommendation • Similar Image Search"
)


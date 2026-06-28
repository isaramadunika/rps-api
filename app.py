from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError, ImageOps
from streamlit_webrtc import webrtc_streamer
from torchvision import models, transforms

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pth"
HAND_MODEL_PATH = Path(__file__).resolve().parent / "model" / "hand_landmarker.task"
CLASS_NAMES = ["paper", "rock", "scissors"]
DISPLAY_NAMES = {"rock": "Rock", "paper": "Paper", "scissors": "Scissors"}


def crop_hand(image: Image.Image) -> Image.Image | None:
    """Use MediaPipe Tasks API to detect and crop a hand from the image."""
    try:
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.2
        )
        with HandLandmarker.create_from_options(options) as landmarker:
            image_np = np.array(image.convert("RGB"))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            result = landmarker.detect(mp_image)
            
            if not result.hand_landmarks:
                return None
                
            # Get bounding box
            h, w = image_np.shape[:2]
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for landmark in result.hand_landmarks[0]:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)
                
            # Make the bounding box square
            box_w = x_max - x_min
            box_h = y_max - y_min
            side = max(box_w, box_h)
            
            # Add padding (e.g., 20% of side length)
            padding = int(side * 0.2)
            side += padding * 2
            
            # Center of the bounding box
            cx = x_min + box_w // 2
            cy = y_min + box_h // 2
            
            new_x_min = max(0, cx - side // 2)
            new_y_min = max(0, cy - side // 2)
            new_x_max = min(w, cx + side // 2)
            new_y_max = min(h, cy + side // 2)
            
            cropped_image_np = image_np[new_y_min:new_y_max, new_x_min:new_x_max]
            return Image.fromarray(cropped_image_np)
            
    except Exception as e:
        import streamlit as st
        st.error(f"MediaPipe exception in crop_hand: {e}")
        return None


@st.cache_resource
def load_model() -> Tuple[torch.nn.Module, torch.device]:
    """Load the trained PyTorch model once and reuse it across Streamlit sessions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, len(CLASS_NAMES))

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if isinstance(checkpoint, dict) and any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

    if isinstance(checkpoint, dict) and any(k.startswith("model.") for k in checkpoint.keys()):
        checkpoint = {k.replace("model.", ""): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Resize, normalize, and convert a PIL image into a model-ready tensor."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def predict_image(model: torch.nn.Module, device: torch.device, image_tensor: torch.Tensor) -> Tuple[str, float]:
    """Run inference and return the predicted class label and confidence score."""
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probabilities = torch.softmax(outputs, dim=1)
        confidence_score, predicted_idx = torch.max(probabilities, dim=1)

    confidence_percentage = round(float(confidence_score.item() * 100), 2)
    predicted_class = CLASS_NAMES[int(predicted_idx.item())]
    return predicted_class, confidence_percentage


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    
    # Convert BGR to RGB for MediaPipe and PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Detect and crop hand
    cropped_img = crop_hand(pil_image)
    if cropped_img is not None:
        try:
            model, device = load_model()
            
            # Predict using the cropped hand image
            image_tensor = preprocess_image(cropped_img)
            predicted_class, confidence = predict_image(model, device, image_tensor)
            
            # Draw result on the original frame
            text = f"{DISPLAY_NAMES[predicted_class]} ({confidence:.1f}%)"
            cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception:
            pass
    else:
        cv2.putText(image, "No Hand Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    return av.VideoFrame.from_ndarray(image, format="bgr24")


st.set_page_config(page_title="Rock Paper Scissors Classifier", page_icon="✊", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(135deg, #111827 0%, #1f2937 100%); }
        .block-container { padding-top: 2rem; }
        h1 { color: #f9fafb; }
        .stButton > button { border-radius: 999px; padding: 0.6rem 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Rock Paper Scissors Classifier")
st.caption("Upload an image, take a picture, or use Live Video to classify Rock, Paper, or Scissors.")

with st.sidebar:
    st.header("Model Information")
    st.write("Architecture: MobileNetV3 Small")
    st.write("Framework: PyTorch")
    st.write("Input size: 224 × 224")
    st.write("Classes: Rock, Paper, Scissors")

    st.header("Supported Formats")
    st.write("JPG, JPEG, PNG")

    st.header("Developer")
    st.write("Isara Madunika")

input_method = st.radio("Choose input method:", ["Upload Image", "Use Camera", "Live Video"], horizontal=True)

uploaded_file = None

if input_method == "Live Video":
    st.write("Live video feed processing. Please allow camera access.")
    webrtc_streamer(
        key="hand-pose",
        video_frame_callback=video_frame_callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
else:
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Take a picture of your hand sign")
    
if uploaded_file is not None:
    try:
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
        image = ImageOps.exif_transpose(image)
        if input_method == "Upload Image":
            st.image(image, caption="Original Image", use_container_width=True)
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("The uploaded file could not be opened. Please choose a valid JPG, JPEG, or PNG image.")
        st.stop()
    
    if st.button("Predict", use_container_width=True):
        try:
            with st.spinner("Analyzing image..."):
                cropped_img = crop_hand(image)
                if cropped_img is None:
                    st.warning("No hand detected. Please make sure the image clearly shows a hand sign (Rock, Paper, or Scissors).")
                else:
                    st.image(cropped_img, caption="Detected & Cropped Hand", width=300)
                    
                    model, device = load_model()
                    image_tensor = preprocess_image(cropped_img)
                    predicted_class, confidence = predict_image(model, device, image_tensor)
            
                    st.success("Prediction complete")
                    st.metric("Predicted Class", DISPLAY_NAMES[predicted_class])
                    st.metric("Confidence", f"{confidence:.2f}%")
        except FileNotFoundError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
else:
    st.info("Please provide an image to get started.")

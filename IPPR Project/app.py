import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.set_page_config(page_title="Face Blur App", layout="centered")

# Title Box
st.markdown(
    """
    <div style='background-color: #5b2c6f; padding: 20px; border-radius: 10px;font-family: 'cursive'; text-align: center;'>
        <h1 style='color: #000000;'>Face Blur for Privacy</h1>
        <p style='color: #ccd1d1;'>Blur faces in uploaded images or through webcam</p>
    </div>
    
""", unsafe_allow_html=True)

# Choose mode
mode = st.radio("Select Mode", ["📤 Upload Image", "🎥 Webcam"])

# Function to blur faces in an image
# Function to blur faces in an image with adjustable blur
def blur_faces(img_array, blur_level):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Ensure kernel size is odd and >=3
    if blur_level % 2 == 0:
        blur_level += 1
    if blur_level < 3:
        blur_level = 3

    for (x, y, w, h) in faces:
        face_region = img_array[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(face_region, (blur_level, blur_level), 30)
        img_array[y:y+h, x:x+w] = blurred

    return img_array

# ---- 📤 Image Upload Mode ----
if mode == "📤 Upload Image":
    uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_array = np.array(img.convert('RGB'))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        # Add slider before processing
        blur_level = st.slider("🌀 Blur Intensity", min_value=3, max_value=99, value=51, step=2)

        # Process image with selected blur level
        blurred = blur_faces(img_bgr, blur_level)

        result = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

        st.image(result, caption="Blurred Image", use_column_width=True)

        # Download Option
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(temp_file.name, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        with open(temp_file.name, "rb") as file:
            st.download_button("📥 Download Blurred Image", file, "blurred_face.png", "image/png")

# ---- 🎥 Webcam Mode ----
else:
    st.warning("Webcam works in desktop mode only.")
    start_camera = st.button("Start Webcam")

    if start_camera:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        blur_level = st.slider("🌀 Blur Intensity", min_value=3, max_value=99, value=51, step=2)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = blur_faces(frame, blur_level)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

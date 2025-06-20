import cv2
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_in_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

def evaluate_on_dataset(face_dir, non_face_dir):
    y_true = []
    y_pred = []

    for filename in os.listdir(face_dir):
        path = os.path.join(face_dir, filename)
        if path.endswith((".jpg", ".jpeg", ".png")):
            detected = detect_faces_in_image(path)
            y_true.append(1)
            y_pred.append(1 if detected else 0)

    for filename in os.listdir(non_face_dir):
        path = os.path.join(non_face_dir, filename)
        if path.endswith((".jpg", ".jpeg", ".png")):
            detected = detect_faces_in_image(path)
            y_true.append(0)
            y_pred.append(1 if detected else 0)

    return y_true, y_pred

# Path
faces_path =r"C:\Users\User\OneDrive\Desktop\IPPR Project\dataset\faces"
nofaces_path=r"C:\Users\User\OneDrive\Desktop\IPPR Project\dataset\nofaces"



# Evaluate
y_true, y_pred = evaluate_on_dataset(faces_path, nofaces_path)

# Print Results
print("✅ Accuracy:", accuracy_score(y_true, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("📋 Classification Report:\n", classification_report(y_true, y_pred))

# Plot
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Haar Cascade Accuracy Evaluation")
plt.show()

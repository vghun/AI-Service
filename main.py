from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import mediapipe as mp
import joblib
import math

app = FastAPI()

# =====================
# LOAD MODEL
# =====================
model = joblib.load("models/face_shape_xgb.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# =====================
# MEDIAPIPE
# =====================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# =====================
# UTILS (GIỐNG TRAIN)
# =====================
def distance_3d(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0])**2 +
        (p1[1] - p2[1])**2 +
        (p1[2] - p2[2])**2
    )

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot / (norm_v1 * norm_v2 + 1e-6)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(theta)

# =====================
# FEATURE EXTRACTION (Y NGUYÊN TRAIN)
# =====================
def extract_features_from_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    def get_point(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    top_forehead = get_point(10)
    chin = get_point(152)
    left_temple = get_point(162)
    right_temple = get_point(389)
    left_cheek = get_point(234)
    right_cheek = get_point(454)
    left_jaw = get_point(58)
    right_jaw = get_point(288)
    chin_left = get_point(172)
    chin_right = get_point(397)
    left_forehead = get_point(67)
    right_forehead = get_point(297)

    face_length = distance_3d(top_forehead, chin)
    forehead_width = distance_3d(left_temple, right_temple)
    cheek_width = distance_3d(left_cheek, right_cheek)
    jaw_width = distance_3d(left_jaw, right_jaw)
    chin_width = distance_3d(chin_left, chin_right)

    length_ratio = face_length / max(forehead_width, cheek_width, jaw_width, 1e-6)
    jaw_forehead_ratio = jaw_width / forehead_width if forehead_width > 0 else 0
    cheek_jaw_ratio = cheek_width / jaw_width if jaw_width > 0 else 0
    cheek_forehead_ratio = cheek_width / forehead_width if forehead_width > 0 else 0
    chin_jaw_ratio = chin_width / jaw_width if jaw_width > 0 else 0

    jaw_angle = calculate_angle(left_jaw, chin, right_jaw)
    forehead_angle = calculate_angle(left_forehead, top_forehead, right_forehead)

    return [[
        face_length,
        forehead_width,
        cheek_width,
        jaw_width,
        chin_width,
        length_ratio,
        jaw_forehead_ratio,
        cheek_jaw_ratio,
        cheek_forehead_ratio,
        chin_jaw_ratio,
        jaw_angle,
        forehead_angle
    ]]

# =====================
# API
# =====================
@app.post("/predict")
async def predict_face_shape(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    features = extract_features_from_image(img)
    if features is None:
        return {"error": "No face detected"}

    pred_idx = model.predict(features)[0]
    face_shape = label_encoder.inverse_transform([pred_idx])[0]

    return {
        "face_shape": face_shape
    }
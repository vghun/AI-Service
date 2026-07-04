from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import cv2
import numpy as np
import mediapipe as mp
import joblib
import math
import os
import requests
import io

load_dotenv()
FACEPP_KEY    = os.getenv("FACEPLUSPLUS_API_KEY")
FACEPP_SECRET = os.getenv("FACEPLUSPLUS_API_SECRET")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# LOAD MODEL
# =====================
ensemble      = joblib.load("models/face_shape_ensemble.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
scaler        = joblib.load("models/scaler.pkl")

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
# UTILS
# =====================
def distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def get_pt(landmarks, idx):
    lm = landmarks[idx]
    return [lm.x, lm.y, lm.z]

def point_to_line_dist(p, a, b):
    ab = np.array(b[:2]) - np.array(a[:2])
    ap = np.array(p[:2]) - np.array(a[:2])
    t  = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8)
    proj = np.array(a[:2]) + t * ab
    return np.linalg.norm(np.array(p[:2]) - proj)

# =====================
# FEATURE EXTRACTION (37 features)
# =====================
def extract_features(landmarks):
    top_forehead  = get_pt(landmarks, 10)
    chin          = get_pt(landmarks, 152)
    left_temple   = get_pt(landmarks, 162)
    right_temple  = get_pt(landmarks, 389)
    left_cheek    = get_pt(landmarks, 234)
    right_cheek   = get_pt(landmarks, 454)
    left_jaw      = get_pt(landmarks, 58)
    right_jaw     = get_pt(landmarks, 288)
    chin_left     = get_pt(landmarks, 172)
    chin_right    = get_pt(landmarks, 397)
    nose_tip      = get_pt(landmarks, 4)
    left_mouth    = get_pt(landmarks, 61)
    right_mouth   = get_pt(landmarks, 291)
    left_eye_out  = get_pt(landmarks, 33)
    right_eye_out = get_pt(landmarks, 263)
    left_eyebrow  = get_pt(landmarks, 70)
    right_eyebrow = get_pt(landmarks, 300)

    face_length    = distance(top_forehead, chin)
    forehead_width = distance(left_temple, right_temple)
    cheek_width    = distance(left_cheek, right_cheek)
    jaw_width      = distance(left_jaw, right_jaw)
    chin_width     = distance(chin_left, chin_right)
    mouth_width    = distance(left_mouth, right_mouth)
    eye_width      = distance(left_eye_out, right_eye_out)
    upper_face     = distance(top_forehead, nose_tip)
    lower_face     = distance(nose_tip, chin)

    max_width            = max(forehead_width, cheek_width, jaw_width, 1e-6)
    length_ratio         = face_length    / max_width
    jaw_forehead_ratio   = jaw_width      / (forehead_width + 1e-6)
    cheek_jaw_ratio      = cheek_width    / (jaw_width + 1e-6)
    cheek_forehead_ratio = cheek_width    / (forehead_width + 1e-6)
    chin_jaw_ratio       = chin_width     / (jaw_width + 1e-6)
    upper_lower_ratio    = upper_face     / (lower_face + 1e-6)
    mouth_jaw_ratio      = mouth_width    / (jaw_width + 1e-6)
    eye_cheek_ratio      = eye_width      / (cheek_width + 1e-6)
    taper_ratio          = cheek_width    / (((forehead_width + jaw_width) / 2) + 1e-6)

    jaw_angle_center = angle(left_jaw, chin, right_jaw)
    jaw_angle_left   = angle(chin_left, left_jaw, left_temple)
    jaw_angle_right  = angle(chin_right, right_jaw, right_temple)

    widths             = [forehead_width, cheek_width, jaw_width]
    width_std          = np.std(widths) / (np.mean(widths) + 1e-6)
    length_cheek_ratio = face_length / (cheek_width + 1e-6)
    brow_mid           = [(left_eyebrow[i] + right_eyebrow[i]) / 2 for i in range(3)]
    forehead_height    = distance(top_forehead, brow_mid)
    forehead_h_ratio   = forehead_height / (face_length + 1e-6)
    jaw_symmetry       = abs(jaw_angle_left - jaw_angle_right)
    width_uniformity   = min(widths) / (max(widths) + 1e-6)
    pure_length_ratio  = face_length / (forehead_width + 1e-6)
    jaw_square_score   = 1 - abs(jaw_width - forehead_width) / (forehead_width + 1e-6)
    eye_face_ratio     = eye_width / (face_length + 1e-6)
    forehead_length_ratio = forehead_width / (face_length + 1e-6)
    chin_ratio         = lower_face / (face_length + 1e-6)

    jaw_pts      = [get_pt(landmarks, i) for i in [172, 58, 136, 150, 149, 176, 148, 152,
                                                    377, 400, 378, 379, 365, 288, 397]]
    jaw_y_values = [p[1] for p in jaw_pts]
    jaw_flatness = np.std(jaw_y_values) / (face_length + 1e-6)

    jaw_corner_angle = angle(chin_left, chin, chin_right)
    jaw_cheek_ratio  = jaw_width / (cheek_width + 1e-6)
    mid_jaw_y        = (left_jaw[1] + right_jaw[1]) / 2
    chin_deviation   = abs(chin[1] - mid_jaw_y) / (face_length + 1e-6)

    true_jaw_angle_left  = angle(left_cheek, left_jaw, chin)
    true_jaw_angle_right = angle(right_cheek, right_jaw, chin)
    true_jaw_angle_avg   = (true_jaw_angle_left + true_jaw_angle_right) / 2

    jaw_straight_left  = point_to_line_dist(left_jaw, left_cheek, chin)
    jaw_straight_right = point_to_line_dist(right_jaw, right_cheek, chin)
    jaw_straightness_n = ((jaw_straight_left + jaw_straight_right) / 2) / (face_length + 1e-6)

    forehead_jaw_sym  = 1 - abs(forehead_width - jaw_width) / (max_width + 1e-6)
    width_profile_n   = np.array(widths) / (max_width + 1e-6)
    width_profile_var = np.var(width_profile_n)
    jaw_dominance     = jaw_width / (cheek_width + 1e-6)

    return [
        face_length, forehead_width, cheek_width, jaw_width, chin_width,
        length_ratio, jaw_forehead_ratio, cheek_jaw_ratio,
        cheek_forehead_ratio, chin_jaw_ratio,
        upper_lower_ratio, mouth_jaw_ratio, eye_cheek_ratio,
        taper_ratio, mouth_width,
        jaw_angle_center, jaw_angle_left, jaw_angle_right,
        pure_length_ratio, jaw_square_score,
        eye_face_ratio, forehead_length_ratio, chin_ratio,
        width_std, length_cheek_ratio, forehead_h_ratio,
        jaw_symmetry, width_uniformity,
        jaw_flatness, jaw_corner_angle, jaw_cheek_ratio, chin_deviation,
        true_jaw_angle_avg, jaw_straightness_n,
        forehead_jaw_sym, width_profile_var, jaw_dominance,
    ]

# =====================
# SKIN TONE ANALYSIS (giữ nguyên — Face++ không có)
# =====================
def analyze_skin_tone(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results   = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return {"error": "Không phát hiện khuôn mặt"}

    landmarks = results.multi_face_landmarks[0].landmark
    h, w      = image.shape[:2]

    skin_regions = {
        "left_cheek":  [116, 117, 118, 119, 100, 101, 36,  205, 206, 207],
        "right_cheek": [345, 346, 347, 348, 329, 330, 266, 425, 426, 427],
        "forehead":    [10,  338, 297, 332, 284, 251, 389, 356, 454, 323],
    }

    skin_pixels = []
    for region, indices in skin_regions.items():
        pts = np.array([
            [int(landmarks[i].x * w), int(landmarks[i].y * h)]
            for i in indices
        ], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        pixels = image_rgb[mask == 255]
        if len(pixels) > 0:
            skin_pixels.extend(pixels.tolist())

    if len(skin_pixels) < 50:
        return {"error": "Không đủ vùng da"}

    skin_pixels = np.array(skin_pixels, dtype=np.float32)

    # Chuẩn hoá ánh sáng — loại bỏ top 5% pixel quá sáng (highlight)
    brightness_per_pixel = skin_pixels.mean(axis=1)
    threshold = np.percentile(brightness_per_pixel, 95)
    skin_pixels = skin_pixels[brightness_per_pixel < threshold]

    R = np.median(skin_pixels[:, 0])
    G = np.median(skin_pixels[:, 1])
    B = np.median(skin_pixels[:, 2])

    pixel_lab = cv2.cvtColor(np.uint8([[[int(B), int(G), int(R)]]]), cv2.COLOR_BGR2Lab)[0][0]
    L, a, b_  = float(pixel_lab[0]), float(pixel_lab[1]), float(pixel_lab[2])
    ita       = np.degrees(np.arctan((L - 50) / (b_ + 1e-6)))

    # Undertone dựa trên a* (red-green axis) và b* (yellow-blue axis)
    if a > 136 and b_ > 128:   undertone = "warm"
    elif a < 128 or b_ < 122:  undertone = "cool"
    else:                       undertone = "neutral"

    # Fitzpatrick scale (6 mức — chuẩn y tế quốc tế)
    if ita > 55:      fitzpatrick = 1  # Very fair
    elif ita > 41:    fitzpatrick = 2  # Fair
    elif ita > 28:    fitzpatrick = 3  # Medium
    elif ita > 10:    fitzpatrick = 4  # Olive
    elif ita > -30:   fitzpatrick = 5  # Brown
    else:             fitzpatrick = 6  # Dark

    fitzpatrick_label = {
        1: "Rất trắng sáng",
        2: "Trắng sáng",
        3: "Trung bình / Bánh mật",
        4: "Olive / Nâu nhạt",
        5: "Nâu đậm",
        6: "Rất đậm",
    }

    color_map = {
        "warm": {
            "recommended": ["Caramel", "Chocolate Brown", "Copper", "Warm Blonde", "Auburn", "Golden Brown"],
            "avoid":       ["Ash Blonde", "Platinum", "Cool Grey", "Blue-Black"],
            "reason":      "Tông da ấm (undertone vàng/đỏ) → chọn màu tóc có ánh vàng/đỏ/nâu ấm"
        },
        "neutral": {
            "recommended": ["Natural Brown", "Dark Blonde", "Soft Black", "Chestnut", "Mocha"],
            "avoid":       ["Extreme Platinum", "Very Cool Ash"],
            "reason":      "Tông da trung tính → hợp với hầu hết màu tóc tự nhiên"
        },
        "cool": {
            "recommended": ["Ash Brown", "Cool Blonde", "Blue-Black", "Platinum", "Cool Grey", "Burgundy"],
            "avoid":       ["Copper", "Warm Orange", "Golden Blonde", "Auburn"],
            "reason":      "Tông da lạnh (undertone hồng/xanh) → chọn màu tóc có ánh xanh/tro/bạch kim"
        },
    }
    rec = color_map[undertone]

    return {
        "undertone":          undertone,
        "fitzpatrick_scale":  fitzpatrick,
        "fitzpatrick_label":  fitzpatrick_label[fitzpatrick],
        "ita":                round(ita, 2),
        "rgb":                {"R": int(R), "G": int(G), "B": int(B)},
        "recommended_colors": rec["recommended"],
        "avoid_colors":       rec["avoid"],
        "reason":             rec["reason"],
    }

# =====================
# SKIN CONDITION — Face++ Skinanalyze API
# =====================
def analyze_skin_condition_facepp(img_bytes: bytes):
    try:
        response = requests.post(
            "https://api-us.faceplusplus.com/facepp/v1/skinanalyze",
            files={
                "api_key":    (None, FACEPP_KEY),
                "api_secret": (None, FACEPP_SECRET),
                "image_file": ("image.jpg", img_bytes, "image/jpeg"),
            },
            timeout=15
        )
        data = response.json()
    except Exception as e:
        return {"error": f"Không kết nối được Face++ API: {str(e)}"}

    # Xử lý lỗi từ Face++
    if "error_message" in data:
        err = data["error_message"]
        if err == "NO_FACE_FOUND":
            return {"error": "Không phát hiện khuôn mặt"}
        if err == "INVALID_IMAGE_FACE":
            return {"error": "Ảnh không hợp lệ hoặc có nhiều khuôn mặt"}
        if err == "IMAGE_FILE_TOO_LARGE":
            return {"error": "Ảnh quá lớn, vui lòng dùng ảnh dưới 2MB"}
        return {"error": f"Face++ lỗi: {err}"}

    # Cảnh báo góc mặt từ Face++ (dùng luôn để validate)
    warnings = data.get("warning", [])
    improper_pose = "improper_headpose" in warnings

    result = data.get("result", {})

    # Parse skin_type
    skin_type_map = {0: "oily", 1: "dry", 2: "normal", 3: "combination"}
    skin_type_raw = result.get("skin_type", {})
    if isinstance(skin_type_raw, dict):
        skin_type_idx = skin_type_raw.get("skin_type", 2)
    else:
        skin_type_idx = skin_type_raw
    skin_type = skin_type_map.get(int(skin_type_idx), "normal")

    skin_type_vi = {
        "oily":        "Da dầu",
        "dry":         "Da khô",
        "normal":      "Da thường",
        "combination": "Da hỗn hợp",
    }

    def flag(key):
        """Trả về True nếu vấn đề tồn tại (value = 1)"""
        val = result.get(key, {})
        if isinstance(val, dict):
            return str(val.get("value", "0")) == "1"
        return False

    def conf(key):
        """Trả về confidence score"""
        val = result.get(key, {})
        if isinstance(val, dict):
            return round(float(val.get("confidence", 0)), 2)
        return 0.0

    # Tổng hợp các vấn đề da
    issues = []
    if flag("acne"):          issues.append("Mụn trứng cá")
    if flag("blackhead"):     issues.append("Mụn đầu đen")
    if flag("skin_spot"):     issues.append("Đốm thâm / Tàn nhang")
    if flag("dark_circle"):   issues.append("Quầng thâm mắt")
    if flag("eye_pouch"):     issues.append("Bọng mắt")
    if flag("pores_forehead") or flag("pores_left_cheek") or flag("pores_right_cheek") or flag("pores_jaw"):
                              issues.append("Lỗ chân lông to")
    if flag("forehead_wrinkle") or flag("glabella_wrinkle"):
                              issues.append("Nếp nhăn trán")
    if flag("crows_feet") or flag("eye_finelines"):
                              issues.append("Nếp nhăn đuôi mắt")
    if flag("nasolabial_fold"):issues.append("Rãnh mũi má")

    # Gợi ý skincare theo loại da
    skincare_map = {
        "oily": {
            "recommend": ["Sữa rửa mặt tạo bọt nhẹ", "Toner BHA/Niacinamide", "Gel dưỡng ẩm oil-free", "Kem chống nắng dạng nước SPF50+"],
            "avoid":     ["Kem dưỡng quá đặc", "Dầu dừa trực tiếp"]
        },
        "dry": {
            "recommend": ["Sữa rửa mặt dạng cream", "Essence Hyaluronic Acid", "Kem dưỡng Ceramide + Squalane", "Kem chống nắng dưỡng ẩm SPF30+"],
            "avoid":     ["Sản phẩm chứa cồn cao", "Rửa mặt nước nóng"]
        },
        "normal": {
            "recommend": ["Sữa rửa mặt dịu nhẹ", "Toner cân bằng pH", "Kem dưỡng ẩm nhẹ", "Kem chống nắng SPF30+"],
            "avoid":     ["Bỏ qua bước chống nắng"]
        },
        "combination": {
            "recommend": ["Sữa rửa mặt dịu nhẹ", "Toner không cồn", "Dưỡng ẩm nhẹ vùng má", "Kem chống nắng đa năng SPF30+"],
            "avoid":     ["Sản phẩm quá mạnh làm khô toàn mặt"]
        },
    }
    sk = skincare_map.get(skin_type, skincare_map["normal"])

    return {
        "skin_type":        skin_type,
        "skin_type_vi":     skin_type_vi[skin_type],
        "issues":           issues if issues else ["Không phát hiện vấn đề da đáng kể"],
        "details": {
            "acne":             {"detected": flag("acne"),          "confidence": conf("acne")},
            "blackhead":        {"detected": flag("blackhead"),     "confidence": conf("blackhead")},
            "skin_spot":        {"detected": flag("skin_spot"),     "confidence": conf("skin_spot")},
            "dark_circle":      {"detected": flag("dark_circle"),   "confidence": conf("dark_circle")},
            "eye_pouch":        {"detected": flag("eye_pouch"),     "confidence": conf("eye_pouch")},
            "pores_forehead":   {"detected": flag("pores_forehead"),"confidence": conf("pores_forehead")},
            "pores_cheek":      {"detected": flag("pores_left_cheek") or flag("pores_right_cheek"), "confidence": max(conf("pores_left_cheek"), conf("pores_right_cheek"))},
            "pores_jaw":        {"detected": flag("pores_jaw"),     "confidence": conf("pores_jaw")},
            "forehead_wrinkle": {"detected": flag("forehead_wrinkle"), "confidence": conf("forehead_wrinkle")},
            "crows_feet":       {"detected": flag("crows_feet"),    "confidence": conf("crows_feet")},
            "nasolabial_fold":  {"detected": flag("nasolabial_fold"), "confidence": conf("nasolabial_fold")},
        },
        "skincare_recommend": sk["recommend"],
        "skincare_avoid":     sk["avoid"],
        "improper_headpose":  improper_pose,
    }

# =====================
# VALIDATE IMAGE (local — nhanh, chạy trước Face++)
# =====================
def validate_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Kiểm tra độ sáng
    brightness = np.mean(gray)
    if brightness < 60:
        return False, "Ảnh quá tối, vui lòng chụp lại nơi đủ sáng hơn"
    if brightness > 220:
        return False, "Ảnh quá sáng/chói, vui lòng tránh đèn flash trực tiếp"

    # Kiểm tra blur
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 20:
        return False, "Ảnh bị mờ, vui lòng giữ tay cố định khi chụp"

    # Kiểm tra có mặt + số người
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 8, minSize=(80, 80))
    if len(faces) == 0:
        return False, "Không phát hiện khuôn mặt, hãy chụp ảnh chính diện"
    if len(faces) > 1:
        return False, "Vui lòng chỉ có 1 người trong ảnh"

    # Kiểm tra khuôn mặt đủ lớn
    h, w = image.shape[:2]
    fx, fy, fw, fh = faces[0]
    if (fw * fh) / (w * h) < 0.08:
        return False, "Khuôn mặt quá nhỏ, vui lòng lại gần camera hơn"

    # Kiểm tra góc mặt bằng MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return False, "Không detect được khuôn mặt, hãy nhìn thẳng vào camera"

    landmarks = results.multi_face_landmarks[0].landmark
    left_cheek  = landmarks[234]
    right_cheek = landmarks[454]
    nose        = landmarks[1]

    dist_left  = abs(nose.x - left_cheek.x)
    dist_right = abs(nose.x - right_cheek.x)
    yaw_ratio  = min(dist_left, dist_right) / (max(dist_left, dist_right) + 1e-6)
    if yaw_ratio < 0.65:
        return False, "Khuôn mặt đang nghiêng ngang, hãy nhìn thẳng vào camera"

    nose_tip = landmarks[4]
    chin     = landmarks[152]
    left_eye = landmarks[33]
    right_eye= landmarks[263]
    eye_mid_y= (left_eye.y + right_eye.y) / 2
    face_h   = abs(chin.y - eye_mid_y)
    nose_pos = (nose_tip.y - eye_mid_y) / (face_h + 1e-6)
    if nose_pos < 0.3:
        return False, "Mặt đang ngẩng lên, hãy nhìn thẳng vào camera"
    if nose_pos > 0.75:
        return False, "Mặt đang cúi xuống, hãy nhìn thẳng vào camera"

    return True, "OK"

# =====================
# API ENDPOINTS
# =====================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "classes": list(label_encoder.classes_),
        "facepp_connected": bool(FACEPP_KEY and FACEPP_SECRET),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image     = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Không đọc được ảnh"}

    # Bước 1 — Validate local (nhanh)
    valid, reason = validate_image(image)
    if not valid:
        return {"error": reason}

    # Bước 2 — MediaPipe landmarks
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results   = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return {"error": "Không detect được khuôn mặt, hãy chụp ảnh chính diện rõ hơn"}

    landmarks = results.multi_face_landmarks[0].landmark

    # Bước 3 — Predict face shape
    features        = extract_features(landmarks)
    features_scaled = scaler.transform(np.array(features).reshape(1, -1))
    probs           = ensemble.predict_proba(features_scaled)[0]
    labels          = label_encoder.inverse_transform(np.arange(len(probs)))
    top3            = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:3]

    face_shape_result = {
        "predicted":  top3[0][0],
        "confidence": round(float(top3[0][1]), 4),
        "top3": [
            {"face_shape": s, "confidence": round(float(p), 4)}
            for s, p in top3
        ],
    }

    # Bước 4 — Skin tone (local MediaPipe)
    skin_tone_result = analyze_skin_tone(image)

    # Bước 5 — Skin condition (Face++ API)
    skin_condition_result = analyze_skin_condition_facepp(img_bytes)

    # Nếu Face++ báo góc mặt sai → thêm warning vào response
    # nhưng vẫn trả về kết quả (không block)
    response = {
        "face_shape":     face_shape_result,
        "skin_tone":      skin_tone_result,
        "skin_condition": skin_condition_result,
    }

    if skin_condition_result.get("improper_headpose"):
        response["warning"] = "Góc mặt chưa chuẩn, kết quả phân tích da có thể kém chính xác hơn"

    return response
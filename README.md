👤 AI Face Shape Service
Dịch vụ API nhận diện hình dáng khuôn mặt sử dụng MediaPipe (Landmarks) và XGBoost (Classification).

🚀 Công nghệ chính
Ngôn ngữ: Python 3.10.x (Bắt buộc)

AI: MediaPipe FaceMesh, XGBoost, Scikit-learn

API: FastAPI, Uvicorn

🛠 Cài đặt nhanh
1. Chuẩn bị môi trường
Yêu cầu đã cài đặt Python 3.10.

Bash

# Clone dự án
git clone <url_cua_ban>
cd ai-service

# Tạo và kích hoạt môi trường ảo
# Windows:
python -m venv venv
venv\Scripts\activate

# Mac/Linux:
python3.10 -m venv venv
source venv/bin/activate
2. Cài đặt thư viện
Bash

pip install --upgrade pip
pip install fastapi uvicorn opencv-python mediapipe==0.10.14 \
            numpy scikit-learn==1.6.1 joblib xgboost dill python-multipart
▶️ Khởi chạy
Chạy lệnh sau tại thư mục gốc:

Bash

uvicorn main:app --reload
Swagger UI (Test API): http://127.0.0.1:8000/docs

Endpoint: POST /predict (Gửi ảnh qua form-data với key là file)

📂 Cấu trúc dự án
Plaintext

ai-service/
├── main.py           # API Logic
├── models/           # Chứa face_shape_xgb.pkl
├── .gitignore        # Loại bỏ venv, __pycache__
└── README.md
⚠️ Lưu ý quan trọng
Lỗi 422: Đảm bảo đã cài python-multipart.

Venv: Không bao giờ push thư mục venv/ lên Git. Đảm bảo file .gitignore đã có dòng venv/.

Version: Nếu gặp lỗi MediaPipe, hãy kiểm tra lại phiên bản Python (phải là 3.10).
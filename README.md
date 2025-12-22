# AI Face Shape Service

**Service nhận diện loại khuôn mặt (Face Shape Recognition)**

Dự án này cung cấp một API Service sử dụng AI để phân tích và nhận diện hình dáng khuôn mặt từ ảnh đầu vào. Hệ thống tương thích với **Windows, macOS và Linux**.

---

## 🚀 Công nghệ sử dụng

* **Ngôn ngữ:** Python 3.10.x
* **Core AI:**
    * **MediaPipe FaceMesh:** Trích xuất 468 điểm landmarks trên khuôn mặt.
    * **XGBoost:** Model phân loại (Classification).
    * Scikit-learn, OpenCV, NumPy.
* **Backend API:** FastAPI & Uvicorn.
* **Utilities:** Joblib, Dill.

---

## 🛠 Yêu cầu hệ thống

* **Hệ điều hành:** Windows, macOS, hoặc Linux (Ubuntu/Debian).
* **Python Version:** **3.10.x** (Bắt buộc để tương thích với MediaPipe & Model).

---

## 📦 Hướng dẫn cài đặt

### 1. Cài đặt Python 3.10

Do thư viện MediaPipe và model đã train yêu cầu tính tương thích cao, bạn cần cài đúng **Python 3.10**.

#### 🪟 Windows
1. Tải bộ cài đặt Python 3.10 tại: [Python 3.10.11 Downloads](https://www.python.org/downloads/release/python-31011/)
2. Chạy file cài đặt, **TÍCH CHỌN** ô `Add Python to PATH` trước khi bấm Install.
3. Kiểm tra bằng PowerShell/CMD:
   ```powershell
   python --version
   # Hoặc: py -3.10 --version
🍎 macOSSử dụng Homebrew:Bashbrew install python@3.10
🐧 Linux (Ubuntu/Debian)Bashsudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
2. Thiết lập môi trường ảo (Virtual Environment)Mở Terminal (Mac/Linux) hoặc PowerShell/CMD (Windows) tại thư mục dự án ai-service.🪟 WindowsPowerShell# Tạo môi trường
python -m venv venv

# Kích hoạt môi trường
venv\Scripts\activate
🍎 macOS / 🐧 LinuxBash# Tạo môi trường
python3.10 -m venv venv

# Kích hoạt môi trường
source venv/bin/activate
⚠️ Lưu ý: Sau khi kích hoạt, đầu dòng lệnh của bạn phải hiện chữ (venv).3. Cài đặt thư viện (Dependencies)Chạy lệnh sau (giống nhau trên mọi hệ điều hành):Bash# Nâng cấp pip
pip install --upgrade pip

# Cài đặt các gói cần thiết
pip install fastapi uvicorn opencv-python mediapipe==0.10.14 numpy scikit-learn==1.6.1 joblib xgboost dill python-multipart
Tại sao cần các thư viện này?dill: Bắt buộc để load model .pkl (xử lý serialization object phức tạp).python-multipart: Để API nhận được file upload từ người dùng.scikit-learn==1.6.1: Phải khớp version với lúc train model.4. Kiểm tra MediaPipeĐể chắc chắn mọi thứ đã cài đúng, hãy chạy thử đoạn code python sau:Bashpython
Sau đó nhập:Pythonimport mediapipe as mp
print(f"Version: {mp.__version__}")
try:
    print(mp.solutions.face_mesh)
    print("✅ OK")
except Exception as e:
    print(f"❌ Error: {e}")
exit()
📂 Cấu trúc thư mụcPlaintextai-service/
├── main.py                 # Code chính
├── models/
│   └── face_shape_xgb.pkl  # File model (Bắt buộc có)
├── venv/                   # Thư mục môi trường ảo (không commit lên git)
├── README.md
└── .gitignore
▶️ Chạy ServerBashuvicorn main:app --reload
Nếu chạy trên Windows và gặp lỗi không tìm thấy lệnh, hãy thử:Bashpython -m uvicorn main:app --reload
Server sẽ chạy tại: http://127.0.0.1:8000🔌 Sử dụng APISwagger UI (Test nhanh)Truy cập: http://127.0.0.1:8000/docsAPI EndpointURL: /predictMethod: POSTBody: form-data -> key: file (upload ảnh).❓ Troubleshooting (Lỗi thường gặp)LỗiHệ điều hànhCách khắc phụcNo module named 'dill'Allpip install dillPowershell: running scripts is disabled...WindowsChạy lệnh: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser sau đó thử activate lại.'python' is not recognizedWindowsBạn chưa tích Add to PATH khi cài Python. Hãy cài lại hoặc thêm thủ công vào biến môi trường.RuntimeError: MediaPipe...AllKiểm tra lại version python phải là 3.10. (Gõ python --version).422 Unprocessable EntityAllThiếu thư viện multipart. Chạy: pip install python-multipart🛑 Tắt môi trườngBashdeactivate
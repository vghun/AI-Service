# AI Face Shape Service

Dịch vụ API nhận diện hình dáng khuôn mặt sử dụng **MediaPipe FaceMesh** để trích xuất landmarks và **XGBoost** để phân loại hình dáng khuôn mặt.

---

## 🚀 Công nghệ sử dụng

* **Ngôn ngữ:** Python **3.10.x** *(bắt buộc)*
* **AI / Machine Learning:**

  * MediaPipe FaceMesh
  * XGBoost
  * Scikit-learn
* **API Framework:**

  * FastAPI
  * Uvicorn

---

## 🛠 Cài đặt

### 1. Chuẩn bị

Yêu cầu:

* Python **3.10.x**
* Git

Clone dự án:

```bash
git clone <repository_url>
cd ai-service
```

---

### 2. Tạo môi trường ảo

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3.10 -m venv venv
source venv/bin/activate
```

---

### 3. Cài đặt thư viện

```bash
pip install --upgrade pip

pip install \
fastapi \
uvicorn \
opencv-python \
mediapipe==0.10.14 \
numpy \
scikit-learn==1.6.1 \
joblib \
xgboost \
dill \
python-multipart
```

Hoặc cài trên một dòng:

```bash
pip install --upgrade pip
pip install fastapi uvicorn opencv-python mediapipe==0.10.14 numpy scikit-learn==1.6.1 joblib xgboost dill python-multipart
```

---

## ▶️ Khởi chạy

Tại thư mục gốc của dự án:

```bash
uvicorn main:app --reload
```

Sau khi khởi động thành công:

* **Swagger UI:** http://127.0.0.1:8000/docs
* **API Endpoint:** `POST /predict`

### Gửi ảnh

* Method: **POST**
* Content-Type: **multipart/form-data**
* Key của file ảnh: **file**

---

## 📁 Cấu trúc dự án

```text
ai-service/
│
├── main.py                  # FastAPI application
├── models/
│   └── face_shape_xgb.pkl    # Mô hình XGBoost đã huấn luyện
├── .gitignore
└── README.md
```

---

## ⚠️ Lưu ý

### Lỗi 422 (Unprocessable Entity)

Nguyên nhân phổ biến:

* Chưa cài `python-multipart`
* Gửi sai tên trường của file (phải là `file`)

Cài đặt:

```bash
pip install python-multipart
```

---

### Không commit môi trường ảo

Không đưa thư mục `venv/` lên Git.

Đảm bảo `.gitignore` có:

```text
venv/
__pycache__/
*.pyc
```

---

### Phiên bản Python

Dự án yêu cầu **Python 3.10.x**.

Nếu sử dụng phiên bản khác, đặc biệt với MediaPipe, có thể phát sinh lỗi không tương thích.

Kiểm tra phiên bản:

```bash
python --version
```

hoặc

```bash
python3 --version
```

# 🤖 DEMO RAG System - Hệ thống Tìm kiếm Tài liệu Thông minh

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Gradio](https://img.shields.io/badge/gradio-5.42.0-orange.svg)
![LangChain](https://img.shields.io/badge/langchain-0.3.27-purple.svg)

### 🌟 Hệ thống RAG (Retrieval-Augmented Generation) 

*Tìm kiếm và trả lời câu hỏi từ tài liệu của bạn một cách thông minh và nhanh chóng!*

[🚀 Bắt đầu](#-cài-đặt-nhanh) • [📖 Hướng dẫn](#-hướng-dẫn-chi-tiết) • [🎯 Tính năng](#-tính-năng) • [❓ Hỗ trợ](#-hỗ-trợ)

</div>

---

## 📖 Về Dự án

**RAG System** là một hệ thống tìm kiếm tài liệu thông minh được xây dựng bằng Python, sử dụng các công nghệ AI tiên tiến như:

- 🧠 **LangChain** - Framework để xây dựng ứng dụng AI
- 🔍 **Vector Search** - Tìm kiếm ngữ nghĩa với embeddings
- 📊 **BM25** - Thuật toán tìm kiếm từ khóa truyền thống
- 🎨 **Gradio** - Giao diện web đẹp mắt và dễ sử dụng
- 🤖 **Sentence Transformers** - Mô hình nhúng văn bản

Hệ thống giúp bạn:
- ✨ Tìm kiếm thông tin trong tài liệu một cách thông minh
- 💬 Đặt câu hỏi bằng ngôn ngữ tự nhiên
- 📝 Nhận câu trả lời chi tiết với nguồn tham khảo
- 🚀 Sử dụng đơn giản qua giao diện web

---

## 🎯 Tính năng

### 🌟 Tính năng chính
- **🔍 Tìm kiếm Hybrid**: Kết hợp Vector Search và BM25 cho kết quả tối ưu
- **🧠 AI Reranking**: Sắp xếp lại kết quả theo độ liên quan
- **💬 Giao diện Chat**: Trò chuyện tự nhiên với tài liệu
- **📱 Responsive Design**: Giao diện đẹp trên mọi thiết bị
- **⚡ Xử lý nhanh**: Tối ưu hóa hiệu suất và tốc độ

### 🎨 Giao diện
- **🌈 Thiết kế hiện đại**: Giao diện thân thiện và dễ sử dụng
- **📋 Lịch sử chat**: Lưu trữ cuộc trò chuyện
- **🔄 Xóa nhanh**: Xóa lịch sử chat dễ dàng
- **📤 Copy kết quả**: Sao chép câu trả lời nhanh chóng

---

## 🚀 Cài đặt Nhanh

### 📋 Yêu cầu hệ thống
- 🐍 **Python 3.8+**
- 💾 **8GB+ RAM** (khuyến nghị)
- 🔌 **Kết nối Internet** (để tải mô hình lần đầu)
- 💿 **10GB+ dung lượng trống**

### ⚡ Cài đặt trong 3 bước

```bash
# 1️⃣ Clone repository
git clone https://github.com/J2TEAMNHQK/rag-system.git
cd rag-system

# 2️⃣ Cài đặt dependencies
pip install -r requirements.txt

# 3️⃣ Chạy hệ thống
python test_simple.py
```

🎉 **Xong!** Mở trình duyệt và truy cập: http://127.0.0.1:7860

---

## 📖 Hướng dẫn Chi tiết

### 🔧 Bước 1: Chuẩn bị môi trường

#### Option A: Sử dụng Virtual Environment (Khuyến nghị)
```bash
# Tạo virtual environment
python -m venv rag_env

# Kích hoạt (Windows)
rag_env\Scripts\activate

# Kích hoạt (Mac/Linux)
source rag_env/bin/activate
```

#### Option B: Cài đặt global
```bash
# Cài đặt trực tiếp (không khuyến nghị)
pip install -r requirements.txt
```

### 📁 Bước 2: Chuẩn bị tài liệu

1. **Tạo thư mục tài liệu**
   ```bash
   mkdir documents
   ```

2. **Thêm file TXT**
   - Sao chép các file `.txt` vào thư mục `documents/`
   - Hỗ trợ encoding UTF-8
   - Không giới hạn số lượng file

3. **Ví dụ cấu trúc thư mục**
   ```
   rag-system/
   ├── documents/
   │   ├── tailieu1.txt
   │   ├── tailieu2.txt
   │   └── ...
   ├── main.py
   └── requirements.txt
   ```

### 🎮 Bước 3: Chạy hệ thống

#### 🌟 Option A: Phiên bản đơn giản (Khuyến nghị cho lần đầu)
```bash
python test_simple.py
```
- ⚡ Khởi động nhanh (< 30 giây)
- 🔍 Tìm kiếm từ khóa cơ bản
- 💫 Giao diện đẹp mắt

#### 🧠 Option B: Phiên bản AI đầy đủ
```bash
python main_working.py
```
- 🤖 Sử dụng AI models
- 🔬 Tìm kiếm ngữ nghĩa
- ⏳ Khởi động lâu hơn (5-15 phút lần đầu)

#### 🚀 Option C: Phiên bản cao cấp
```bash
python main.py
```
- 🎯 Đầy đủ tính năng
- 🔄 Query rewriting
- 🏆 Hiệu suất tối ưu

### 🌐 Bước 4: Sử dụng giao diện web

1. **Mở trình duyệt**
   - Truy cập: http://127.0.0.1:7860
   - Hoặc: http://localhost:7860

2. **Giao diện chính**
   ```
   ┌─────────────────────────────────────┐
   │            RAG System               │
   ├─────────────────────────────────────┤
   │                                     │
   │  [Lịch sử chat hiển thị ở đây]      │
   │                                     │
   ├─────────────────────────────────────┤
   │ [Nhập câu hỏi của bạn...]    [Gửi]  │
   └─────────────────────────────────────┘
   ```

3. **Cách sử dụng**
   - 💬 Nhập câu hỏi vào ô text
   - 🔍 Nhấn "Send" hoặc Enter
   - ⏳ Đợi hệ thống xử lý
   - 📖 Xem kết quả với nguồn tham khảo

---

## 💡 Ví dụ Sử dụng

### 🔍 Các loại câu hỏi có thể đặt:

#### 📝 Tìm kiếm thông tin cụ thể
```
"Thông tin về quy trình đăng ký"
"Điều kiện tham gia chương trình"
"Danh sách yêu cầu kỹ thuật"
```

#### ❓ Câu hỏi mở
```
"Làm thế nào để..."
"Tại sao cần phải..."
"Khi nào thì..."
```

#### 🎯 Tìm kiếm từ khóa
```
"API documentation"
"pricing model"
"security requirements"
```

### 📊 Kết quả mẫu:
```
🤖 Hệ thống: Tôi tìm thấy thông tin liên quan đến câu hỏi của bạn:

📄 Từ file "huongdan.txt":
"Quy trình đăng ký bao gồm 3 bước chính: 
1. Điền form thông tin
2. Xác thực email  
3. Hoàn tất thanh toán"

📚 Nguồn tham khảo:
1. huongdan.txt: Quy trình đăng ký bao gồm 3 bước...
2. quydinh.txt: Điều kiện tham gia chương trình...
```

---

## 🛠️ Troubleshooting

### ❌ Lỗi thường gặp và cách khắc phục

#### 1. 🔌 "Port 7860 is already in use"
```bash
# Kiểm tra port đang sử dụng
netstat -ano | findstr 7860

# Thay đổi port trong code
# Sửa: server_port=7860 → server_port=8080
```

#### 2. 📦 "ModuleNotFoundError"
```bash
# Kiểm tra virtual environment
pip list

# Cài đặt lại dependencies
pip install -r requirements.txt --force-reinstall
```

#### 3. 🐌 "Hệ thống khởi động chậm"
```bash
# Sử dụng phiên bản đơn giản
python test_simple.py

# Hoặc tăng timeout
# Đợi 5-15 phút cho lần đầu tải models
```

#### 4. 📁 "No documents found"
```bash
# Kiểm tra thư mục documents
ls documents/

# Thêm file .txt vào thư mục
copy *.txt documents/
```

#### 5. 🌐 "Cannot access web interface"
```bash
# Kiểm tra firewall
# Cho phép Python qua Windows Firewall

# Thử các URL khác
http://localhost:7860
http://0.0.0.0:7860
```

### 🔧 Debug mode
```bash
# Chạy với debug để xem log chi tiết
python main.py --debug

# Hoặc kiểm tra console output
```

---

## 📁 Cấu trúc Dự án

```
rag-system/
├── 📄 README.md                 # Tài liệu hướng dẫn
├── 📄 requirements.txt          # Dependencies
├── 📄 setup.py                  # Package setup
├── 📄 MANUAL_SETUP.md          # Hướng dẫn thủ công
├── 📁 documents/               # Thư mục tài liệu
│   ├── 📄 1.txt
│   └── 📄 2.txt
├── 🐍 main.py                  # Phiên bản đầy đủ
├── 🐍 main_simplified.py       # Phiên bản đơn giản
├── 🐍 main_working.py          # Phiên bản AI
├── 🐍 test_simple.py           # Phiên bản test
├── 📁 rag_env/                 # Virtual environment
└── 📁 chroma_db/              # Vector database (tự tạo)
```

### 📋 Mô tả files chính:

| File               | Mô tả                               | Khuyến nghị         |
|--------------------|--------------------------------------|---------------------|
| `test_simple.py`   | 🌟 Phiên bản cơ bản, khởi động nhanh | Người mới bắt đầu   |
| `main_working.py`  | 🧠 Phiên bản AI với embeddings       | Sử dụng nâng cao    |
| `main.py`          | 🚀 Phiên bản đầy đủ tính năng        | Production          |
| `main_simplified.py` | ⚡ Phiên bản tối ưu                 | Development         |


---

## ⚙️ Cấu hình Nâng cao

### 🎛️ Tùy chỉnh hệ thống

#### 📝 Chỉnh sửa cấu hình trong code:
```python
class SystemConfig:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Mô hình embedding
    CHUNK_SIZE = 500                      # Kích thước chunk
    CHUNK_OVERLAP = 100                   # Độ chồng lấp
    TOP_K_RETRIEVAL = 3                   # Số kết quả trả về
    DEVICE = "cpu"                        # CPU hoặc CUDA
    DOCUMENTS_DIR = "./documents"         # Thư mục tài liệu
```

#### 🔧 Các tùy chọn khác:
- **Thay đổi port**: Sửa `server_port=7860`
- **Thay đổi theme**: Sửa `theme=gr.themes.Soft()`
- **Tăng timeout**: Sửa `timeout=300000`

### 🚀 Tối ưu hiệu suất

#### 💻 Cho máy tính yếu:
```python
# Sử dụng model nhẹ hơn
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
TOP_K_RETRIEVAL = 2
```

#### 🖥️ Cho máy tính mạnh:
```python
# Sử dụng model tốt hơn
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
CHUNK_SIZE = 1000
TOP_K_RETRIEVAL = 5
```

---

## 🤝 Đóng góp

### 💡 Cách đóng góp:
1. 🍴 Fork repository
2. 🌿 Tạo branch mới: `git checkout -b feature/amazing-feature`
3. 💾 Commit changes: `git commit -m 'Add amazing feature'`
4. 📤 Push branch: `git push origin feature/amazing-feature`
5. 🔄 Tạo Pull Request

### 🐛 Báo lỗi:
- 🎯 Mở [GitHub Issues](https://github.com/J2TEAMNHQK/rag-system/issues)
- 📝 Mô tả chi tiết lỗi
- 🖼️ Attach screenshots nếu có

### 💬 Thảo luận:
- 💭 [GitHub Discussions](https://github.com/J2TEAMNHQK/rag-system/discussions)
- 📧 Email: j2teamnhqk@gmail.com

---

## 📜 License

Dự án này được phân phối dưới license **MIT**. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

```
MIT License - Bạn có thể:
✅ Sử dụng thương mại
✅ Sửa đổi code  
✅ Phân phối
✅ Sử dụng riêng tư
```

---

## 👨‍💻 Tác giả

<div align="center">

### 🌟 **J2TEAM NHQK**

*Đội ngũ phát triển đam mê công nghệ AI và Machine Learning*

[![GitHub](https://img.shields.io/badge/GitHub-J2TEAMNHQK-181717?style=for-the-badge&logo=github)](https://github.com/J2TEAMNHQK)
[![Website](https://img.shields.io/badge/Website-j2team.dev-blue?style=for-the-badge&logo=google-chrome)].

**Cảm ơn bạn đã sử dụng RAG System! ❤️**

*Nếu dự án hữu ích, đừng quên để lại ⭐ trên GitHub nhé!*

</div>

---

## 🔖 Changelog

### 📅 Version 1.0.0 (2025-08-17)
- 🎉 Ra mắt phiên bản đầu tiên
- ✨ Tính năng tìm kiếm cơ bản
- 🎨 Giao diện Gradio đẹp mắt
- 📚 Hỗ trợ multiple documents

### 🚀 Tính năng sắp tới:
- 🔊 Hỗ trợ giọng nói
- 📊 Dashboard analytics
- 🌐 Multilingual support
- 📱 Mobile app
- ☁️ Cloud deployment

---

<div align="center">

### 🎯 **Hãy để RAG System giúp bạn tìm kiếm thông minh hơn!**

*Made with ❤️ by J2TEAM NHQK*

</div>

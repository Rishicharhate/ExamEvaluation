Here’s a clean, professional **README.md** for your **AI Handwritten Exam Evaluation System** 👇 (you can directly paste this into your GitHub repo)

---

# 🧠 AI Handwritten Exam Evaluation System

An AI-powered system that automatically evaluates handwritten exam sheets (images or PDFs) using OCR and NLP techniques. This project extracts handwritten text, compares it with model answers, and generates marks with feedback.

---

## 🚀 Features

* 📝 **Handwritten Text Recognition (OCR)**
* 📄 **Supports Images & PDFs**
* 🤖 **AI-Based Answer Evaluation**
* 📊 **Marks + Feedback Generation**
* 🌐 **Web Interface (Flask आधारित)**
* 📥 **Bulk Upload Support**
* 📑 **Auto Report Generation (JSON/PDF)**

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Flask (Python)
* **OCR Model:** TrOCR (Transformers)
* **NLP Evaluation:** LLM / Semantic Similarity
* **Libraries:**

  * `transformers`
  * `torch`
  * `opencv-python`
  * `pytesseract`
  * `pdf2image`
  * `flask`

---

## 🏗️ Project Structure

```
AI-Exam-Evaluator/
│
├── app.py                  # Main Flask server
├── pdf_processor.py        # PDF → Image conversion
├── ocr_engine.py           # Handwritten text extraction
├── evaluator.py            # Answer evaluation logic
├── static/                 # CSS, JS
├── templates/              # HTML files
├── uploads/                # Uploaded files
├── outputs/                # Results
└── requirements.txt
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/AI-Exam-Evaluator.git
cd AI-Exam-Evaluator
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

* Windows:

```bash
venv\Scripts\activate
```

* Linux/Mac:

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Install Tesseract OCR

Download and install from:
👉 [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)

Then set path in Python:

```python
pytesseract.pytesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## ▶️ Run the Project

```bash
python app.py
```

Open browser:

```
http://localhost:5000
```

---

## 🔄 Workflow

1. User uploads handwritten answer sheet (Image/PDF)
2. PDF is converted into images (if needed)
3. OCR extracts handwritten text
4. Extracted text is cleaned and structured
5. AI compares with model answers
6. Marks + feedback are generated
7. Results are displayed/downloaded

---

## 🧠 AI Evaluation Logic

* Uses:

  * Keyword Matching
  * Semantic Similarity (Sentence Transformers)
  * LLM-based scoring (optional)

Evaluation Criteria:

* Relevance
* Completeness
* Keywords coverage
* Answer structure

---

## 📊 Output Example

```json
{
  "question_1": {
    "marks_awarded": 4,
    "max_marks": 5,
    "feedback": "Answer is correct but missing key points."
  }
}
```

---

## 📌 Future Improvements

* ✨ Better handwriting accuracy (fine-tuned models)
* 🗣️ Voice-based evaluation
* 📱 Mobile app integration
* 📈 Teacher analytics dashboard
* 🌍 Multi-language support

---

## 🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Rishi Charhate**

---

If you want, I can also:

* ✅ Generate **requirements.txt**
* ✅ Create **GitHub repo structure**
* ✅ Add **LLM evaluation (like Ollama / OpenAI)**
* ✅ Convert this into a **PPT for your project presentation**

Just tell me 👍

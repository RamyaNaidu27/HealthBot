# 🩺 MediAssist Pro - AI Medical Assistant

Welcome to **MediAssist Pro**, an AI-powered medical assistant designed to help users understand medical documents (PDFs and prescription images) and receive safe, general medical guidance. Built using cutting-edge AI technologies, this tool leverages **Retrieval-Augmented Generation (RAG)** for delivering context-aware responses grounded in uploaded content.

---

## 📚 Table of Contents

- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [RAG Workflow](#-rag-workflow)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🚀 Features

- 📄 Upload and process PDF medical reports and prescription images.
- 🧠 Extract text from images using **Mistral OCR**.
- 💬 Real-time chat interface with streaming responses.
- 🔍 Context-aware answers using RAG from uploaded documents.
- 💡 Generalized health advice for out-of-context questions.
- 🎙️ Voice input support for hands-free interaction.
- 📱 Responsive design with drag-and-drop file uploads.
- 📊 Status tracking for uploaded files and processed chunks.

---

## 🛠 Technologies Used

| Area             | Tech Stack                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Frontend**     | HTML5, CSS3 (with custom animations), Vanilla JavaScript                   |
| **Backend**      | FastAPI, LangChain, Google Generative AI (Gemini 2.5-flash), Mistral OCR   |
| **Image Handling** | PIL (Python Imaging Library)                                             |
| **Vector Search**| FAISS                                                                      |
| **Validation**   | Pydantic                                                                   |
| **Env. Management** | dotenv                                                                 |
| **Retry Logic**  | tenacity                                                                   |
| **CORS Handling**| FastAPI middleware                                                         |

---

## 🧠 RAG Workflow

**MediAssist Pro** utilizes a **Retrieval-Augmented Generation (RAG)** framework to provide accurate, personalized responses.

1. ### 📥 Document Ingestion  
   - PDFs: Parsed using `PyPDFLoader`  
   - Images: Processed with `Mistral OCR`  

2. ### ✂️ Chunking & Embedding  
   - Text is chunked using `RecursiveCharacterTextSplitter`  
   - Embeddings are generated via `text-embedding-004` from Google GenAI  

3. ### 🧠 Vector Store  
   - All embeddings are stored in `FAISS` for efficient similarity search  

4. ### 🔎 Retrieval  
   - Top 5 most relevant document chunks are retrieved using `similarity_search(k=5)`

5. ### 🗣️ Response Generation  
   - Gemini 2.5-flash generates natural language answers using the context  
   - Falls back to generalized advice if context is missing

---

## ⚙️ Installation

### ✅ Prerequisites

- Python 3.9+
- Node.js *(Optional for advanced frontend setup)*
- API Keys:
  - `GOOGLE_API_KEY`
  - `MISTRAL_API_KEY`

---

### 🧪 Setup Steps

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/mediassist-pro.git
cd mediassist-pro
```
2. **Create .env File**
```bash
GOOGLE_API_KEY=your_google_api_key
MISTRAL_API_KEY=your_mistral_api_key
```
3. **Install Dependencies**
```bash
pip install -r requirements.txt
```
4. **Run the Backend Server**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
5. **🖥️ Usage**
- 📄 Upload Documents
   - Use drag-and-drop or "Upload PDF/Image" buttons
   - Supported file types: .pdf, .jpg, .png
- 🧑‍⚕️ Ask Health Questions
  - Enter your query in the chat box
  - Click 🎙️ to use voice input (supported browsers only)

- 💬 View Real-Time Responses
  - AI responses stream in real-time
  - If context from a document exists, it will be used
  - Otherwise, the bot will offer general safe advice

- 📁 Track File Status
  - Sidebar shows recent files with number of processed chunks
---
**📁 Project Structure**
```bash
mediassist-pro/
│
├── index.html         # Frontend HTML file
├── main.py            # FastAPI backend server
├── requirements.txt   # Python dependencies
├── .env               # Environment variables (not tracked)
└── README.md          # You're reading this

```
---
**🤝 Contributing**
- We welcome contributions from the community!
    - Fork this repo
    - Create your feature branch
    - git checkout -b feature-branch
    - Commit your changes
    - git commit -m "Add new feature"
    - Push to your fork
    - git push origin feature-branch
    - Open a Pull Request 🎉

---
**📄 License**
- This project is licensed under the MIT License.
- See LICENSE for more information.

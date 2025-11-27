# 🧠 MindBridge AI: Mental Health Support System

**MindBridge** is an AI-powered mental health assistant that prioritizes safety and empathy. Unlike standard chatbots, it uses **Retrieval Augmented Generation (RAG)** to provide advice grounded in verified counselor transcripts and includes a dedicated **Safety Layer** to detect crisis situations.

## 🚀 Features
- **Crisis Detection:** A BERT-based classifier intercepts high-risk messages (suicide/self-harm) and redirects to helplines.
- **Empathetic RAG:** Retrieves responses from a database of 3,000+ real therapy conversations.
- **Privacy-First:** Runs locally using `LangChain` and `Streamlit`.

## 🛠️ Tech Stack
- **Language:** Python 3.11
- **Frameworks:** Streamlit, LangChain, Transformers
- **Models:** `all-MiniLM-L6-v2` (Embeddings), `LogisticRegression` (Safety)

## 📦 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/Arceus18/MindBridge-AI-Capstone.git
   
2. Install dependencies:
pip install -r requirements.txt

3.Run the app:
python -m streamlit run src/app.py

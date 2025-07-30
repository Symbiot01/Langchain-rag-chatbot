# 🧠 RAG Chatbot (Groq + LangChain)

This is a Retrieval-Augmented Generation (RAG) chatbot built using [LangChain](https://www.langchain.com/), [Groq API](https://console.groq.com/), and [Streamlit](https://streamlit.io/). The chatbot can answer questions based solely on the contents of a `.txt` document using semantic search and an LLM backend (Groq).

## 🔧 Features

-  Loads and splits a text document into semantic chunks.
-  Stores embeddings using ChromaDB with `BAAI/bge-base-en-v1.5`.
-  Uses Groq's LLMs (e.g., `llama3-8b-8192`) via `langchain-groq`.
-  Implements RetrievalQA with context-aware prompts.
-  Built-in Streamlit interface for interactive querying.
-  Source document traceability for every answer.

---

## 🗂️ Project Structure

```

📁 langchain-chatbot/
│
├── config.py              # API keys and config variables
├── rag\_pipeline.py        # Document loader, vector store setup, and RAG chain
├── app.py                 # Streamlit app for user interaction
├── data.txt               # Your text knowledge base
├── chroma/                # Chroma vector store persistence directory
└── README.md              # This file

````

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/SudoAnxu/Langchain-RAG-Chatbot.git
cd Langchain-RAG-Chatbot
````

### 2. Create a virtual environment

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

You can generate `requirements.txt` using:

```bash
pip freeze > requirements.txt
```

---

## 🔑 Configuration

Create a `config.py` file in the root directory with the following:

```python
# config.py
GROQ_API_KEY = "your-groq-api-key"
LLM_MODEL = "llama3-8b-8192"  # or any supported Groq model
CHROMA_DIR = "chroma"
DATA_FILE = "data.txt"
```

---

## 🚀 Running the App

Make sure `data.txt` contains your source knowledge. Then launch the app:

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## 🧪 Example Query

```text
💬 Ask a question based on the document:
> What is the purpose of this document?

✅ Answer:
> [The answer based strictly on the context]

📄 Source Documents:
> [Shows chunks that answer was based on]
```

---

## 📌 Notes

* The chatbot will **only answer questions using the provided context** from `data.txt`.
* If the answer is not found, it will respond:

  > *"I'm sorry, the information is not available in the provided context."*

---

## 📃 License

[MIT License](./LICENSE)

---

## ✨ Acknowledgments

* [LangChain](https://docs.langchain.com/)
* [Groq](https://groq.com/)
* [FastEmbed](https://github.com/lazymatrix/fastembed)
* [Streamlit](https://streamlit.io/)

---

## 🙋‍♂️ Author

**Priyangshu Karmakar**
*AI Developer & ML Enthusiast*
GitHub: [@SudoAnxu](https://github.com/SudoAnxu)

```


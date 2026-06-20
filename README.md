# SupportSense AI

SupportSense AI is a persona-aware customer support assistant built with a Streamlit frontend and a LangChain-powered backend. It uses a FAISS semantic index over documentation content to deliver grounded responses tailored to user personas such as Technical Expert, Frustrated User, and Business Executive.

## Key Features

- Streamlit UI with live system status, persona override, and escalation controls
- Document retrieval using FAISS and HuggingFace embeddings
- Persona-adaptive response generation with Google Gemini integration
- Configurable confidence threshold and session memory management
- Rebuildable knowledge index from `backend/docs/`

## Repository Structure

- `backend/`
  - `src/` - core backend modules for persona detection, RAG pipeline, response generation, escalation handling, and memory management
  - `docs/` - input documents used to build the FAISS knowledge index
  - `faiss_index/` - local FAISS index storage (ignored by Git)
  - `requirements.txt` - backend Python dependencies
- `frontend/`
  - `app.py` - Streamlit application entrypoint
  - `components.py` - UI rendering helpers
  - `style.css` - application styling
  - `requirements.txt` - frontend Python dependencies
- `.gitignore` - repository ignore rules for Python, IDE files, virtual environments, FAISS index files, and logs

## Setup

1. Create a Python virtual environment.

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install backend dependencies.

   ```bash
   pip install -r backend/requirements.txt
   ```

3. Install frontend dependencies.

   ```bash
   pip install -r frontend/requirements.txt
   ```

4. Create a `backend/.env` file with at least the Gemini API key.

   ```text
   GEMINI_API_KEY=your_google_gemini_api_key
   LLM_MODEL=gemini-2.5-flash
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   DOCS_DIR=docs
   DB_PATH=faiss_index
   CONFIDENCE_THRESHOLD=0.40
   LOG_LEVEL=INFO
   ```

## Running the App

Start the Streamlit frontend from the repository root:

```bash
streamlit run frontend/app.py
```

Then open the URL shown in the terminal.

## Using the App

- Enter your Google Gemini API key in the sidebar to enable live LLM generation.
- Use the persona override dropdown to simulate different response styles.
- Use the "Rebuild Knowledge Vector Index" button to regenerate the FAISS index from documents in `backend/docs/`.
- Reset the live session to clear in-memory chat history and escalation state.

## Notes

- The project uses `backend/faiss_index/` to store FAISS artifacts. This directory is ignored by Git.
- Add support documents to `backend/docs/` in `.txt`, `.md`, or `.pdf` format.
- If the knowledge index is missing or invalid, the app will rebuild it automatically.

## License

This repository does not include a license file by default. Add one if you want to define reuse terms.

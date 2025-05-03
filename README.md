# Document RAG Assistant

This project is a Retrieval-Augmented Generation (RAG) system that allows you to:
1. Upload PDF documents
2. Process and index their content
3. Chat with your documents
4. Perform web searches

## Architecture

The project consists of two main components:
- **FastAPI Backend**: Handles document processing, vector storage, and RAG functionality
- **Streamlit Frontend**: Provides a user-friendly interface for uploading documents and chatting

## Setup Instructions

### 1. System Dependencies

The application requires the following system dependencies:

#### macOS (using Homebrew):
```bash
# Install poppler (required for PDF processing)
brew install poppler

# Install tesseract (required for OCR)
brew install tesseract
```

#### Ubuntu/Debian:
```bash
# Install poppler and tesseract
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr
```

#### Windows:
- Download and install poppler from: https://github.com/oschwartz10612/poppler-windows/releases/
- Download and install tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Add both to your PATH environment variable

### 2. Create a Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 4. Configure OpenAI API Key

Create a `.env` file in the root directory with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Application

### 1. Start the FastAPI Backend

```bash
# Start the backend server
python main.py
```

The API server will run at http://localhost:8000

### 2. Start the Streamlit Frontend

In a new terminal window (with the virtual environment activated):

```bash
# Activate virtual environment
source venv/bin/activate

# Start the Streamlit app
streamlit run streamlit_app.py
```

The Streamlit app will open in your browser at http://localhost:8501

## Usage

1. **Upload Documents**: Use the file uploader to select one or more PDF files
2. **Process Documents**: Click the "Process Documents" button to index the content
3. **Chat with Documents**: Ask questions about your documents in the chat interface
4. **Web Search**: Switch to "Web Search" mode to search the web for information

## API Endpoints

The FastAPI backend provides the following endpoints:

- `POST /process_document`: Process and index a document
- `POST /chat`: Ask questions about processed documents
- `POST /web_search`: Search the web for information

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (see `requirements.txt`)
- System dependencies: `poppler` and `tesseract`

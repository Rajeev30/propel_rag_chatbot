import streamlit as st
import requests
import os
import time
import tempfile
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

# API endpoints
API_BASE_URL = "http://localhost:8000"
PROCESS_DOCUMENT_URL = f"{API_BASE_URL}/process_document"
CHAT_URL = f"{API_BASE_URL}/chat"
WEB_SEARCH_URL = f"{API_BASE_URL}/web_search"
HEALTH_CHECK_URL = f"{API_BASE_URL}/"

# Initialize session state variables if they don't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "user_input" not in st.session_state:
    st.session_state.user_input = "Summarize this document"
if "api_running" not in st.session_state:
    st.session_state.api_running = False

def check_api_status():
    """Check if the FastAPI backend is running"""
    try:
        logger.info(f"Checking API status at {HEALTH_CHECK_URL}")
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        if response.status_code == 200:
            logger.info("API connection successful")
            st.session_state.api_running = True
            return True
        else:
            logger.warning(f"API responded with status code: {response.status_code}")
            st.session_state.api_running = False
            return False
    except requests.RequestException as e:
        logger.error(f"API connection error: {str(e)}")
        st.session_state.api_running = False
        return False

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path"""
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    logger.info(f"Saved uploaded file to {file_path}")
    return file_path

def process_document(file_path):
    """Send file to the API for processing"""
    if not check_api_status():
        return False, "Error: FastAPI backend is not running. Please start it with 'python main.py'."
    
    st.session_state.processing = True
    
    try:
        start_time = time.time()
        
        payload = {
            "document_path": file_path,
            "vector_db_dir": "faiss_index",
            "content_dir": "content"
        }
        
        logger.info(f"Sending document processing request: {payload}")
        response = requests.post(PROCESS_DOCUMENT_URL, json=payload, timeout=300)  # 5 minute timeout
        
        if response.status_code == 200:
            processing_time = time.time() - start_time
            st.session_state.documents_processed = True
            logger.info(f"Document processed successfully in {processing_time:.2f} seconds")
            return True, f"Document processed successfully in {processing_time:.2f} seconds"
        else:
            logger.error(f"Error processing document: {response.status_code} - {response.text}")
            return False, f"Error processing document: {response.text}"
    except requests.Timeout:
        logger.error("Processing request timed out")
        return False, f"Error: Processing timed out. The document may be too large or complex."
    except requests.ConnectionError:
        logger.error("Connection error during document processing")
        return False, f"Error: Connection to API failed. Please ensure the FastAPI backend is running."
    except Exception as e:
        logger.error(f"Unexpected error during document processing: {str(e)}")
        return False, f"Error: {str(e)}"
    finally:
        st.session_state.processing = False

def send_message(message, message_type="chat"):
    """Send a message to the appropriate API endpoint"""
    if not check_api_status():
        return False, "Error: FastAPI backend is not running. Please start it with 'python main.py'.", "Connection Error"
    
    st.session_state.processing = True
    
    try:
        start_time = time.time()
        
        if message_type == "chat":
            url = CHAT_URL
        elif message_type == "web_search":
            url = WEB_SEARCH_URL
        else:
            raise ValueError("Invalid message type")
        
        payload = {"question": message}
        logger.info(f"Sending {message_type} request: {message[:50]}...")
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            processing_time = time.time() - start_time
            data = response.json()
            
            if message_type == "chat":
                answer = data.get("answer", "No answer received")
                log_message = f"Response time: {processing_time:.2f}s"
                logger.info(f"Chat response received in {processing_time:.2f}s")
                return True, answer, log_message
            else:  # web_search
                results = data.get("results", [])
                answer = "Web search results:\n\n"
                for idx, result in enumerate(results[:5], 1):  # Show top 5 results
                    answer += f"{idx}. {result.get('title', 'No title')}\n"
                    answer += f"   {result.get('href', 'No link')}\n"
                    answer += f"   {result.get('body', 'No description')}\n\n"
                log_message = f"Found {len(results)} results in {processing_time:.2f}s"
                logger.info(f"Web search response received with {len(results)} results in {processing_time:.2f}s")
                return True, answer, log_message
        else:
            error_data = response.json() if response.text else {"detail": "Unknown error"}
            error_message = error_data.get("detail", "Unknown error")
            logger.error(f"API error: {response.status_code} - {error_message}")
            
            if "Document not processed" in error_message:
                return False, "You need to upload and process a document first. Please upload a PDF file and click 'Process Documents'.", "Document Not Processed"
            else:
                return False, f"Error: {error_message}", "API Error"
    except requests.Timeout:
        logger.error("Request timed out")
        return False, "Error: Request timed out. The server might be overloaded.", "Timeout"
    except requests.ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        return False, "Error: Connection to API failed. Please ensure the FastAPI backend is running.", "Connection Error"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False, f"Error: {str(e)}", "Error"
    finally:
        st.session_state.processing = False

def handle_message(message, message_type="chat"):
    """Handle a user message, update chat history, and get response"""
    # Add user message to chat history
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append({
        "sender": "user",
        "message": message,
        "timestamp": timestamp
    })
    
    # Send message to API and get response
    success, response, log = send_message(message, message_type)
    
    # Add response to chat history
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append({
        "sender": "assistant",
        "message": response,
        "timestamp": timestamp,
        "log": log if success else None,
        "error": not success
    })
    
    # Reset the input field by updating the session state variable
    st.session_state.user_input = ""

def clear_chat():
    """Clear the chat history"""
    logger.info("Clearing chat history")
    st.session_state.chat_history = []

def clear_files():
    """Clear uploaded files"""
    logger.info("Clearing uploaded files")
    st.session_state.uploaded_files = []
    st.session_state.documents_processed = False

def on_send_message():
    """Function to handle sending messages"""
    if st.session_state.user_input:
        message_type_value = "web_search" if st.session_state.message_type == "Web Search" else "chat"
        
        if message_type_value == "chat" and not st.session_state.documents_processed:
            logger.warning("Chat attempted before document processing")
            st.warning("Please process documents first before chatting.")
            st.session_state.chat_history.append({
                "sender": "assistant",
                "message": "You need to upload and process a document first. Please upload a PDF file and click 'Process Documents'.",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "error": True
            })
        else:
            message = st.session_state.user_input
            logger.info(f"Sending message: {message[:50]}...")
            with st.spinner("Getting response..."):
                handle_message(message, message_type_value)

# Check API status at startup
logger.info("Starting Streamlit app")
api_running = check_api_status()

# Main app layout
st.title("📚 Document RAG Assistant")

# Create two columns: file upload and chat interface
col1, col2 = st.columns([1, 2])

# File Upload Section (Left Column)
with col1:
    st.header("Upload Documents")
    
    if not api_running:
        st.error("⚠️ FastAPI backend is not running! Please start it in another terminal with:\n```\nsource venv/bin/activate && python main.py\n```")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [f["name"] for f in st.session_state.uploaded_files]:
                file_path = save_uploaded_file(uploaded_file)
                st.session_state.uploaded_files.append({
                    "name": uploaded_file.name,
                    "size": f"{uploaded_file.size / 1024:.1f} KB",
                    "type": uploaded_file.type,
                    "path": file_path
                })
    
    st.subheader("Uploaded Files")
    if st.session_state.uploaded_files:
        for idx, file in enumerate(st.session_state.uploaded_files):
            st.write(f"{idx + 1}. {file['name']} ({file['size']})")
        
        if not st.session_state.documents_processed:
            process_button = st.button(
                "Process Documents",
                disabled=st.session_state.processing or not api_running,
                use_container_width=True
            )
            
            if process_button:
                with st.spinner("Processing documents... This may take several minutes for large files."):
                    for file in st.session_state.uploaded_files:
                        success, message = process_document(file["path"])
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
        else:
            st.success("Documents processed successfully!")
    else:
        st.info("No files uploaded yet.")
    
    st.button("Clear Files", on_click=clear_files, use_container_width=True)

# Chat Interface (Right Column)
with col2:
    st.header("Chat with your Documents")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["sender"] == "user":
                st.markdown(f"**You ({message['timestamp']}):** {message['message']}")
            else:
                st.markdown(f"**Assistant ({message['timestamp']}):**")
                
                if message.get("error", False):
                    st.error(message["message"])
                else:
                    st.markdown(message["message"])
                    
                if message.get("log"):
                    st.caption(message["log"])
    
    # Add a separator
    st.markdown("---")
    
    # Message input and buttons
    st.session_state.message_type = st.radio(
        "Message Type:",
        options=["Chat with Documents", "Web Search"],
        horizontal=True,
        index=0
    )
    
    # Message input area with Send button
    col_input, col_button = st.columns([5, 1])
    
    with col_input:
        st.text_input(
            "Type your message:",
            key="user_input",
            disabled=st.session_state.processing or (not api_running) or 
                    (st.session_state.message_type == "Chat with Documents" and not st.session_state.documents_processed)
        )
    
    with col_button:
        send_button = st.button(
            "Send",
            on_click=on_send_message,
            disabled=st.session_state.processing or (not api_running) or 
                    (st.session_state.message_type == "Chat with Documents" and not st.session_state.documents_processed),
            use_container_width=True
        )
    
    # Clear chat button
    st.button("Clear Chat", on_click=clear_chat, use_container_width=True)
    
    # Show a warning if chat is enabled but no documents processed
    if st.session_state.message_type == "Chat with Documents" and not st.session_state.documents_processed:
        st.warning("⚠️ Please upload and process a document first before chatting.")
        
    # Show API status
    if not api_running:
        st.error("⚠️ FastAPI backend is not running! Please start it in another terminal.")
    elif not st.session_state.documents_processed and st.session_state.message_type == "Chat with Documents":
        st.info("ℹ️ Upload and process a document to start chatting, or switch to Web Search.")
    elif st.session_state.documents_processed:
        st.success("✅ Ready to chat with your documents!")
    else:
        st.success("✅ Ready for web search!") 
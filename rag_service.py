import os
from dotenv import load_dotenv

# Load environment variables from .env (API keys, etc.)
load_dotenv()

# Set environment variable to fix OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import uuid
import shutil
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verify that the OpenAI API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("⚠️ OPENAI_API_KEY not set. Define it in your .env file.")

# Global state for vectorstore and retriever
vectorstore = None
retriever = None


def process_document(
    document_path: str,
    vector_db_dir: str = "faiss_index",
    content_dir: str = "content"
):
    """
    - Extracts and partitions PDF content (text and tables).
    - Summarizes each chunk via ChatOpenAI.
    - Builds and persists a FAISS index locally.
    - Initializes a retriever for later queries.
    """
    logger.info(f"Processing document: {document_path}")
    # Clean and recreate content directory
    if os.path.exists(content_dir):
        shutil.rmtree(content_dir)
    os.makedirs(content_dir, exist_ok=True, mode=0o777)

    try:
        # Partition PDF into elements
        from unstructured.partition.pdf import partition_pdf
        logger.info(f"Partitioning PDF...")
        elements = partition_pdf(
            filename=document_path,
            infer_table_structure=True,
            strategy="auto",
            extract_image_block_types=["Image"],
            image_output_dir_path=content_dir,
            chunking_strategy="by_title",
            max_characters=4000,
            combine_text_under_n_chars=2000,
        )
        logger.info(f"PDF partitioning complete. Found {len(elements)} elements.")

        # Summarize and collect docs
        docs: List[Document] = []
        for el in elements:
            # Determine content and prompt based on element type
            if hasattr(el, "text") and el.text:
                content = el.text
                prompt_txt = "Summarize this technical text focusing on key specifications:"
            elif hasattr(el.metadata, "text_as_html") and el.metadata.text_as_html:
                content = el.metadata.text_as_html
                prompt_txt = "Explain this table's key data points and relationships:"
            else:
                continue

            # Summarize via ChatOpenAI
            model = ChatOpenAI(model="gpt-3.5-turbo-0125")
            prompt = ChatPromptTemplate.from_template(f"{prompt_txt}\n\n{{content}}")
            chain = prompt | model | StrOutputParser()
            summary = chain.invoke({"content": content})
            text = summary if isinstance(summary, str) else summary.content
            if text:
                docs.append(
                    Document(
                        page_content=text.strip(),
                        metadata={"id": str(uuid.uuid4()), "type": el.__class__.__name__}
                    )
                )

        if not docs:
            raise ValueError("No content extracted to index.")

        logger.info(f"Document processing complete. Created {len(docs)} document chunks.")

        # Clean and recreate vector DB directory
        if os.path.exists(vector_db_dir):
            shutil.rmtree(vector_db_dir)
        os.makedirs(vector_db_dir, exist_ok=True, mode=0o777)

        # Build FAISS index
        embedding = OpenAIEmbeddings()
        global vectorstore, retriever
        logger.info("Creating vector embeddings...")
        vectorstore = FAISS.from_documents(docs, embedding)
        vectorstore.save_local(vector_db_dir)
        retriever = vectorstore.as_retriever()
        logger.info("Vector store created and saved successfully.")
        return True
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise


def build_rag_chain(question: str) -> str:
    """
    - Retrieves context documents via FAISS.
    - Prompts ChatOpenAI (gpt-4-turbo) to answer strictly based on context.
    """
    logger.info(f"Received question: {question}")
    if retriever is None:
        logger.error("Document not processed. Call process_document() first.")
        raise RuntimeError("Document not processed. Call process_document() first.")

    try:
        # Update to use newer invoke method instead of get_relevant_documents
        logger.info("Retrieving relevant documents...")
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        logger.info(f"Retrieved {len(docs)} relevant document chunks.")

        detailed_prompt = (
            """
You are an AI assistant designed to provide precise, document-based answers 
for an instruction manual. Your responses must strictly adhere to the 
information contained in the provided context. If a section of the manual 
is referenced (e.g., 'Section 4.4', or a specific error code), ensure that your explanation is highly 
detailed and aligns exactly with what the manual states.

When explaining a section, include:
- **A clear breakdown** of the procedures, instructions, or guidelines.
- **Step-by-step details** for tasks, if applicable.
- **Any safety precautions, warnings, or restrictions** mentioned in the manual.
- **Specific terminology and references** exactly as they appear in the document.
- **Technical specifications** (e.g., measurements, equipment details).
- **Tables, figures, or examples** if relevant.

If the provided context does not contain the required information, 
state clearly that you cannot answer instead of making assumptions.


---
**Context:**  {context}
---
**Question:** {question}
"""
        )
        prompt_template = ChatPromptTemplate.from_template(detailed_prompt)
        chain = prompt_template | ChatOpenAI(model="gpt-4-turbo") | StrOutputParser()
        logger.info("Generating answer...")
        result = chain.invoke({"context": context, "question": question})
        logger.info("Answer generated successfully.")
        return result if isinstance(result, str) else result.content
    except Exception as e:
        logger.error(f"Error in RAG chain: {str(e)}", exc_info=True)
        raise

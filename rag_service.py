import os
import uuid
import shutil
import logging
from typing import List
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.getenv("pcsk_496rDQ_2AFKKJ9FEVZEuMKg1dHRtwpp2Vw7zUQet3HCaAyXRfQDYuW1eWynaJGSbwcK3zg"))
index_name = "https://ragpropel-4ie225z.svc.aped-4627-b74a.pinecone.io" 

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # match your embedding size
        metric="cosine",  # or 'euclidean' or 'dotproduct' depending on use case
        spec=ServerlessSpec(
            cloud="aws",
            region=os.getenv("PINECONE_REGION")  # e.g., "us-west-2"
        )
    )


# Access the index
index = pc.Index(index_name)

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("⚠️ OPENAI_API_KEY not set. Define it in your .env file.")


index_name = os.getenv("PINECONE_INDEX_NAME")
embedding_model = OpenAIEmbeddings()
vectorstore = None
retriever = None


def process_document(document_path: str, content_dir: str = "content") -> bool:
    """Parse, summarize, and store document embeddings in Pinecone."""
    logger.info(f"Processing document: {document_path}")

    if os.path.exists(content_dir):
        shutil.rmtree(content_dir)
    os.makedirs(content_dir, exist_ok=True, mode=0o777)

    from unstructured.partition.pdf import partition_pdf
    logger.info("Partitioning PDF...")
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

    logger.info(f"PDF partitioned into {len(elements)} elements.")
    docs: List[Document] = []

    for el in elements:
        if hasattr(el, "text") and el.text:
            content = el.text
            prompt_txt = "Summarize this technical text focusing on key specifications:"
        elif hasattr(el.metadata, "text_as_html") and el.metadata.text_as_html:
            content = el.metadata.text_as_html
            prompt_txt = "Explain this table's key data points and relationships:"
        else:
            continue

        model = ChatOpenAI(model="gpt-3.5-turbo-0125")
        prompt = ChatPromptTemplate.from_template(f"{prompt_txt}\n\n{{content}}")
        chain = prompt | model | StrOutputParser()
        summary = chain.invoke({"content": content})
        text = summary if isinstance(summary, str) else summary.content

        docs.append(
            Document(
                page_content=text.strip(),
                metadata={"id": str(uuid.uuid4()), "type": el.__class__.__name__}
            )
        )

    if not docs:
        raise ValueError("No content extracted to index.")

    # Create Pinecone index if not exists
    if index_name not in pinecone.list_indexes():
        logger.info(f"Creating Pinecone index: {index_name}")
        pinecone.create_index(name=index_name, dimension=1536, metric="cosine")

    index = pinecone.Index(index_name)
    global vectorstore, retriever
    logger.info("Indexing documents in Pinecone...")
    vectorstore = PineconeStore.from_documents(docs, embedding_model, index_name=index_name)
    retriever = vectorstore.as_retriever()
    logger.info("Documents indexed successfully.")
    return True


def build_rag_chain(question: str) -> str:
    logger.info(f"Received question: {question}")
    if retriever is None:
        logger.error("Document not processed. Call process_document() first.")
        raise RuntimeError("Document not processed. Call process_document() first.")

    try:
        logger.info("Retrieving relevant documents...")
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        logger.info(f"Retrieved {len(docs)} relevant documents.")

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
        result = chain.invoke({"context": context, "question": question})
        return result if isinstance(result, str) else result.content
    except Exception as e:
        logger.error(f"Error in RAG chain: {str(e)}", exc_info=True)
        raise

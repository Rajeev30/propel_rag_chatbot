# ─── PREVENT STREAMLIT/TORCH RUNTIME ERRORS ───────────────────────
import os
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # disable watcher that crashes on torch.classes

import nest_asyncio
import asyncio
nest_asyncio.apply()
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import types, sys
try:
    import torch
    if not isinstance(torch.classes, types.ModuleType):
        dummy = types.ModuleType("torch.classes")
        dummy.__path__ = []
        torch.classes = dummy
        sys.modules["torch.classes"] = dummy
except Exception as e:
    print(f"[patch] torch patch skipped: {e}")
# ──────────────────────────────────────────────────────────────────

import re, pytesseract
import spacy, csv, mimetypes
import os
from dotenv import load_dotenv
from pathlib import Path
import fitz  # PyMuPDF
import os, base64
from pathlib import Path
from tqdm import tqdm
import fitz                         # PyMuPDF 2.0+
import tabula                       # needs Java;  pip install tabula-py jpype1
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import os, base64, json, fitz, pdfplumber
from tqdm import tqdm
import tabula
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from PIL import Image
import imagehash
import base64, mimetypes, uuid, os, time
from tqdm import tqdm
import openai                     # v1.14+
from langchain_openai import OpenAIEmbeddings
import pinecone 
from pinecone import ServerlessSpec
# import streamlit as st

load_dotenv()                           # expects .env in the repo root

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV")  # e.g. "gcp-starter"

nlp = spacy.load("en_core_web_sm")

def load_pdf(path: Path) -> str:
    doc = fitz.open(path)
    text = []
    for page in doc:
        text.append(page.get_text("text"))
    return "\n".join(text)

raw_text = load_pdf(Path("PyMUPDF_Approach/doc2.pdf"))


# ---------- CONFIG ----------
PDF_PATH  = Path("PyMUPDF_Approach/doc2.pdf")
BASE_DIR  = Path("data")
DIAGRAM_THRESHOLD = 0.60   # % of page area to be considered full-page diagram
LOGO_THRESHOLD    = 0.10   # % of page area under which we skip as logo/ornament
TEXT_EMPTY_LIMIT   = 100   # page is “empty” if ≤ 100 chars
NEIGHBOR_MIN_CHARS = 200   # char count to decide page is "mostly image"
PIXMAP_ZOOM       = 3      # render factor for full page images
# ----------------------------


# 1 · utils ------------------------------------------------------
def ensure_dirs():
    for sub in ["images", "text", "tables", "page_images"]:
        (BASE_DIR / sub).mkdir(parents=True, exist_ok=True)

def save_txt(fname: Path, txt: str):
    fname.write_text(txt, encoding="utf-8")

def save_pixmap(pix: fitz.Pixmap, fname: Path):
    if pix.alpha or pix.colorspace.n > 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    pix.save(fname)


# 2 · caption cache stub ----------------------------------------
CAPTION_CACHE_FILE = "captions_cache.json"
if Path(CAPTION_CACHE_FILE).exists():
    CAPTION_CACHE = json.loads(Path(CAPTION_CACHE_FILE).read_text())
else:
    CAPTION_CACHE = {}

def caption_image_cached(path: Path) -> str:
    key = str(path.resolve())
    if key in CAPTION_CACHE:
        return CAPTION_CACHE[key]
    # --- call your GPT-4o vision captioner here ---
    caption = "[CAPTION]"  # placeholder
    CAPTION_CACHE[key] = caption
    Path(CAPTION_CACHE_FILE).write_text(json.dumps(CAPTION_CACHE, indent=2))
    return caption

# Global trackers
HASH_COUNTS = {}
SKIP_HASHES = set()

def pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    """Convert PyMuPDF pixmap to PIL Image, with fallback for broken samples"""
    try:
        # Convert via bytes safely
        mode = "RGB" if pix.n < 5 else "RGBA"
        img_bytes = pix.tobytes(output="ppm")  # robust fallback
        return Image.open(BytesIO(img_bytes)).convert(mode)
    except Exception as e:
        print(f"[warn] pixmap_to_pil failed: {e}")
        return None

def should_skip_pil_image(pil_img: Image.Image, min_repeat: int = 3) -> bool:
    """Decide whether to skip an image based on perceptual hash frequency"""
    img_hash = str(imagehash.phash(pil_img))
    if img_hash in SKIP_HASHES:
        return True
    HASH_COUNTS[img_hash] = HASH_COUNTS.get(img_hash, 0) + 1
    if HASH_COUNTS[img_hash] >= min_repeat:
        SKIP_HASHES.add(img_hash)
        return True
    return False

def extract_headings(words):
    """Heuristic: return candidate headings by largest font sizes (based on height)."""
    if not words: 
        return []
    # Estimate font size from height
    for w in words:
        w["size"] = round(w["bottom"] - w["top"], 2)

    # Pick largest 2-3 sizes as likely headings
    sizes = sorted({w["size"] for w in words}, reverse=True)[:3]
    headings = [w for w in words if w["size"] in sizes and w["text"].strip()]

    # Group into lines
    result = []
    line = ""
    y_prev = None
    for w in headings:
        if y_prev is None or abs(w["top"] - y_prev) < 2:
            line += " " + w["text"]
        else:
            result.append(line.strip())
            line = w["text"]
        y_prev = w["top"]
    if line.strip(): 
        result.append(line.strip())
    return result

WARNING_TAGS = ("WARNING", "CAUTION", "DANGER", "NOTE", "STEP")
def tag_special_chunks(txt:str)->str:
    for tag in WARNING_TAGS:
        if txt.upper().startswith(tag):
            return f"[{tag}] {txt}"
    return txt

# 3 · main extractor --------------------------------------------
def extract_pdf(pdf_path: Path):
    ensure_dirs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)

    items = []
    doc   = fitz.open(pdf_path)

    for page_idx in tqdm(range(len(doc)), desc="Processing pages"):
        page = doc[page_idx]
        # Default section heading for page
        current_h = ""
        try:
            with pdfplumber.open(pdf_path) as plumber_pdf:
                plumber_page = plumber_pdf.pages[page_idx]
                pl_words = plumber_page.extract_words()
                headings = extract_headings(pl_words)
                current_h = headings[0] if headings else ""
        except Exception as e:
            print(f"[warn] heading extraction failed on page {page_idx}: {e}")

        raw_text = page.get_text("text", sort=True).strip()
        if len(raw_text) < 20:            # likely scanned text
            ocr_img = page.get_pixmap(matrix=fitz.Matrix(2,2))
            ocr_txt = pytesseract.image_to_string(Image.open(BytesIO(ocr_img.tobytes()))).strip()
            if len(ocr_txt) > 20:
                raw_text = ocr_txt

        # ---------- Table extraction ----------
        # ---------- Table extraction with fallback ----------
        import csv

        # ---------- Table extraction with fallback and cleanup ----------
        tables = []

        # Try Tabula (best for ruled tables)
        try:
            tables = tabula.read_pdf(str(pdf_path), pages=page_idx + 1, multiple_tables=True)
        except Exception as e:
            print(f"[warn] tabula failed on page {page_idx}: {e}")

        # Fallback to pdfplumber
        if not tables:
            try:
                with pdfplumber.open(pdf_path) as plumber_pdf:
                    plumber_page = plumber_pdf.pages[page_idx]
                    extracted = plumber_page.extract_tables()
                    for t in extracted:
                        if any(cell for row in t for cell in row if cell and cell.strip()):
                            tables.append(t)
            except Exception as e:
                print(f"[warn] pdfplumber failed on page {page_idx}: {e}")

        # Process extracted tables
        for t_idx, table in enumerate(tables):
            cleaned_rows = []

            # Tabula: pandas DataFrame
            if hasattr(table, "values"):
                rows = table.values.tolist()
            else:
                rows = table  # pdfplumber: list of lists

            # Clean: strip whitespace, flatten \n, fill empty with ""
            for row in rows:
                row = [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
                if any(cell for cell in row):  # skip empty rows
                    cleaned_rows.append(row)

            # --- Reconstruct multiline rows ---
            reconstructed_rows = []
            current_row = None

            for row in cleaned_rows:
                if row[0].strip():  # new row
                    if current_row:
                        reconstructed_rows.append(current_row)
                    current_row = row
                else:  # continuation of previous row
                    if current_row:
                        for i in range(len(row)):
                            if row[i].strip():
                                current_row[i] += " " + row[i]

            if current_row:
                reconstructed_rows.append(current_row)

            # Convert to clean text for embedding
            table_text = "\n".join([" | ".join(row) for row in reconstructed_rows])

            # Save .txt for RAG
            f = BASE_DIR / "tables" / f"{pdf_path.stem}_table_{page_idx}_{t_idx}.txt"
            save_txt(f, table_text)

            # Save .csv for inspection
            csv_path = f.with_suffix(".csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(reconstructed_rows)

            # Append to items for embedding
            items.append({
                "page": page_idx,
                "type": "table",
                "text": table_text,
                "path": str(f)
            })

            m = re.search(r"(Table\s+\d+\s*[\–\-—]\s*[^\n]{5,80})", raw_text)
            if m:
                items[-1]["caption_title"] = m.group(1).strip()

        # ---------- Text chunking ----------
        if raw_text:
            for i, chunk in enumerate(splitter.split_text(raw_text)):
                f = BASE_DIR / "text" / f"{pdf_path.stem}_text_{page_idx}_{i}.txt"
                save_txt(f, chunk)
                items.append({
                    "page": page_idx,
                    "type": "text",
                    "text": chunk,
                    "section": current_h,           # ▶️  add detected heading
                    "path": str(f)
                })

        # ---------- Image logic ----------
        page_bbox = page.rect
        page_area = page_bbox.width * page_bbox.height
        full_page_diagram = False

        with pdfplumber.open(pdf_path) as plumber_pdf:
            plumber_page = plumber_pdf.pages[page_idx]
            imgs = plumber_page.images  # gives bbox coords
            print(f"[Page {page_idx}] {len(imgs)} images found")
            for img in imgs:
                img_area = img["width"] * img["height"]
                print(f"  → {round(img_area / page_area * 100, 2)}% of page")

        # Filter out tiny logos
        large_imgs = [
            img for img in imgs
            if (img["width"] * img["height"]) / page_area > LOGO_THRESHOLD
        ]

        # Heuristic: one large image covers most of the page
        composite_area = sum(img["width"] * img["height"] for img in imgs if (img["width"] * img["height"]) / page_area > 0.15)

        if composite_area / page_area > 0.40:
            full_page_diagram = True

        if full_page_diagram:
            # Render whole page at high res
            pix = page.get_pixmap(matrix=fitz.Matrix(PIXMAP_ZOOM, PIXMAP_ZOOM))
            f = BASE_DIR / "page_images" / f"diagram_page_{page_idx:03d}.png"
            save_pixmap(pix, f)
            caption = caption_image_cached(f)
            # If page text is too short, pull context from neighbors
            related = ""
            if len(raw_text) <= TEXT_EMPTY_LIMIT:
                for adj_idx in (page_idx - 1, page_idx + 1):
                    if 0 <= adj_idx < len(doc):
                        adj_text = doc[adj_idx].get_text("text", sort=True).strip()
                        if len(adj_text) >= NEIGHBOR_MIN_CHARS:
                            related += f"\n[[Neighbor page {adj_idx}]]\n" + adj_text
            items.append({
            "page": page_idx,
            "type": "page",
            "path": str(f),
            "text": caption,          # the GPT-4o caption
            "related_text": related.strip()  # may be empty if nothing useful
            })
            print(f"✅ Detected composite diagram on page {page_idx} covering {round(composite_area / page_area * 100, 2)}%")

            m = re.search(r"(Figure\s+\d+\s*[\–\-—]\s*[^\n]{5,80})", raw_text)
            if m:
                items[-1]["caption_title"] = m.group(1).strip()

        else:
            # Normal inline images
            page_area = page.rect.width * page.rect.height

            with pdfplumber.open(pdf_path) as plumber_pdf:
                plumber_page = plumber_pdf.pages[page_idx]
                imgs = plumber_page.images  # contains width, height, x0, y0, etc.

            for idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]

                # 🔍 Find the matching image in plumber by xref (or fallback to estimated area)
                matching = next((i for i in imgs if i.get("name") == img[7]), None)

                if matching:
                    img_area = matching["width"] * matching["height"]
                else:
                    pix = fitz.Pixmap(doc, xref)
                    img_area = pix.width * pix.height

                percent = img_area / page_area

                if percent < 0.01:
                    print(f"🗑️ Skipped small image on page {page_idx}: {round(percent * 100, 2)}% of page")
                    continue

                pix = fitz.Pixmap(doc, xref)
                fname = BASE_DIR / "images" / f"{pdf_path.stem}_img_{page_idx}_{idx}_{xref}.png"
                save_pixmap(pix, fname)

                items.append({
                    "page": page_idx,
                    "type": "image",
                    "path": str(fname),
                    "text": caption_image_cached(fname)
                })

                m = re.search(r"(Figure\s+\d+\s*[\–\-—]\s*[^\n]{5,80})", raw_text)
                if m:
                    items[-1]["caption_title"] = m.group(1).strip()

    print("\n🧾 Pages with full-page diagrams:")
    print([item["page"] for item in items if item["type"] == "page"])
    


    print(f"✅ finished – extracted {len(items)} items")
    return items


# ---------------- RUN ----------------
items = extract_pdf(PDF_PATH)

# quick sanity prints
first_text  = next(i for i in items if i["type"] == "text")
first_image = next(i for i in items if i["type"] == "image" or i["type"] == "page")
print("\n[Sample text]", first_text["text"][:200])
print("[Sample image]", first_image["path"])


openai.api_key      = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")

# ── embedding & chat models (adjust if your account differs) ────────────
EMBED_MODEL = "text-embedding-3-large"    # 1536-D, GPT-4 family
CHAT_MODEL  = "gpt-4o-mini"               # vision-capable

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # keep this

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)
index_name = pinecone.Index("multimodal-manual")

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # or embeddings.embedding_dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pinecone.Index(index_name)

def img_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path)
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime or 'image/png'};base64,{b64}"

def caption_image(path: Path, retry: int = 3) -> str:
    """Return a single-sentence literal caption for the image."""
    data_uri = img_to_data_uri(path)
    for attempt in range(retry):
        try:
            resp = openai.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "Describe this image in one concise sentence, no interpretation."},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }],
                max_tokens=60
            )
            return resp.choices[0].message.content.strip()
        except openai.RateLimitError:
            wait = 2 ** attempt
            print(f"rate-limited, retrying in {wait}s"); time.sleep(wait)
    return "unavailable caption"

import json

caption_cache_file = "captions_cache.json"
caption_cache = {}

# Load existing cache if present
if Path(caption_cache_file).exists():
    caption_cache = json.loads(Path(caption_cache_file).read_text())

def caption_image_cached(path: Path) -> str:
    key = str(path)
    if key in caption_cache:
        return caption_cache[key]
    caption = caption_image(path)  # your existing GPT-4 captioner
    caption_cache[key] = caption
    Path(caption_cache_file).write_text(json.dumps(caption_cache, indent=2))
    return caption


docs, meta, ids = [], [], []

# 🧾 Build a quick index of text chunks by page
text_by_page = {
    obj["page"]: obj["text"].strip()
    for obj in items if obj["type"] == "text" and obj["text"].strip()
}

# Prepare logs for captions and skipped pages
captioned_pages = []
skipped_pages = []

for obj in tqdm(items, desc="Preparing docs + metadata"):
    # ---------------- TEXT & SECTION ----------------
    if obj["type"] == "text":
        section  = obj.get("section", "").strip()
        chunk    = tag_special_chunks(obj["text"])
        text_repr = f"[SECTION] {section}\n{chunk}" if section else chunk

        keywords = [tok.lemma_ for tok in nlp(text_repr)
                    if tok.pos_ in {"NOUN","PROPN"}]

        docs.append(text_repr)
        meta.append({**obj, "keywords": keywords})
        ids.append(str(uuid.uuid4()))

    # ---------------- TABLE  (row-level) ------------
    elif obj["type"] == "table":
        rows = obj["text"].splitlines()
        if not rows:
            continue

        # Try to parse header row from first line
        header_fields = [col.strip() for col in rows[0].split(" | ")]

        for row in rows[1:]:
            cells = [col.strip() for col in row.split(" | ")]
            if not any(cells):
                continue

            # Align columns if possible
            if len(cells) == len(header_fields):
                structured = " | ".join(f"{h}: {v}" for h, v in zip(header_fields, cells))
            else:
                structured = row  # fallback to raw

            text_repr = f"[TABLE ROW] {structured}"

            # optional: extract keywords for filtering
            keywords = [tok.lemma_ for tok in nlp(text_repr) if tok.pos_ in {"NOUN", "PROPN"}]

            docs.append(text_repr)
            meta.append({**obj, "keywords": keywords})
            ids.append(str(uuid.uuid4()))


    # ---------------- IMAGE & PAGE ------------------
    elif obj["type"] in {"image", "page"}:
        caption  = obj.get("caption") or obj.get("caption_title","")
        related  = obj.get("related_text","").strip()
        prefix   = "[PAGE IMAGE]" if obj["type"]=="page" else "[IMAGE]"
        text_repr = f"{prefix} {caption}\n{related}" if related else f"{prefix} {caption}"

        keywords = [tok.lemma_ for tok in nlp(text_repr)
                    if tok.pos_ in {"NOUN","PROPN"}]

        docs.append(text_repr)
        meta.append({**obj, "keywords": keywords})
        ids.append(str(uuid.uuid4()))


# ── embed in batches ───────────────────────────────────────────
batch = 100
vectors = []
for i in tqdm(range(0, len(docs), batch), desc="Embedding"):
    vecs = embeddings.embed_documents(docs[i:i+batch])
    vectors.extend(vecs)

to_upsert = [
    (
        ids[i],
        vectors[i],
        {**meta[i], "text": docs[i]}          # ensure "text" key exists
    )
    for i in range(len(docs))
]
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

index = pinecone.Index("multimodal-manual")
        # ← already created with dim=3072

def payload_size(batch):
    return len(json.dumps(batch).encode())

MAX_PAYLOAD = 3_000_000        # ≈3 MB (well under 4 MB limit)
batch = []

print("📤  Upserting to Pinecone …")

for tup in tqdm(to_upsert):
    batch.append(tup)
    if payload_size(batch) >= MAX_PAYLOAD:
        index.upsert(batch, namespace="v1")
        batch = []

# any leftovers
if batch:
    index.upsert(batch, namespace="v1")

# # ------------------------------------------------------------
# # 4 · VERIFY
# # ------------------------------------------------------------
# stats = index.describe_index_stats(namespace="v1")
# print("\n✅  Done!  Pinecone now holds:")
# print(f"   • vectors : {stats['namespaces']['v1']['vector_count']}")
# print(f"   • dim      : {stats['dimension']}")

from collections import Counter

def extract_table_keywords(meta, limit=20):
    keyword_counts = Counter()

    for m in meta:
        if m.get("type") == "table":
            keyword_counts.update(m.get("keywords", []))

    top_keywords = [kw for kw, _ in keyword_counts.most_common(limit)]
    return top_keywords

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from IPython.display import display, Image
from pathlib import Path


# Vector store
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace="v1",
    text_key="text"
)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"  # ✅ Tells memory what to store
)

# RAG Chain with output key fix
chat_rag = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)


# def ask(query):
#     def is_table_query(q):
#         # Add more generic patterns here as needed
#         table_keywords = extract_table_keywords(meta)
#         return any(kw in q.lower() for kw in table_keywords)

#     # 1. Run table-prioritized retrieval if query looks structured
#     table_mode = is_table_query(query)
#     if table_mode:
#         all_docs = retriever.get_relevant_documents(query)
#         table_docs = [doc for doc in all_docs if doc.metadata.get("type") == "table"]
#         source_docs = table_docs[:5] if table_docs else all_docs[:5]
#     else:
#         res = chat_rag.invoke({"question": query})
#         source_docs = res["source_documents"]

#     # 2. If not using the RAG chain (manual filtered docs), rerun LLM
#     if "source_docs" in locals() and isinstance(source_docs, list) and "res" not in locals():
#         content_blob = "\n\n".join(doc.page_content for doc in source_docs)
#         system_prompt = "You are answering from a technical manual. Prioritize structured information when available, especially from tables."
#         res = llm.invoke([
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": f"{query}\n\nRelevant context:\n{content_blob}"}
#         ])
#         res = {"answer": res.content, "source_documents": source_docs}

#     # 3. Display any image or diagram found
#     top_image_path = None
#     top_image_doc  = None
#     for doc in res["source_documents"]:
#         doc_type = doc.metadata.get("type")
#         path     = doc.metadata.get("path")
#         if doc_type in {"image", "page"} and Path(path).suffix in {".png", ".jpg", ".jpeg"}:
#             top_image_path = path
#             top_image_doc  = doc
#             break

#     if top_image_path:
#         print("\n🖼️  You can refer to the following diagram:")
#         print(f"• {top_image_doc.metadata.get('type')} → {top_image_path} (page {top_image_doc.metadata.get('page')})")
#         try:
#             display(Image(filename=top_image_path, width=400))
#         except Exception as e:
#             print(f"[!] Failed to display image: {e}")

#     # 4. Print response and sources
#     print(f"\n🧑: {query}")
#     print(f"🤖: {res['answer']}")

#     print("\n--- Sources ---")
#     for doc in res["source_documents"]:
#         print(f"• {doc.metadata.get('type')} → {doc.metadata.get('path')} (page {doc.metadata.get('page')})")

from pathlib import Path
from PIL import Image

def ask(query):
    def is_table_query(q):
        table_keywords = extract_table_keywords(meta)  # Ensure meta is global or passed in
        return any(kw in q.lower() for kw in table_keywords)

    # Step 1: Determine if it's a table-driven query
    table_mode = is_table_query(query)

    if table_mode:
        all_docs = retriever.invoke(query)
        table_docs = [doc for doc in all_docs if doc.metadata.get("type") == "table"]
        source_docs = table_docs[:5] if table_docs else all_docs[:5]
    else:
        res = chat_rag.invoke({"question": query})
        source_docs = res["source_documents"]

    # Step 2: If not using the RAG chain, invoke manually
    if "source_docs" in locals() and isinstance(source_docs, list) and "res" not in locals():
        context_blob = "\n\n".join(doc.page_content for doc in source_docs)
        system_prompt = "You are answering from a technical manual. Prioritize structured information when available, especially from tables."
        completion = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{query}\n\nRelevant context:\n{context_blob}"}
        ])
        res = {"answer": completion.content, "source_documents": source_docs}

    # Step 3: Look for top image (if any)
    top_image_path = None
    top_image_doc = None

    for doc in res["source_documents"]:
        doc_type = doc.metadata.get("type")
        path = doc.metadata.get("path")
        if doc_type in {"image", "page"} and Path(path).suffix in {".png", ".jpg", ".jpeg"}:
            top_image_path = path
            top_image_doc = doc
            break

    # Step 4: Return everything as structured data
    # return {
    # Deduplicate page numbers from sources
    page_numbers = sorted({
        int(doc.metadata.get("page", -1))
        for doc in res["source_documents"]
        if doc.metadata.get("page") is not None
    })

    # Format explanation sentence
    if page_numbers:
        if len(page_numbers) == 1:
            source_summary = f"You can refer to page {page_numbers[0]} in the document for more details."
        else:
            pages = ", ".join(map(str, page_numbers))
            source_summary = f"Relevant information is available on pages {pages} of the document."
    else:
        source_summary = "No source pages identified."

    # Return the final answer, summary sentence, and image (if any)
    return res["answer"], [source_summary], top_image_path


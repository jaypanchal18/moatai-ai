import hashlib
import tiktoken
import pdfplumber
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DIMENSION = 1536
PINECONE_REGION = "us-east-1"
INDEX_NAME = "rag-index"

# Initialize Pinecone
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
pinecone_spec = ServerlessSpec(cloud="aws", region=PINECONE_REGION)

if INDEX_NAME not in [index.name for index in pinecone_client.list_indexes()]:
    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="euclidean",
        spec=pinecone_spec
    )

pinecone_index = pinecone_client.Index(INDEX_NAME)

# Persistent data structures
doc_store = []
bm25_vectorizer = None
bm25_matrix = None
indexed_files = {}


# Rate limiter for OpenAI API
class RateLimiter:
    def __init__(self, calls_per_second):
        self.interval = 1.0 / calls_per_second
        self.last_call = 0
        self.lock = Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()

rate_limiter = RateLimiter(calls_per_second=2)
openai_lock = Lock()

# Helper functions
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def hash_file(file_content):
    hasher = hashlib.sha256()
    hasher.update(file_content)
    return hasher.hexdigest()

def call_openai_api_with_retries(endpoint, data, headers, retries=5, delay=1, timeout=30):
    for attempt in range(retries):
        try:
            rate_limiter.wait()
            with openai_lock:
                response = requests.post(endpoint, json=data, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            elif response.status_code in {429, 503}:
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                response.raise_for_status()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise Exception("OpenAI API is unavailable after retries.")

def generate_embedding(text):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {"input": text, "model": "text-embedding-ada-002"}
    response = call_openai_api_with_retries("https://api.openai.com/v1/embeddings", data, headers)
    return response["data"][0]["embedding"]

def split_text_into_chunks(text, max_tokens=500, overlap=250):
    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokenizer.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks


most_recent_file_hash = None

def index_document(file_hash, text):
    global doc_store, bm25_vectorizer, bm25_matrix, most_recent_file_hash
    if file_hash in indexed_files:
        return "File already indexed."

    chunks = split_text_into_chunks(text)
    documents = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        for i, chunk in enumerate(chunks):
            documents.append(index_chunk(i, chunk, text, file_hash))

    indexed_files[file_hash] = True
    most_recent_file_hash = file_hash 

    if documents:
        bm25_vectorizer = TfidfVectorizer().fit(documents)
        bm25_matrix = bm25_vectorizer.transform(documents)
    return "Document indexed successfully."




def generate_context(chunk, full_text, max_context_tokens=3000):
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    full_text_tokens = tokenizer.encode(full_text)
    if len(full_text_tokens) > max_context_tokens:
        full_text = tokenizer.decode(full_text_tokens[:max_context_tokens])

    prompt = f"""
    <document>
    {full_text}
    </document>
    <chunk>
    {chunk}
    </chunk>
    Extract specific context related to this chunk.
    """
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {"model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}], "max_tokens": 75}
    response = call_openai_api_with_retries(
        "https://api.openai.com/v1/chat/completions", data, headers
    )
    return response.get("choices", [{}])[0].get("message", {}).get("content")


def index_chunk(i, chunk, full_text, file_hash):
    embedding = generate_embedding(chunk)
    if embedding:
       
        vector = np.array(embedding).astype("float32").tolist()
        metadata = {"chunk_id": str(i), "content": chunk, "file_hash": file_hash}
       
        pinecone_index.upsert([(str(i), vector, metadata)])
        doc_store.append({"id": i, "text": chunk, "file_hash": file_hash})
        
        return chunk
    return None


def query_search_combined(prompt, top_k=5, file_hash=None):
   
    file_hash = file_hash or most_recent_file_hash
    
    if not file_hash:
        raise ValueError("No file uploaded or available for querying.")

    try:
        query_embedding = generate_embedding(prompt)
        pinecone_results = pinecone_index.query(
            vector=query_embedding, top_k=top_k, include_metadata=True
        )
        # Filter pinecone results by the most recent file_hash
        pinecone_texts = [
            (match.metadata["content"], match.metadata["file_hash"])
            for match in pinecone_results.matches if match.metadata["file_hash"] == file_hash
        ]
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        pinecone_texts = []

    bm25_texts = []
    if bm25_vectorizer and bm25_matrix is not None:
        query_vector_bm25 = bm25_vectorizer.transform([prompt])
        bm25_scores = (bm25_matrix @ query_vector_bm25.T).toarray().flatten()
        top_bm25_indices = bm25_scores.argsort()[-top_k:][::-1]
        
        # Filter BM25 results by the most recent file_hash
        bm25_texts = [
            (doc_store[idx]["text"], doc_store[idx]["file_hash"]) 
            for idx in top_bm25_indices if doc_store[idx]["file_hash"] == file_hash
        ]

    # Combine and remove duplicates
    combined_results = list(dict.fromkeys(pinecone_texts + bm25_texts))
    return combined_results[:top_k]


def rerank_with_openai(prompt, retrieved_chunks):
    context = "\n\n".join([chunk[0] for chunk in retrieved_chunks]) 
    metadata = [chunk[1] for chunk in retrieved_chunks] 
    
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    messages = [
        {"role": "system", "content": "You are an assistant that retrieves and summarizes relevant information with high precision."},
        {"role": "user", "content": f"Context (Files: {', '.join(metadata)}): {context}"},
        {"role": "user", "content": f"Query: {prompt}\n\nProvide the most relevant information based on the context."}
    ]
    data = {"model": "gpt-4", "messages": messages, "max_tokens": 300}
    response = call_openai_api_with_retries("https://api.openai.com/v1/chat/completions", data, headers)
    
    if response and "choices" in response and len(response["choices"]) > 0:
        return response["choices"][0].get("message", {}).get("content", "No response content.")
    return "No valid response from OpenAI API."


@csrf_exempt
def upload_with_predefined(request):
    if request.method == "POST":
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file provided."}, status=400)

        file = request.FILES['file']
        file_content = file.read()
        file_hash = hash_file(file_content)

        # Extract text from the PDF
        text = extract_text_from_pdf(file)

        # Predefined queries
        predefined_prompts = {
            "soc_auditor": "Who did Soc Report.",
            "system_scope": "Extract scope of the soc report only without further details.",
            "tsp_control":"Extract TSP Covered in SOC Report",
        }

        # Process predefined prompts
        results = {}
        for key, prompt in predefined_prompts.items():
            results[key] = rerank_with_openai(prompt, [(text, file_hash)])  

        # Check if the file is already indexed
        if file_hash not in indexed_files:
            
            index_document(file_hash, text)
            message = "File uploaded and indexed successfully."
        else:
            message = "File already indexed. Predefined results processed."

        # Format results for UI
        response_data = {
            "message": message,
            "predefined_results": [
                {
                    "title": "SOC auditor",
                    "type": "audit",
                    "content": results["soc_auditor"]
                },
                {
                    "title": "System in the SOC audit scope",
                    "type": "audit",
                    "content": results["system_scope"]
                },
                {
                    "title": "TSP Control in the scope of audit",
                    "type": "audit",
                    "content": results["tsp_control"]
                }
            ]
        }

        return JsonResponse(response_data)

    return JsonResponse({"error": "Only POST method is allowed."}, status=405)


@csrf_exempt
def query(request):
    if request.method == "POST":
        try:
            # Parse JSON payload
            data = json.loads(request.body.decode('utf-8'))
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON payload."}, status=400)

        prompt = data.get('query')
        file_hash = data.get('file_hash') 
        if not prompt:
            return JsonResponse({"error": "Query is required."}, status=400)

        result_chunks = query_search_combined(prompt, file_hash=file_hash)
        refined_answer = rerank_with_openai(prompt, result_chunks)
        return JsonResponse({"result": refined_answer})

    return JsonResponse({"error": "Only POST method is allowed."}, status=405)


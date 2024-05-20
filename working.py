import fitz  # PyMuPDF
import ollama
import json
import os
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()

src_dir = "/home/bharat/codebase/text_searching/kb"
target_dir = "/home/bharat/codebase/text_searching/"
document_name = "literature/english-12.pdf"

# break dwon the pdf into smaller text para and return "array of strings"
def create_paragraphs(src_dir, document_name):
    paragraphs = []
    pdf_path = f"{src_dir}/{document_name}"
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text().strip()
        if text:
            paragraphs.append(text)
    return paragraphs
    
# create a DIR and save the embeddings with the given document_name
def save_embeddings(document_name, embeddings, target_dir):
    filename_without_ext = os.path.splitext(document_name)[0]
    embeddings_folder = os.path.join(target_dir, f"bharat/{filename_without_ext}")
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)
    with open(f"{embeddings_folder}/embeddings.json", "w") as f:
        json.dump(embeddings, f)

# return the embeddings available at the embedding_location
def get_embeddings(embeddings_folder):
    embeddings_path = os.path.join(embeddings_folder, "embeddings.json")
    if not os.path.exists(embeddings_path):
        return False
    with open(embeddings_path, "r") as f:
        return json.load(f)

# return embeddings from the array of strings (aka chunks)
def create_embeddings(model_name, document_name, paragraphs):
    filename_without_ext = os.path.splitext(document_name)[0]
    embeddings_folder = os.path.join(target_dir, f"vectors/{filename_without_ext}")
    if (embeddings := get_embeddings(embeddings_folder)) is not False:
        return embeddings
    embeddings = [ollama.embeddings(model=model_name, prompt=paragraph)["embedding"] for paragraph in paragraphs]
    save_embeddings(document_name, embeddings, target_dir)
    return embeddings

 # use math tools to perform match
def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

# this is the starting point
def generate_embeddings(src_dir, document_name, target_dir, model_name):
    paragraphs = create_paragraphs(src_dir, document_name)
    embeddings = create_embeddings(model_name, document_name, paragraphs)
    save_embeddings(document_name, embeddings, target_dir)
    return paragraphs, embeddings

# Define a Pydantic model for the question payload
class QuestionBody(BaseModel):
    question: str 
    embeddings_folder_path: str
    model_name_for_embeddings: str
    model_name_for_chat: str 
    best_match_count: int

    class Config:
        # Set protected_namespaces to an empty tuple to resolve the conflict
        protected_namespaces = ()

# Generate embeddings for the initial document
paragraphs, embeddings = generate_embeddings(src_dir, document_name, target_dir, "mxbai-embed-large")

# Endpoint to handle question and generate response
@app.post("/answer/")
async def main(question: QuestionBody):
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
        based on snippets of text provided in context. Answer only using the context provided,
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """
    
    prompt = question.question.strip()
    embeddings_folder = question.embeddings_folder_path
    all_embeddings = get_embeddings(embeddings_folder)
    
    if all_embeddings is False:
        return {"answer": "No embeddings found in the specified folder."}
    
    prompt_embedding = ollama.embeddings(model=question.model_name_for_embeddings, prompt=prompt)["embedding"]

    # Check if embeddings are of the same shape
    if len(prompt_embedding) != len(all_embeddings[0]):
        return {"answer": "Embedding dimension mismatch."}
    
    most_similar_chunks = find_most_similar(prompt_embedding, all_embeddings)[:question.best_match_count]

    response_texts = "\n".join(paragraphs[item[1]] for item in most_similar_chunks)
    
    response = ollama.chat(
        model=question.model_name_for_chat,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT + response_texts,
            },
            {"role": "user", "content": prompt},
        ],
    )
    print(response["message"])
    return {"answer": response["message"]}

# Start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


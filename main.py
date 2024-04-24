import fitz  # PyMuPDF
import ollama
import json
import os
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()

# Function to parse the text from a PDF file and extract paragraphs 
def parse_pdf(filename):
    paragraphs = []
    doc = fitz.open(filename)
    for page in doc:
        text = page.get_text().strip()
        if text:
            paragraphs.append(text)
    return paragraphs

# Function to save embeddings to a file
def save_embeddings(pdf_filename, embeddings):
    filename_without_ext = os.path.splitext(pdf_filename)[0]
    embeddings_folder = f"vectors/{filename_without_ext}"
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)
    with open(f"{embeddings_folder}/embeddings.json", "w") as f:
        json.dump(embeddings, f)

# Function to load embeddings from a file
def load_embeddings(embeddings_folder):
    if not os.path.exists(embeddings_folder):
        return False
    with open(f"{embeddings_folder}/embeddings.json", "r") as f:
        return json.load(f)

# Function to generate embeddings for paragraphs    
def get_embeddings(pdf_filename, modelname, chunks):
    embeddings_folder = f"embeddings/{pdf_filename}"
    if (embeddings := load_embeddings(embeddings_folder)) is not False:
        return embeddings
    embeddings = [ollama.embeddings(model=modelname, prompt=chunk)["embedding"] for chunk in chunks] #chunks: The list of text paragraphs
    save_embeddings(pdf_filename, embeddings)
    return embeddings

# Function to find the most similar embeddings to a given embedding
def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

# Define a Pydantic model for the question payload
class Question(BaseModel):
    question: str
    embeddings_folder_path: str

# List of PDF filenames
pdf_filenames = ["kb/literature/english-12.pdf","kb/literature/bio.pdf", "kb/technical/internet.pdf"]  

# Lists to store all paragraphs and embeddings from all PDFs
all_paragraphs = []
all_embeddings = []

# Loop over PDF files to parse text, generate embeddings, and store data
for filename in pdf_filenames:
    paragraphs = parse_pdf(filename)
    all_paragraphs.extend(paragraphs)
    embeddings = get_embeddings(filename, "mxbai-embed-large", paragraphs)
    all_embeddings.extend(embeddings)

# Endpoint to handle question and generate response
@app.post("/answer/")
async def main(question: Question):
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
        based on snippets of text provided in context. Answer only using the context provided,
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """
    prompt = question.question.strip()
    embeddings_folder = question.embeddings_folder_path
    all_embeddings = load_embeddings(embeddings_folder)
    if all_embeddings is False:
        return {"answer": "No embeddings found in the specified folder."}
    
    prompt_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=prompt)["embedding"]
    most_similar_chunks = find_most_similar(prompt_embedding, all_embeddings)[:5]
    response = ollama.chat(
        model="dolphin-phi",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n".join(all_paragraphs[item[1]] for item in most_similar_chunks),
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
import fitz  # PyMuPDF
import ollama
import json
import time
import os
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import requests
import os


app = FastAPI()

###############################

def download_pdf(url, save_path):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open a file in binary write mode and write the content of the response
        with open(save_path, 'wb') as pdf_file:
            pdf_file.write(response.content)
        print("PDF downloaded successfully!")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")

# Example usage
pdf_url = "https://freekidsbooks.org/wp-content/uploads/2019/12/GreenPixie2021Color-FKB.pdf"
local_folder = "/home/bharat/codebase/text_searching"
pdf_file_name = "KB.pdf"
pdf_save_path = os.path.join(local_folder, pdf_file_name)

download_pdf(pdf_url, pdf_save_path)

#######################################

def parse_pdf(filename):
    paragraphs = []
    doc = fitz.open(filename)
    for page in doc:
        text = page.get_text().strip()
        if text:
            paragraphs.append(text)
    return paragraphs

def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)

def get_embeddings(filename, modelname, chunks):
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    embeddings = [ollama.embeddings(model=modelname, prompt=chunk)["embedding"] for chunk in chunks]
    save_embeddings(filename, embeddings)
    return embeddings

def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

class Question(BaseModel):
    question: str

@app.post("/answer/")
async def main(question: Question,):
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """
    filename = pdf_file_name
    paragraphs = parse_pdf(filename)
    start = time.perf_counter()
    embeddings = get_embeddings(filename, "nomic-embed-text", paragraphs)
    
    # with open("question.txt", "r") as file:
    #     question_text = file.read().strip()

    prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=question.question)["embedding"]
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n".join(paragraphs[item[1]] for item in most_similar_chunks),
            },
            {"role": "user", "content": question.question},
        ],
    )
    print("\n\n")
    print(start)
    print(response["message"]["content"])
    return {"answer": response["message"]["content"]}

if __name__ == "__main__":
    main()
# import fitz  # PyMuPDF
# import ollama
# import json
# import time
# import os
# import numpy as np
# from numpy.linalg import norm
# from fastapi import FastAPI, File, UploadFile
# from pydantic import BaseModel
# import requests
# import os

# app = FastAPI()

# def parse_pdf(filename):
#     paragraphs = []
#     doc = fitz.open(filename)
#     for page in doc:
#         text = page.get_text().strip()
#         if text:
#             paragraphs.append(text)
#     return paragraphs

# def save_embeddings(filename, embeddings):
#     if not os.path.exists("embeddings"):
#         os.makedirs("embeddings")
#     with open(f"embeddings/{filename}.json", "w") as f:
#         json.dump(embeddings, f)

# def load_embeddings(filename):
#     if not os.path.exists(f"embeddings/{filename}.json"):
#         return False
#     with open(f"embeddings/{filename}.json", "r") as f:
#         return json.load(f)

# def get_embeddings(filename, modelname, chunks):
#     if (embeddings := load_embeddings(filename)) is not False:
#         return embeddings
#     embeddings = [ollama.embeddings(model=modelname, prompt=chunk)["embedding"] for chunk in chunks]
#     save_embeddings(filename, embeddings)
#     return embeddings

# def find_most_similar(needle, haystack):
#     needle_norm = norm(needle)
#     similarity_scores = [np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack]
#     return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

# class Question(BaseModel):
#     question: str

# @app.post("/answer/")
# async def main(question: Question,):
#     SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
#         based on snippets of text provided in context. Answer only using the context provided, 
#         being as concise as possible. If you're unsure, just say that you don't know.
#         Context:
#     """
#     filename = "ncert2.pdf"
#     paragraphs = parse_pdf(filename)
#     start = time.perf_counter()
#     embeddings = get_embeddings(filename, "nomic-embed-text", paragraphs)
    
#     # with open("question.txt", "r") as file:
#     #     question_text = file.read().strip()

#     prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=question.question)["embedding"]
#     most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

#     response = ollama.chat(
#         model="mistral",
#         messages=[
#             {
#                 "role": "system",
#                 "content": SYSTEM_PROMPT
#                 + "\n".join(paragraphs[item[1]] for item in most_similar_chunks),
#             },
#             {"role": "user", "content": question.question},
#         ],
#     )
#     print("\n\n")
#     print(start)
#     print(response["message"]["content"])
#     return {"answer": response["message"]["content"]}

# if __name__ == "__main__":
#     main()

import fitz  # PyMuPDF
import ollama
import json
import time
import os
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()

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
async def main(question: Question):
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """
    pdf_filenames = ["ncert2.pdf", "ncert3.pdf"]  # Add your PDF filenames here
    all_paragraphs = []
    all_embeddings = []

    for filename in pdf_filenames:
        paragraphs = parse_pdf(filename)
        all_paragraphs.extend(paragraphs)
        embeddings = get_embeddings(filename, "nomic-embed-text", paragraphs)
        all_embeddings.extend(embeddings)

    prompt = question.question.strip()
    prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]
    most_similar_chunks = find_most_similar(prompt_embedding, all_embeddings)[:5]

    response = ollama.chat(
        model="mistral",
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

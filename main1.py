# import fitz  # PyMuPDF
# import ollama
# import json
# import os
# import numpy as np
# from numpy.linalg import norm
# from fastapi import FastAPI, File, UploadFile
# from pydantic import BaseModel

# app = FastAPI()

# src_dir = "/home/bharat/codebase/text_searching/kb"
# target_dir="/home/bharat/codebase/text_searching/"
# document_name = "technical/internet.pdf"

# def create_paragraphs(src_dir, document_name):
#     # Break down the PDF into smaller text paragraphs and return an array of strings
#     para = []
#     pdf_path = f"{src_dir}/{document_name}"
#     doc = fitz.open(pdf_path)
#     for page in doc:
#         text = page.get_text().strip()
#         if text:
#             para.append(text)
#     return para

# def create_embeddings(src_txt_array, model_name):
#     # return embeddings from the array of strings (aka chunks)
#     embedding_location = f"embeddings/{src_txt_array}"
#     if (src_embeddings := get_embedding(embedding_location)) is not False:
#         return src_embeddings
#     src_embeddings = ollama.embeddings(model=model_name, prompt=src_txt_array)["embeddings"]
#     save_embeddings(target_dir,document_name,src_embeddings)
#     return src_embeddings    

# def save_embeddings(target_dir, document_name, src_embeddings):
#     # create a DIR and save the embeddings with the given document_name
#     filename_without_ext = os.path.splitext(document_name)[0]
#     embedding_location = os.path.join(target_dir, f"vectors/{filename_without_ext}")
#     if not os.path.exists(embedding_location):
#         os.makedirs(embedding_location)
#     with open(os.path.join(embedding_location, "embeddings.json"), "w") as f:
#         json.dump(src_embeddings, f)


# def get_embedding(embedding_location):
#     # return the embeddings available at the embedding_location
#     if not os.path.exists(embedding_location):
#         return False
#     with open(f"{embedding_location}/embeddings.json", "r") as f:
#         return json.load(f)

# def find_most_similar(needle, haystack):
#     # use math tools to perform match
#     needle_norm = norm(needle)
#     similarity_scores = [np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack]
#     return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

# def generate_chat_response(most_similar_chunks, model_name_for_chat):
#     # Use the model to generate a human-friendly response
#     similar_content = "\n".join(str(item[0]) for item in most_similar_chunks)
#     response = f"Using model '{model_name_for_chat}', here are some relevant snippets:\n{similar_content}"    
#     return response


# def generate_embedding(src_dir, document_name, traget_dir, model_name):
#     # 
#     src_txt_array = create_paragraphs(src_dir, document_name)

#     src_embeddings = create_embeddings(traget_dir, src_txt_array, model_name)

#     save_embeddings(traget_dir, document_name, src_embeddings)

# # Define a Pydantic model for the question payload
# class QuestionBody(BaseModel):
#     question: str # 'how many novels did abc write?'
#     embedding_location: str # dir/biolog-class12.json
#     model_name_for_embeddings: str # mxbai-embed-large
#     model_name_for_chat: str # mxbai-embed-large
#     best_match_count: int # top best matching

#     class Config:
#         # Set protected_namespaces to an empty tuple to resolve the conflict
#         protected_namespaces = ()

# @app.post("/answer/")
# async def handle_question(qPayload: QuestionBody):
#     file_embeddings = get_embedding(qPayload.embedding_location)
#     prompt_embedding = ollama.embeddings(model=qPayload.model_name_for_embeddings, prompt=qPayload.question)["embedding"]
#     most_similar_chunks = find_most_similar(prompt_embedding, file_embeddings)[:qPayload.best_match_count]
#     final_response = generate_chat_response(most_similar_chunks, qPayload.model_name_for_chat)
#     # ollama.chat()
#     return {"answer": final_response}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)   

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
document_name = "technical/internet.pdf"

def create_paragraphs(src_dir, document_name):
    # Break down the PDF into smaller text paragraphs and return an array of strings
    para = []
    pdf_path = f"{src_dir}/{document_name}"
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text().strip()
        if text:
            para.append(text)
    return para

def create_embeddings(src_txt_array, model_name, target_dir, document_name):
    embedding_location = os.path.join(target_dir, f"embeddings/{document_name}")
    src_embeddings = get_embedding(embedding_location)
    if src_embeddings:
        return src_embeddings
    src_txt_string = "\n".join(src_txt_array)  # Join paragraphs into a single string
    src_embeddings = ollama.embeddings(model=model_name, prompt=src_txt_string)
    save_embeddings(target_dir, document_name, src_embeddings)
    return src_embeddings

def save_embeddings(target_dir, document_name, src_embeddings):
    filename_without_ext = os.path.splitext(document_name)[0]
    embedding_location = os.path.join(target_dir, f"vectors/{filename_without_ext}")
    if not os.path.exists(embedding_location):
        os.makedirs(embedding_location)
    with open(os.path.join(embedding_location, "embeddings.json"), "w") as f:
        json.dump(src_embeddings, f)


def get_embedding(embedding_location):
    # return the embeddings available at the embedding_location
    if not os.path.exists(embedding_location):
        return False  # Return an empty list if embeddings file does not exist
    with open(f"{embedding_location}/embeddings.json", "r") as f:
        return json.load(f)

def find_most_similar(needle, haystack):
    # use math tools to perform match
    needle_norm = norm(needle)
    similarity_scores = [np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def generate_chat_response(most_similar_chunks, model_name_for_chat):
    # Use the model to generate a human-friendly response
    similar_content = "\n".join(str(item[0]) for item in most_similar_chunks)
    response = f"Using model '{model_name_for_chat}', here are some relevant snippets:\n{similar_content}"    
    return response

def generate_embedding(src_dir, document_name, target_dir, model_name):
    # Generate embeddings and save them
    src_txt_array = create_paragraphs(src_dir, document_name)
    src_embeddings = create_embeddings(src_txt_array, model_name, target_dir, document_name)
    save_embeddings(target_dir, document_name, src_embeddings)

# Define a Pydantic model for the question payload
class QuestionBody(BaseModel):
    question: str # 'how many novels did abc write?'
    embedding_location: str # dir/biolog-class12.json
    model_name_for_embeddings: str # mxbai-embed-large
    model_name_for_chat: str # mxbai-embed-large
    best_match_count: int # top best matching

    class Config:
        # Set protected_namespaces to an empty tuple to resolve the conflict
        protected_namespaces = ()

# Generate embeddings before running the FastAPI server
generate_embedding(src_dir, document_name, target_dir, "mxbai-embed-large")

@app.post("/answer/")
async def handle_question(qPayload: QuestionBody):
    prompt = qPayload.question.strip()
    embedding_folder=qPayload.embedding_location
    print(embedding_folder)
    file_embeddings = get_embedding(embedding_folder)
    print(file_embeddings)
    print("Hello")
    prompt_embedding = ollama.embeddings(model=qPayload.model_name_for_embeddings, prompt=prompt)["embedding"]
    print(prompt_embedding)
    most_similar_chunks = find_most_similar(prompt_embedding, file_embeddings)[:qPayload.best_match_count]
    print(most_similar_chunks)
    final_response = generate_chat_response(most_similar_chunks, qPayload.model_name_for_chat)
    # ollama.chat()
    print(final_response)
    return {"answer": final_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



import fitz  # PyMuPDF
import ollama
import json
import os
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()

def create_paragraphs(src_dir, document_name):
    # break dwon the pdf into smaller text para and return "array of strings"

def create_embeddings(traget_dir, src_txt_array, model_name):
    # return embeddings from the array of strings (aka chunks)

def save_embeddings(traget_dir, document_name, src_embeddings):
    # create a DIR and save the embeddings with the given document_name

def get_embedding(embedding_location):
    # return the embeddings available at the embedding_location

def find_most_similar(prompt_embedding, kb_embedding):
    # use math tools to perform match

def generate_chat_response(most_similar_chunks, model_name_for_chat):
    # use the model to generate human friendly response


# src_dir -> directory path where the KB is located in local file system
# document_name -> name of the KB file
# traget_dir -> directory path where the embedding will be stored
# model_name -> name of the model to convert document into embeddings
# this is the starting point
def generate_embedding(src_dir, document_name, traget_dir, model_name):
    # 
    src_txt_array = create_paragraphs(src_dir, document_name)

    src_embeddings = create_embeddings(traget_dir, src_txt_array, model_name)

    save_embeddings(traget_dir, document_name, src_embeddings)

# Define a Pydantic model for the question payload
class QuestionBody(BaseModel):
    question: str # 'how many novels did abc write?'
    embedding_location: str # dir/biolog-class12.json
    model_name_for_embeddings: str # mxbai-embed-large
    model_name_for_chat: str # mxbai-embed-large
    best_match_count: int # top best matching

@app.post("/answer/")
async def handle_question(qPayload: QuestionBody):
    file_embeddings = get_embedding(qPayload.embedding_location)
    prompt_embedding = ollama.embeddings(model=qPayload.model_name, prompt=qPayload.question)["embedding"]
    most_similar_chunks = find_most_similar(prompt_embedding, file_embeddings)[:qPayload.best_match_count]
    final_response = generate_chat_response(most_similar_chunks, qPayload.model_name_for_chat)
    # ollama.chat()

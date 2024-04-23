# import torch
# from transformers import BertTokenizer, BertModel

# # Load BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")

# # Read the text file
# with open("Text.txt") as f:
#     text_file = f.read()

# # Split the text into chunks (you can adjust the chunk size)
# def split_text_into_chunks(text, chunk_size=500):
#     chunks = []
#     for i in range(0, len(text), chunk_size):
#         chunks.append(text[i : i + chunk_size])
#     return chunks

# # Example user query
# user_query = "when was angular 8 released"

# # Process each chunk
# text_chunks = split_text_into_chunks(text_file)
# query_encoding = tokenizer(user_query, return_tensors="pt")

# # Calculate similarity and retrieve relevant chunks
# similar_chunks = []
# for chunk in text_chunks:
#     chunk_encoding = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**chunk_encoding)
#     chunk_embedding = outputs.last_hidden_state.mean(dim=1)  # Calculate mean of token embeddings
#     query_embedding = model(**query_encoding).last_hidden_state.mean(dim=1)
#     similarity = torch.cosine_similarity(query_embedding, chunk_embedding).item()
#     similar_chunks.append((chunk, similarity))

# # Sort chunks by similarity (highest first)
# similar_chunks.sort(key=lambda x: x[1], reverse=True)

# # Combine relevant chunks for context
# context = " ".join(chunk[0] for chunk in similar_chunks[:3])  # Combine top 3 chunks

# # Now use the context + query for LLM input (e.g., GPT-3)
# llm_input = f"{context} {user_query}"
# # Pass llm_input to your LLM and retrieve the answer

# print("Context:")
# # print(context)
# print("\nUser Query:")
# print(user_query)
# print("\nLLM Input:")
# print(llm_input)


#################################################################
# with open("Text.txt", "r") as file:
#     text_content = file.read()

# from langchain import prompts,llms,hub

# # Define a prompt template for question-answering
# template = """Question: {question} Answer: Let's think step by step."""
# prompt = prompts(template=template, input_variables=["question"])

# # Initialize the LangChain with the prompt template
# llm_chain = llms(prompt=prompt)

# # Load a pre-trained Hugging Face model (e.g., GPT-2)
# hf_model_id = "gpt2"  # You can choose a different model if needed
# hf_pipeline = hub.from_model_id(model_id=hf_model_id, task="text-generation")


# user_question = "What is the Angular?"
# answer = llm_chain.generate_response(user_question, context=text_content)

# print(f"User Question: {user_question}")
# print(f"Answer: {answer}")


######################################################################


# import ollama
# import time
# import os
# import json
# import numpy as np
# from numpy.linalg import norm


# # open a file and return paragraphs
# def parse_file(filename):
#     with open(filename, encoding="utf-8-sig") as f:
#         paragraphs = []
#         buffer = []
#         for line in f.readlines():
#             line = line.strip()
#             if line:
#                 buffer.append(line)
#             elif len(buffer):
#                 paragraphs.append((" ").join(buffer))
#                 buffer = []
#         if len(buffer):
#             paragraphs.append((" ").join(buffer))
#         return paragraphs


# def save_embeddings(filename, embeddings):
#     # create dir if it doesn't exist
#     if not os.path.exists("embeddings"):
#         os.makedirs("embeddings")
#     # dump embeddings to json
#     with open(f"embeddings/{filename}.json", "w") as f:
#         json.dump(embeddings, f)


# def load_embeddings(filename):
#     # check if file exists
#     if not os.path.exists(f"embeddings/{filename}.json"):
#         return False
#     # load embeddings from json
#     with open(f"embeddings/{filename}.json", "r") as f:
#         return json.load(f)


# def get_embeddings(filename, modelname, chunks):
#     # check if embeddings are already saved
#     if (embeddings := load_embeddings(filename)) is not False:
#         return embeddings
#     # get embeddings from ollama
#     embeddings = [
#         ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
#         for chunk in chunks
#     ]
#     # save embeddings
#     save_embeddings(filename, embeddings)
#     return embeddings


# # find cosine similarity of every chunk to a given embedding
# def find_most_similar(needle, haystack):
#     needle_norm = norm(needle)
#     similarity_scores = [
#         np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
#     ]
#     return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)


# def main():
#     SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
#         based on snippets of text provided in context. Answer only using the context provided, 
#         being as concise as possible. If you're unsure, just say that you don't know.
#         Context:
#     """
#     # ollama.pull("nomic-embed-text")
#     # ollama.pull("mistral")
#     # open file
#     filename = "Text.txt"
#     paragraphs = parse_file(filename)

#     embeddings = get_embeddings(filename, "nomic-embed-text", paragraphs)

#     prompt = input("what do you want to know? -> ")
#     # strongly recommended that all embeddings are generated by the same model (don't mix and match)
#     prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]
#     # find most similar to each other
#     most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

#     response = ollama.chat(
#         model="mistral",
#         messages=[
#             {
#                 "role": "system",
#                 "content": SYSTEM_PROMPT
#                 + "\n".join(paragraphs[item[1]] for item in most_similar_chunks),
#             },
#             {"role": "user", "content": prompt},
#         ],
#     )
#     print("\n\n")
#     print(response["message"]["content"])


# if __name__ == "__main__":
#     main()

###################################################################################################

# import fitz  # PyMuPDF
# import ollama
# import json
# import time
# import os
# import numpy as np
# from numpy.linalg import norm
# from fastapi import FastAPI, File, UploadFile
# from pydantic import BaseModel

# app = FastAPI()

# def parse_pdf(filename):
#     paragraphs = []
#     doc = fitz.open(filename)
#     for page in doc:
#         text = page.get_text().strip()
#         if text:
#             paragraphs.append(text)
#     return paragraphs

# def save_embeddings(pdf_filename, embeddings):
#     filename_without_ext = os.path.splitext(pdf_filename)[0]
#     embeddings_folder = f"embeddings/{filename_without_ext}"
#     if not os.path.exists(embeddings_folder):
#         os.makedirs(embeddings_folder)
#     with open(f"{embeddings_folder}/embeddings.json", "w") as f:
#         json.dump(embeddings, f)

# def load_embeddings(pdf_filename):
#     filename_without_ext = os.path.splitext(pdf_filename)[0]
#     embeddings_folder = f"embeddings/{filename_without_ext}"
#     if not os.path.exists(embeddings_folder):
#         return False
#     with open(f"{embeddings_folder}/embeddings.json", "r") as f:
#         return json.load(f)

# def get_embeddings(pdf_filename, modelname, chunks):
#     if (embeddings := load_embeddings(pdf_filename)) is not False:
#         return embeddings
#     embeddings = [ollama.embeddings(model=modelname, prompt=chunk)["embedding"] for chunk in chunks]
#     save_embeddings(pdf_filename, embeddings)
#     return embeddings

# def find_most_similar(needle, haystack):
#     needle_norm = norm(needle)
#     similarity_scores = [np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack]
#     return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

# class Question(BaseModel):
#     question: str

# @app.post("/answer/")
# async def main(question: Question):
#     SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
#         based on snippets of text provided in context. Answer only using the context provided, 
#         being as concise as possible. If you're unsure, just say that you don't know.
#         Context:
#     """
#     pdf_filenames = ["Text.pdf","ncert2.pdf","ncert3.pdf"]  # Add your PDF filenames here
#     all_paragraphs = []
#     all_embeddings = []

#     for filename in pdf_filenames:
#         paragraphs = parse_pdf(filename)
#         all_paragraphs.extend(paragraphs)
#         embeddings = get_embeddings(filename, "nomic-embed-text", paragraphs)
#         all_embeddings.extend(embeddings)

#     prompt = question.question.strip()
#     prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]
#     most_similar_chunks = find_most_similar(prompt_embedding, all_embeddings)[:5]

#     response = ollama.chat(
#         model="mistral",
#         messages=[
#             {
#                 "role": "system",
#                 "content": SYSTEM_PROMPT
#                 + "\n".join(all_paragraphs[item[1]] for item in most_similar_chunks),
#             },
#             {"role": "user", "content": prompt},
#         ],
#     )
#     print(response["message"])
#     return {"answer": response["message"]}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


###############################################################################################################


import fitz  # PyMuPDF
import ollama
import json
import os
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, Query
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

def save_embeddings(embeddings, folder_path):
    with open(folder_path, "w") as f:
        json.dump(embeddings, f)

def load_embeddings(folder_path):
    embeddings_file = os.path.join(folder_path, "embeddings.json")
    if not os.path.exists(embeddings_file):
        return False
    with open(embeddings_file, "r") as f:
        return json.load(f)

def get_embeddings(pdf_filenames, modelname, chunks, folder_path):
    all_embeddings = []
    for pdf_filename in pdf_filenames:
        pdf_folder_path = os.path.join(folder_path, pdf_filename)
        if not os.path.exists(pdf_folder_path):
            os.makedirs(pdf_folder_path)
        
        embeddings = load_embeddings(pdf_folder_path)
        if not embeddings:
            embeddings = [ollama.embeddings(model=modelname, prompt=chunk)["embedding"] for chunk in chunks]
            save_embeddings(embeddings, os.path.join(pdf_folder_path, "embeddings.json"))
        all_embeddings.extend(embeddings)
    return all_embeddings

def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

class Question(BaseModel):
    question: str
    folder_path: str

@app.post("/answer/")
async def main(question: Question):
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """
    pdf_filenames = ["Text.pdf", "ncert2.pdf", "ncert3.pdf", "bio.pdf"] 
    all_paragraphs = []

    for filename in pdf_filenames:
        paragraphs = parse_pdf(filename)
        all_paragraphs.extend(paragraphs)
    
    embeddings = get_embeddings(pdf_filenames, "nomic-embed-text", all_paragraphs, question.folder_path)

    prompt = question.question.strip()
    prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

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






# import fitz  # PyMuPDF
# import ollama
# import json
# import os
# import numpy as np
# from numpy.linalg import norm
# from fastapi import FastAPI, File, UploadFile
# from pydantic import BaseModel

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
#     pdf_filename: str # Added input for PDF filename

# pdf_filenames = ["ncert2.pdf", "Text.pdf"]  # Original list of PDF filenames

# @app.post("/answer/")
# async def main(question: Question, pdf_file: UploadFile = File(...), pdf_filenames: list = pdf_filenames):
#     SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
#         based on snippets of text provided in context. Answer only using the context provided, 
#         being as concise as possible. If you're unsure, just say that you don't know.
#         Context:
#     """
#     specified_pdf_filename = question.pdf_filename
#     if specified_pdf_filename not in pdf_filenames:
#         return {"error": f"Specified PDF file '{specified_pdf_filename}' not found in the list"}

#     if pdf_file.filename.endswith(".pdf"):
#         with open(pdf_file.filename, "wb") as f:
#             f.write(pdf_file.file.read())

#         paragraphs = parse_pdf(pdf_file.filename)
#         embeddings = get_embeddings(pdf_file.filename, "nomic-embed-text", paragraphs)

#         prompt = question.question.strip()
#         prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]
#         most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

#         response = ollama.chat(
#             model="mistral",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": SYSTEM_PROMPT
#                     + "\n".join(paragraphs[item[1]] for item in most_similar_chunks),
#                 },
#                 {"role": "user", "content": prompt},
#             ],
#         )

#         os.remove(pdf_file.filename)  # Remove the uploaded PDF file after processing

#         return {"answer": response["message"]}
#     else:
#         return {"error": "Only PDF files are allowed"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



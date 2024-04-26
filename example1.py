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

# def save_embeddings(pdf_filename, embeddings):
#     filename_without_ext = os.path.splitext(pdf_filename)[0]
#     embeddings_folder = f"vectors/{filename_without_ext}"
#     if not os.path.exists(embeddings_folder):
#         os.makedirs(embeddings_folder)
#     with open(f"{embeddings_folder}/embeddings.json", "w") as f:
#         json.dump(embeddings, f)

# def load_embeddings(embeddings_folder):
#     if not os.path.exists(embeddings_folder):
#         return False
#     with open(f"{embeddings_folder}/embeddings.json", "r") as f:
#         return json.load(f)
    
# def get_embeddings(pdf_filename, modelname, chunks):
#     embeddings_folder = f"embeddings/{pdf_filename}"
#     if (embeddings := load_embeddings(embeddings_folder)) is not False:
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
#     embeddings_folder_path: str

# pdf_filenames = ["kb/literature/english-12.pdf","kb/literature/bio.pdf", "kb/technical/internet.pdf"]  

# all_paragraphs = []
# all_embeddings = []
# for filename in pdf_filenames:
#     paragraphs = parse_pdf(filename)
#     all_paragraphs.extend(paragraphs)
#     embeddings = get_embeddings(filename, "mxbai-embed-large", paragraphs)
#     all_embeddings.extend(embeddings)


# @app.post("/answer/")
# async def main(question: Question):
#     SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
#         based on snippets of text provided in context. Answer only using the context provided,
#         being as concise as possible. If you're unsure, just say that you don't know.
#         Context:
#     """
#     prompt = question.question.strip()
#     embeddings_folder = question.embeddings_folder_path
#     all_embeddings = load_embeddings(embeddings_folder)
#     if all_embeddings is False:
#         return {"answer": "No embeddings found in the specified folder."}
    
#     prompt_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=prompt)["embedding"]
#     most_similar_chunks = find_most_similar(prompt_embedding, all_embeddings)[:5]
#     response = ollama.chat(
#         model="dolphin-phi",
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

# import fitz

# def parse_pdf():
#     para=[]
#     doc = fitz.open("kb/technical/internet.pdf")
#     for page in doc:
#         text = page.get_text().strip()
#         if text:
#             para.append(text)
#     # return para
#     print(para)


# import fitz  # PyMuPDF

# def process_pdf(src_directory, pdf_name):
#     # Construct the full path to the PDF file
#     pdf_path = f"{src_directory}/{pdf_name}"

#     try:
#         # Open the PDF file
#         doc = fitz.open(pdf_path)

#         # Your processing logic here...
#         # For example, you can iterate through pages and extract text:
#         for page_num in range(len(doc)):
#             page = doc.load_page(page_num)
#             text = page.get_text()
#             print(f"Page {page_num + 1}:\n{text}\n")

#         # Close the PDF file
#         doc.close()
#     except Exception as e:
#         print(f"Error processing PDF: {e}")

# # Example usage:
# src_directory = "/home/bharat/codebase/text_searching/kb"
# pdf_name = "technical/internet.pdf"
# process_pdf(src_directory, pdf_name)

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
#     embedding_location = f"embeddings/{document_name}.json"
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
#     # use the model to generate human friendly response
#     similar_content = "\n".join(item[0] for item in most_similar_chunks)
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
document_name = "literature/political.pdf"

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
    src_embeddings = [ollama.embeddings(model=model_name, prompt=src_txt_string)["embedding"]]
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

# def generate_chat_response(most_similar_chunks, model_name_for_chat):
#     # Use the model to generate a human-friendly response
#     similar_content = "\n".join(str(item[0]) for item in most_similar_chunks)
#     response = f"Using model '{model_name_for_chat}', here are some relevant snippets:\n{similar_content}"    
#     return response

# def generate_response(system_prompt, most_similar_chunks, prompt):
#     response = ollama.chat(
        
#         messages=[
#             {
#                 "role": "system",
#                 "content": system_prompt + "\n".join(str(item[0]) for item in most_similar_chunks),
#             },
#             {"role": "user", "content": prompt},
#         ],
#     )
#     return response["message"]


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
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
        based on snippets of text provided in context. Answer only using the context provided,
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """
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

    # final_response = generate_chat_response(most_similar_chunks, qPayload.model_name_for_chat)
    # ollama.chat()
    # print(final_response)
    # return {"answer": final_response}

    # response_message = generate_response(
    #     model=qPayload.model_name_for_chat,
    #     system_prompt=SYSTEM_PROMPT,
    #     most_similar_chunks=most_similar_chunks,
    #     prompt=prompt
    # )
    # print(response_message)
    # return {"answer": response_message}

    response = ollama.chat(
        model=qPayload.model_name_for_chat,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n".join(str(item[0]) for item in most_similar_chunks),
            },
            {"role": "user", "content": prompt},
        ],
    )

    print(response["message"])
    return {"answer": response["message"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


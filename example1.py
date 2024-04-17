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


# from langchain_community.document_loaders import PyPDFLoader
# loader = PyPDFLoader("internet.pdf")
# docs= loader.load()

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
# text_splitter.split_documents(docs)
# documents= text_splitter.split_documents(docs)

# from langchain_community.embeddings import Word2VecEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS

# word2vec_model = Word2Vec.load("/home/bharat/Downloads/numberbatch.txt.gz")

# db = FAISS.from_documents(documents[:30], Word2VecEmbeddings(word2vec_model))

# print(db)


# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from gensim.models import Word2Vec

# # Load documents
# loader = PyPDFLoader("internet.pdf")
# docs = loader.load()

# # Split documents
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
# text_splitter.split_documents(docs)
# documents = text_splitter.split_documents(docs)

# # Load pre-trained Word2Vec model
# word2vec_model_path = "numberbatch-en.txt"
# word2vec_model = Word2Vec.load_word2vec_format(word2vec_model_path, binary=False)  # Load Word2Vec model

# # Create FAISS index
# db = FAISS.from_documents(documents[:30], word2vec_model.wv)  # Pass word vectors directly
# print(db)


# from sentence_transformers import SentenceTransformer

# # Load a pre-trained model (e.g., 'multi-qa-MiniLM-L6-cos-v1')
# model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# # Encode a sentence to get its embedding
# sentence = "How big is London"
# query_embedding = model.encode(sentence)

# # You can now use this embedding for semantic search or other tasks
# print(f"Embedding for '{sentence}': {query_embedding}")


# from sentence_transformers import SentenceTransformer
# sentences = ["This is an example sentence", "Each sentence is converted"]

# model = SentenceTransformer('sentence-transformers/gtr-t5-xxl')
# embeddings = model.encode(sentences)
# print(embeddings)

def divide_text_into_chunks(file_path, chunk_size):
    with open(file_path, 'r') as file:
        text = file.read()
    
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    
    return chunks

# Example usage:
file_path = 'Text.txt'
chunk_size = 1000  # Adjust the chunk size as per your requirement
chunks = divide_text_into_chunks(file_path, chunk_size)
print(chunks)
# Print or do something else with the chunks
# for i, chunk in enumerate(chunks):
#     print(chunk)
#     print("\n")

# from sentence_transformers import SentenceTransformer
# sentences = chunks

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)
# print(embeddings)

import chromadb
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.Client()

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
collection = client.create_collection("all-my-documents")

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=chunks, # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
    metadatas=[{"source": "notion"}], # filter on these!
    ids=["doc1"], # unique for each doc
)

# Query/search 2 most similar results. You can also .get by id
results = collection.query(
    query_texts=["When was Angular 7 released"],
    n_results=1,
    # include=['distances','metadatas','embeddings','documents']
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)
print(results)
# from sentence_transformers import SentenceTransformer
# sentences = chunks

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)
# print(embeddings)

# from langchain_community.vectorstores import FAISS
# from sentence_transformers import SentenceTransformer
# from pdfminer.high_level import extract_text

# # Extract text from PDF
# pdf_path = "internet.pdf"
# text = extract_text(pdf_path)

# # Initialize SentenceTransformer model
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Encode the entire text
# embedding = model.encode([text])[0]  # Extracting the embedding from the NumPy array

# # Create FAISS vectorstore
# db = FAISS.from_documents([embedding.tolist()], model)  # Convert embedding to list

# # Print the vectorstore (optional)
# print(db)





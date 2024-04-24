def load_embeddings(embeddings_folder):
    if not os.path.exists(embeddings_folder):
        return False
    with open(f"{embeddings_folder}/embeddings.json", "r") as f:
        return json.load(f)
    

    def save_embeddings(pdf_filename, embeddings):
    filename_without_ext = os.path.splitext(pdf_filename)[0]
    embeddings_folder = f"vectors/{filename_without_ext}"
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)
    with open(f"{embeddings_folder}/embeddings.json", "w") as f:
        json.dump(embeddings, f)
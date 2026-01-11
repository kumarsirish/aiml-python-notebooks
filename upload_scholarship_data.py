import os
from datasets import load_dataset, Features, Value, Pdf

# 1. Configuration
# Ensure this folder contains BOTH your PDFs and your metadata.csv
DATA_PATH = "/mnt/c/Users/sirkumar/CMRIT/RAG-WORKSHOP/data"  
REPO_ID = "NetraVerse/indian-govt-scholarships"  

# 2. Define Features
# This is crucial: it forces the 'file_name' to be treated as a PDF object
features = Features({
    "file_name": Pdf(),    
    "label": Value("string"),
    "text": Value("string")
})

def upload_to_hf():
    print("🚀 Loading local dataset with pdffolder...")
    
    # We pass the 'features' here so the local validation 
    # and the remote schema are locked to PDF mode.
    dataset = load_dataset(
        "pdffolder", 
        data_dir=DATA_PATH, 
        features=features
    )

    # 3. Push to Hugging Face Hub
    print(f"📤 Uploading to https://huggingface.co/datasets/{REPO_ID}...")
    
    # dataset.push_to_hub automatically handles the creation of the repo
    dataset.push_to_hub(REPO_ID, private=True)
    
    print("\n✅ Upload Complete!")
    print(f"Check your dataset here: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    upload_to_hf()
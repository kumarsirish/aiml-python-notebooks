"""
Upload scholarship PDFs to HuggingFace Dataset

"""
import os
from datasets import Dataset, Features, Value
from huggingface_hub import HfApi
import pandas as pd
import PyPDF2  # For PDF text extraction

# Configuration
DATA_PATH = "/mnt/c/Users/sirkumar/CMRIT/RAG-WORKSHOP/data"
REPO_ID = "NetraVerse/indian-govt-scholarships"

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyPDF2
    """
    try:
        # Option 1: PyPDF2 (faster, but less accurate)
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            return ' '.join(text).strip()
        
        # Option 2: pdfplumber (better quality, requires: pip install pdfplumber)
        # import pdfplumber
        # with pdfplumber.open(pdf_path) as pdf:
        #     text = []
        #     for page in pdf.pages:
        #         text.append(page.extract_text())
        #     return ' '.join(text).strip()
        
    except Exception as e:
        print(f"⚠️ Error extracting text from {pdf_path}: {e}")
        return f"Error extracting text: {str(e)}"

def create_metadata_csv():
    """Create metadata.csv from PDFs"""
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    
    metadata = []
    for pdf_file in pdf_files:
        print(f"📄 Processing {pdf_file}...")
        pdf_path = os.path.join(DATA_PATH, pdf_file)
        
        # Extract actual text from PDF
        extracted_text = extract_text_from_pdf(pdf_path)
        
        metadata.append({
            'file_name': pdf_file,
            'label': pdf_file.replace('.pdf', ''),  # Use filename as label
            'text': extracted_text  # Actual extracted text
        })
        
        print(f"   ✅ Extracted {len(extracted_text)} characters")
    
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(DATA_PATH, 'metadata.csv'), index=False)
    return df

def upload_to_hf():
    print("🚀 Creating metadata and uploading to HF...")
    
    # Create metadata CSV
    df = create_metadata_csv()
    
    # Create dataset from pandas DataFrame
    dataset = Dataset.from_pandas(df)
    
    # Upload dataset
    print(f"📤 Uploading to https://huggingface.co/datasets/{REPO_ID}...")
    dataset.push_to_hub(REPO_ID, private=True)
    
    # Upload PDF files separately
    api = HfApi()
    for pdf_file in os.listdir(DATA_PATH):
        if pdf_file.endswith('.pdf'):
            api.upload_file(
                path_or_fileobj=os.path.join(DATA_PATH, pdf_file),
                path_in_repo=f"pdfs/{pdf_file}",
                repo_id=REPO_ID,
                repo_type="dataset"
            )
    
    print("✅ Upload Complete!")

if __name__ == "__main__":
    upload_to_hf()

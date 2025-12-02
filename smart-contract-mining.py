import os
import json
import logging
import re
from typing import List, Dict, Optional
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import docx
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, pipeline
from cryptography.fernet import Fernet
import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ENCRYPTION_KEY = os.getenv("CONTRACT_MINER_ENCRYPTION_KEY") or Fernet.generate_key()
cipher_suite = Fernet(ENCRYPTION_KEY)

def encrypt_data(data: str) -> bytes:
    if data:
        return cipher_suite.encrypt(data.encode())
    return b""

def decrypt_data(data: bytes) -> str:
    if data:
        return cipher_suite.decrypt(data).decode()
    return ""

def mask_sensitive_data(data: str) -> str:
    if data and len(data) > 2:
        return data[0] + "*" * (len(data) - 2) + data[-1]
    return data

class SmartContractMiner:
    def __init__(self,
                 ner_model_name: str = "allenai/longformer-base-4096",
                 max_chunk_tokens: int = 4000,  # For Longformer
                 cache_dir: str = "cache_responses",
                 chroma_persist_dir: str = "chroma_persist",
                 input_filter_pattern: Optional[str] = None,
                 output_filter_fields: Optional[List[str]] = None,
                 sensitive_terms_file: Optional[str] = None):
        logging.info("Initialized Smart Contract Miner")

        config = AutoConfig.from_pretrained(ner_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(ner_model_name, config=config)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")

        self.max_chunk_tokens = max_chunk_tokens
        self.cache_dir = cache_dir
        self.sensitive_terms_file = sensitive_terms_file
        self.input_filter_pattern = re.compile(input_filter_pattern) if input_filter_pattern else None
        self.output_filter_fields = [f.lower() for f in (output_filter_fields or [])]

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(chroma_persist_dir, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
        if "contract_mining_docs" in [c.name for c in self.chroma_client.list_collections()]:
            self.collection = self.chroma_client.get_collection("contract_mining_docs")
        else:
            self.collection = self.chroma_client.create_collection("contract_mining_docs")

        # Load sensitive terms for exclusion from the file, if provided
        self.sensitive_terms = self._load_sensitive_terms() if sensitive_terms_file else []

    def _load_sensitive_terms(self) -> List[str]:
        try:
            with open(self.sensitive_terms_file, "r") as f:
                terms = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(terms)} sensitive terms from {self.sensitive_terms_file}")
            return terms
        except Exception as e:
            logging.error(f"Failed to load sensitive terms file: {e}")
            return []

    ### Utility: PDF text extraction per page ###
    def extract_text_per_page(self, pdf_path: str) -> List[str]:
        try:
            doc = fitz.open(pdf_path)
            pages_text = [page.get_text() for page in doc]
            return pages_text
        except Exception as e:
            logging.error(f"PDF extraction per page failed: {e}")
            return []

    ### Vectorize and store entire PDF into ChromaDB ###
    def vectorize_pdf_and_store(self, pdf_path: str, chunk_size: int = 10):
        pages_text = self.extract_text_per_page(pdf_path)
        page_chunks = []
        for i in range(0, len(pages_text), chunk_size):
            chunk_text = "\n".join(pages_text[i:i+chunk_size])
            page_chunks.append((i, chunk_text))
        for start_page, chunk_text in page_chunks:
            emb = self.get_embedding(chunk_text)
            chunk_id = f"pages_{start_page+1}_{min(start_page+chunk_size, len(pages_text))}"
            self.collection.add(documents=[chunk_text], embeddings=[emb], ids=[chunk_id], metadatas=[{"start_page": start_page+1}])
        logging.info(f"Stored {len(page_chunks)} chunks into vector DB")

    ### Get embedding (placeholder, replace with real embeddings) ###
    def get_embedding(self, text: str) -> List[float]:
        return [0.0] * 1536  # Dummy zero vector for demo

    ### Search vector DB for matching contexts based on fields metadata ###
    def search_contexts_by_fields(self, field_query: str, top_k: int = 100) -> List[str]:
        emb = self.get_embedding(field_query)
        results = self.collection.query(query_embeddings=[emb], n_results=top_k)
        return results['documents'][0] if 'documents' in results else []

    ### Sensitive input filtering (remove sensitive terms & input regex) ###
    def filter_input_text(self, text: str) -> str:
        filtered_text = text
        # Remove sensitive terms from text
        for term in self.sensitive_terms:
            filtered_text = re.sub(re.escape(term), "[FILTERED]", filtered_text, flags=re.IGNORECASE)
        # Apply input regex filter if exists
        if self.input_filter_pattern:
            filtered_text = self.input_filter_pattern.sub("[FILTERED]", filtered_text)
        return filtered_text

    ### Apply output filtering (mask sensitive fields) ###
    def filter_output_field(self, field_name: str, value: str) -> str:
        if field_name.lower() in self.output_filter_fields:
            return mask_sensitive_data(value)
        return value

    ### Cache LLM responses ###
    def cache_response(self, chunk_id: str, response_text: str):
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, f"{chunk_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"response": response_text}, f)

    ### Load cached responses ###
    def load_cached_responses(self) -> Dict[str, str]:
        cached = {}
        if not os.path.exists(self.cache_dir):
            return cached
        for fname in os.listdir(self.cache_dir):
            if fname.endswith(".json"):
                path = os.path.join(self.cache_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = json.load(f)
                    cached[fname[:-5]] = content.get("response", "")
                except Exception as e:
                    logging.error(f"Failed to load cached response {fname}: {e}")
        return cached

    ### Call LLM (placeholder) - replace with real API call ###
    def call_openai_model(self, prompt: str) -> str:
        logging.info(f"Calling LLM on prompt (first 100 chars): {prompt[:10000]}")
        return f"Simulated LLM response to prompt:\n{prompt[:20000]}..."

    ### Calculate hallucination score - simple heuristic ###
    def hallucination_score(self, val: str, context: str) -> float:
        return 0.0 if val in context else 0.9

    ### Process PDF by chunks and query LLM with vector context alongside page context ###
    def process_pdf_with_refinement(self, pdf_path: str, extraction_fields: List[str], query: str,
                                   pages_per_chunk: int = 1000):
        # Step 1: Store entire PDF (in 10-page chunks) in vector DB
        self.vectorize_pdf_and_store(pdf_path, chunk_size=pages_per_chunk)

        # Step 2: Search vector DB for related contexts using extraction fields as query
        metadata_context = self.search_contexts_by_fields(query)

        # Step 3: Extract page-wise PDF text chunks
        pages_text = self.extract_text_per_page(pdf_path)
        total_pages = len(pages_text)
        cached_responses = {}

        # Step 4-7: For each 10-page chunk, apply input filter and call LLM with page & vector metadata context
        for start_idx in range(0, total_pages, pages_per_chunk):
            end_idx = min(start_idx + pages_per_chunk, total_pages)
            chunk_id = f"chunk_{start_idx+1}_{end_idx}"
            # Skip cached if present
            if chunk_id in cached_responses:
                continue
            chunk_text = "\n".join(pages_text[start_idx:end_idx])
            filtered_chunk_text = self.filter_input_text(chunk_text)

            prompt = f"Context Metadata:\n{metadata_context}\n\nPage Text:\n{filtered_chunk_text}\n\nPlease extract relevant fields."
            response = self.call_openai_model(prompt)
            self.cache_response(chunk_id, response)
            cached_responses[chunk_id] = response

        # Load complete cached responses after processing all chunks
        cached_responses = self.load_cached_responses()

        # Step 8: Final aggregation over cached responses, extraction fields to build final result w/ confidence etc
        final_results = {field: {"value": None, "confidence_score": 0.0, "hallucination_score": 1.0} for field in extraction_fields}

        for chunk_resp in cached_responses.values():
            for field in extraction_fields:
                # Use regex to find field values in LLM responses
                pattern = re.compile(rf"{re.escape(field)}[:\-]?\s*(.+)", re.IGNORECASE)
                match = pattern.search(chunk_resp)
                if match:
                    val = match.group(1).strip()
                    conf = 0.9  # simple confidence heuristic
                    halluc = self.hallucination_score(val, chunk_resp)
                    # Update if confidence higher or no prior value
                    if conf > final_results[field]["confidence_score"]:
                        final_results[field].update({"value": val, "confidence_score": conf, "hallucination_score": halluc})

        # Step 9: Output filter to mask sensitive fields
        for f, d in final_results.items():
            d["value"] = self.filter_output_field(f, d.get("value"))

        return final_results

    ### Save final extraction results to file ###
    def save_results(self, results: Dict[str, Dict], outputfile: str):
        try:
            with open(outputfile, "w", encoding="utf-8") as f:
                for field, d in results.items():
                    f.write(f"{field}: {d.get('value')} | Confidence: {d.get('confidence_score')} | Hallucination: {d.get('hallucination_score')}\n")
            logging.info(f"Saved final results to {outputfile}")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Smart Contract Miner with PDF vector storage, refinement, and filtering")
    parser.add_argument("document", help="Path to contract PDF document")
    parser.add_argument("fieldsfile", help="File containing list of fields to extract")
    parser.add_argument("outputfile", help="File to write extracted results")
    parser.add_argument("--query", type=str, required=True, help="Query text for context refinement")
    parser.add_argument("--inputfilter", type=str, default=None, help="Regex pattern to filter input text")
    parser.add_argument("--outputfilterfields", type=str, nargs='*', default=[], help="List of sensitive fields to mask in output")
    parser.add_argument("--sensitivetermsfile", type=str, default=None, help="File with sensitive exclusion terms")
    args = parser.parse_args()

    miner = SmartContractMiner(
        input_filter_pattern=args.inputfilter,
        output_filter_fields=args.outputfilterfields,
        sensitive_terms_file=args.sensitivetermsfile
    )

    extraction_fields = []
    try:
        with open(args.fieldsfile, "r", encoding="utf-8") as f:
            extraction_fields = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Could not read extraction fields file: {e}")

    final_results = miner.process_pdf_with_refinement(args.document, extraction_fields, args.query)

    miner.save_results(final_results, args.outputfile)

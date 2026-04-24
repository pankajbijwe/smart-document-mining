
Comprehensive solution for secure document mining leveraging  AI patterns with data governance, auditability, and robust AI-powered extraction field accuracy. Key features of the solution are as below

# Security & Encryption
  The module includes a standalone security layer using cryptography.fernet to ensure data privacy.
  Encryption/Decryption: Uses a DOC_MINER_ENCRYPTION_KEY from environment variables. If not found, it generates a volatile key for the session.
  Data Masking: The mask_sensitive_data utility obscures strings (e.g., "Password" becomes "P******d"), useful for logs or UI displays.

# Configuration Parameters
  Parameter	Type	Default	Description
  ner_model_name	str	allenai/longformer...	The HuggingFace model used for Entity Recognition.
  max_chunk_tokens	int	4000	Token limit for the Longformer model.
  cache_dir	str	"cache_responses"	Directory to store local JSON LLM caches.
  chroma_persist_dir	str	"chroma_persist"	Path for the persistent ChromaDB vector store.
  sensitive_terms_file	str	None	Path to a text file containing terms to be redacted.

# Core Functionalities
  1. PDF Processing & Vectorization
    extract_text_per_page(pdf_path): Uses PyMuPDF (fitz) to convert PDF pages into a list of strings.
    vectorize_pdf_and_store(pdf_path): Chunks the PDF (default 10 pages), generates embeddings, and stores them in ChromaDB. This enables global context searching across document.
  2. Sensitive Data Filtering
    Input Filtering: filter_input_text scans text for terms defined in the sensitive_terms_file or matching the input_filter_pattern regex, replacing them with [FILTERED].
    Output Filtering: filter_output_field automatically masks specific fields (like "Social Security Number") if they are defined in the output_filter_fields list.
  3. RAG & LLM Integration
    search_contexts_by_fields(field_query): Queries the vector database to find the most relevant document sections based on a specific query.
    process_pdf_with_refinement(...): The primary pipeline. It:
    Indexes the PDF into the vector store.
    Retrieves relevant global metadata.
    Iterates through page chunks.
    Applies filters and prepares a "Refined Prompt" for the LLM (OpenAI) combining local page text and global vector context.
  4. Performance & Reliability
    Caching: cache_response and load_cached_responses minimize API costs and latency by storing LLM outputs locally.
    Hallucination Check: hallucination_score provides a heuristic check to see if the extracted value actually exists within the source context.

# Example Usage
  miner = SmartDocMiner(
      sensitive_terms_file="blacklist.txt",
      output_filter_fields=["doc_name", "account_id"]
  )

 Run the pipeline
  miner.process_pdf_with_refinement(
      pdf_path="doc_v1.pdf",
      extraction_fields=["Termination Clause", "Payment Terms"],
      query="What are the late payment penalties?"
  )

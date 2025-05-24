#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configure UTF-8 encoding for console output
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    # For older Python versions
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    
    # 프로젝트 루트 디렉토리 찾기 (.env 파일 위치 기준)
    current_path = Path(__file__).resolve()
    project_root_candidates = [current_path.parent.parent, current_path.parent, Path.cwd()]
    env_path_found = None
    
    for candidate_path in project_root_candidates:
        potential_env_path = candidate_path / '.env'
        if potential_env_path.exists():
            env_path_found = potential_env_path
            break
    
    if env_path_found:
        load_dotenv(dotenv_path=env_path_found)
        print(f"Loaded .env file from: {env_path_found}")
    else:
        load_dotenv()
        print("Attempting to load .env from current working directory or default locations.")
        
    # API 키 확인 (예시)
    # if os.environ.get('GROK_API_KEY'):
    #     print(f"GROK_API_KEY is set: {os.environ.get('GROK_API_KEY')[:4]}...{os.environ.get('GROK_API_KEY')[-4:]}")
    # if os.environ.get('PINECONE_API_KEY'): # Assuming Milvus, Pinecone was an example
    #     print(f"PINECONE_API_KEY is set: {os.environ.get('PINECONE_API_KEY')[:4]}...{os.environ.get('PINECONE_API_KEY')[-4:]}")
    
except Exception as e:
    print(f"Warning: Error loading environment variables: {e}. Some features may not work properly.")

"""
Main module for the RAG system.
Provides a command-line interface and orchestrates the entire RAG pipeline.
"""

import os
import argparse
import logging
import sys
import json
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple

from src.utils.config import Config
from src.utils.logger import get_logger, setup_logger
from src.rag.parser import DocumentParser  # Corrected import
from src.rag.embedder import DocumentEmbedder
from src.rag.retriever import DocumentRetriever
from src.rag.generator import ResponseGenerator

logger = get_logger("main")


def process_files(args, config: Config):
    """
    Process files to extract content and generate embeddings.
    
    Args:
        args: Command-line arguments.
        config: Configuration object.
    """
    logger.info("Starting file processing")
    
    # Initialize parser
    parser = DocumentParser(config) # Uses the new DocumentParser
    
    # Process files
    document = None # Initialize document to None
    if args.file:
        # Process a single file
        file_path = args.file
        logger.info(f"Processing file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
        
        # Determine file type and process accordingly
        if file_path.lower().endswith('.pdf'):
            document = parser.process_pdf(file_path, args.output)
        elif file_path.lower().endswith('.csv'):
            document = parser.process_csv(file_path, args.output)
        elif file_path.lower().endswith('.txt'):
            document = parser.process_text(file_path, args.output)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return
        
        if document and document.chunks:
            logger.info(f"Processed {len(document.chunks)} small chunks from {file_path}")
            if document.parent_chunks:
                 logger.info(f"Generated {len(document.parent_chunks)} parent chunks for {file_path}")
        elif document:
            logger.warning(f"No chunks were generated for {file_path}, but document object was created.")
        else:
            logger.error(f"Failed to process document {file_path}")
            return # Exit if document processing failed
        
        # Generate embeddings if requested and document was processed
        if args.embed and document:
            logger.info(f"Generating embeddings for {file_path}")
            embedder = DocumentEmbedder(config)
            # embed_document expects a Document object and modifies it in place or returns it
            document_with_embeddings = embedder.embed_document(document) 
            if document_with_embeddings:
                document = document_with_embeddings # Update document with embeddings
            else:
                logger.error(f"Failed to generate embeddings for {file_path}")
                # Decide if you want to proceed without embeddings or return
            
            # Save embeddings output (usually the modified document object)
            if args.output and document: # Check document again
                output_name = os.path.splitext(os.path.basename(file_path))[0] + "_parsed_with_embeddings.json"
                output_path = os.path.join(args.output, output_name)
                
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved parsed document with embeddings to {output_path}")
                except Exception as e:
                    logger.error(f"Error saving document with embeddings to {output_path}: {e}")

            # Store in vector database if requested and document has embeddings
            if args.store_db and document and any(hasattr(chunk, 'metadata') and chunk.metadata and 'embedding' in chunk.metadata for chunk in document.chunks):
                logger.info(f"Storing document {file_path} in vector database")
                # Ensure embedder is initialized if not done above (though it should be for args.embed)
                if 'embedder' not in locals(): 
                    embedder = DocumentEmbedder(config)

                collection_name_base = os.path.splitext(os.path.basename(file_path))[0]
                # Assuming embedder.milvus_client.sanitize_name exists for robust collection names
                collection_name = embedder.milvus_client.sanitize_name(collection_name_base) if embedder.milvus_client else collection_name_base
                
                success = embedder.store_document_in_db(document, collection_name) # Pass the potentially modified document
                
                if success:
                    logger.info(f"Successfully stored document in collection: {collection_name}")
                else:
                    logger.error(f"Failed to store document in vector database: {collection_name}")
            elif args.store_db:
                 logger.warning(f"Skipping storage for {file_path}: Document not processed, or no embeddings found.")
    
    elif args.directory:
        # Process a directory of files
        input_dir = args.directory
        logger.info(f"Processing directory: {input_dir}")
        
        if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
            logger.error(f"Directory not found or not a directory: {input_dir}")
            return
        
        # parser.process_directory should return a list of Document objects
        processed_documents = parser.process_directory(input_dir, args.output)
        logger.info(f"Processed {len(processed_documents)} documents from {input_dir}")
        
        if args.embed:
            logger.info("Generating embeddings for all processed documents in directory.")
            embedder = DocumentEmbedder(config)
            
            for doc_idx, document_obj in enumerate(processed_documents):
                if not document_obj:
                    logger.warning(f"Skipping a null document object at index {doc_idx} from directory processing.")
                    continue
                
                logger.info(f"Embedding document: {document_obj.source}")
                document_with_embeddings = embedder.embed_document(document_obj)
                if document_with_embeddings:
                    processed_documents[doc_idx] = document_with_embeddings # Update with embeddings
                else:
                    logger.error(f"Failed to generate embeddings for {document_obj.source}")
                    continue # Skip saving and storing if embedding failed

                # Save individual document with embeddings
                if args.output and document_with_embeddings: # Check again
                    output_name = os.path.splitext(os.path.basename(document_with_embeddings.source))[0] + "_parsed_with_embeddings.json"
                    output_path = os.path.join(args.output, output_name)
                    try:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(document_with_embeddings.to_dict(), f, ensure_ascii=False, indent=2)
                        logger.info(f"Saved parsed document with embeddings to {output_path}")
                    except Exception as e:
                        logger.error(f"Error saving document {output_path}: {e}")

                if args.store_db and document_with_embeddings and any(hasattr(chunk, 'metadata') and chunk.metadata and 'embedding' in chunk.metadata for chunk in document_with_embeddings.chunks):
                    logger.info(f"Storing document {document_with_embeddings.source} in vector database")
                    collection_name_base = os.path.splitext(os.path.basename(document_with_embeddings.source))[0]
                    collection_name = embedder.milvus_client.sanitize_name(collection_name_base) if embedder.milvus_client else collection_name_base

                    success = embedder.store_document_in_db(document_with_embeddings, collection_name)
                    if success:
                        logger.info(f"Successfully stored document in collection: {collection_name}")
                    else:
                        logger.error(f"Failed to store document in vector database: {collection_name}")
                elif args.store_db:
                    logger.warning(f"Skipping storage for {document_with_embeddings.source if document_with_embeddings else 'unknown_doc'}: No embeddings found.")
    else:
        logger.error("No file or directory specified for 'parse' command.")


def process_embeddings(args, config: Config):
    """
    Process document chunks (previously parsed JSONs) to generate and store embeddings.
    Assumes input files are JSONs created by the parser.py (Document.to_dict() format).
    """
    logger.info("Starting embedding processing from parsed JSON files")
    
    embedder = DocumentEmbedder(config)
    
    files_to_process = []
    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"Input JSON file not found: {args.file}")
            return
        if not args.file.lower().endswith('.json'):
            logger.error(f"Input file must be a .json file (parsed document). Got: {args.file}")
            return
        files_to_process.append(args.file)
    elif args.directory:
        if not os.path.isdir(args.directory):
            logger.error(f"Input directory not found: {args.directory}")
            return
        for fname in os.listdir(args.directory):
            if fname.lower().endswith('_parsed.json') or fname.lower().endswith('_structured_chunks.json'): # Match files from parser
                files_to_process.append(os.path.join(args.directory, fname))
        if not files_to_process:
            logger.warning(f"No '*_parsed.json' or '*_structured_chunks.json' files found in directory: {args.directory}")
            return
    else:
        logger.error("No input file or directory specified for 'embed' command.")
        return

    logger.info(f"Found {len(files_to_process)} parsed JSON files to process for embeddings.")

    for file_path in files_to_process:
        logger.info(f"Processing embeddings for parsed file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_dict = json.load(f)
            
            document = Document.from_dict(doc_dict) # Recreate Document object
            
            # Ensure embedder is re-initialized if config changed or for clarity
            # embedder = DocumentEmbedder(config) # Potentially redundant if config is stable

            document_with_embeddings = embedder.embed_document(document)
            if not document_with_embeddings:
                logger.error(f"Failed to generate embeddings for {file_path}")
                continue

            # Save embeddings to a new file or overwrite
            output_file_path = None
            if args.output:
                output_dir = args.output
                if not os.path.isdir(output_dir): # If output is a directory
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    base_name = os.path.basename(file_path).replace('_parsed.json', '').replace('_structured_chunks.json', '')
                    output_name = f"{base_name}_with_embeddings.json"
                    output_file_path = os.path.join(output_dir, output_name)
                else: # if args.output is a file path (less likely for directory processing)
                    output_file_path = args.output 
            else: # Default: save in the same directory as input
                base_name = os.path.basename(file_path).replace('_parsed.json', '').replace('_structured_chunks.json', '')
                output_name = f"{base_name}_with_embeddings.json"
                output_file_path = os.path.join(os.path.dirname(file_path), output_name)
            
            if output_file_path:
                try:
                    Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        json.dump(document_with_embeddings.to_dict(), f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved document with embeddings to {output_file_path}")
                except Exception as e:
                    logger.error(f"Error saving document with embeddings to {output_file_path}: {e}")

            if args.store_db:
                # Use document title or filename for collection name
                collection_name_base = document_with_embeddings.title or os.path.splitext(os.path.basename(document_with_embeddings.source))[0]
                collection_name = embedder.milvus_client.sanitize_name(collection_name_base) if embedder.milvus_client else collection_name_base
                
                logger.info(f"Storing embeddings from {file_path} into collection: {collection_name}")
                success = embedder.store_document_in_db(document_with_embeddings, collection_name_override=collection_name)
                if success:
                    logger.info(f"Successfully stored embeddings from {file_path} into DB.")
                else:
                    logger.error(f"Failed to store embeddings from {file_path} into DB.")
        
        except FileNotFoundError:
            logger.error(f"Parsed JSON file not found: {file_path}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {file_path}")
        except Exception as e:
            logger.error(f"Unexpected error processing parsed file {file_path}: {e}", exc_info=True)


def process_query(args, config: Config):
    """
    Process a query to generate a response.
    
    Args:
        args: Command-line arguments.
        config: Configuration object.
    """
    logger.info("Starting query processing")
    
    if not args.query:
        logger.error("No query specified")
        return
    
    query = args.query
    logger.info(f"Processing query: {query}")
    
    retriever = DocumentRetriever(config)
    
    # Determine which collections to search
    target_collections = retriever.collections # Default to all known by retriever
    if args.collections:
        specified_collections = [c.strip() for c in args.collections.split(',')]
        # Filter against known collections to ensure they exist
        valid_specified_collections = [c for c in specified_collections if c in retriever.collections]
        if not valid_specified_collections:
            logger.warning(f"None of the specified collections ({specified_collections}) exist or are known. Searching all available collections: {retriever.collections}")
        else:
            target_collections = valid_specified_collections
            logger.info(f"Searching in specified collections: {target_collections}")
    
    if not target_collections and not retriever.offline_mode:
        logger.error("No collections available or specified to search. Please parse/embed documents first or check collection names.")
        return
    elif not target_collections and retriever.offline_mode:
        logger.info("Offline mode, retriever will use dummy data if no internal collections set.")


    top_k = args.top_k if args.top_k is not None else config.retrieval.get('top_k', 5)
    threshold = args.threshold if args.threshold is not None else config.retrieval.get('similarity_threshold', 0.7)
    use_parent_chunks_flag = args.parent_chunks # From argparse

    # Retrieve documents
    # Note: retriever methods need to be updated to accept use_parent_chunks
    if args.hybrid:
        logger.info(f"Using hybrid retrieval (Small-to-Big: {'ON' if use_parent_chunks_flag else 'OFF'})")
        # Assuming hybrid_retrieve will be updated to handle use_parent_chunks
        results = retriever.hybrid_retrieve(query, top_k, threshold, target_collections) # Add use_parent_chunks if method supports
    else:
        logger.info(f"Using vector retrieval (Small-to-Big: {'ON' if use_parent_chunks_flag else 'OFF'})")
        # Assuming retrieve will be updated to handle use_parent_chunks
        results = retriever.retrieve(query, top_k, threshold, target_collections) # Add use_parent_chunks if method supports
    
    logger.info(f"Retrieved {len(results)} documents initially")

    # --- Small-to-Big Context Expansion Placeholder ---
    # This is where the logic to swap small chunks for parent chunks would go,
    # if use_parent_chunks_flag is True and retriever itself doesn't handle it.
    # This would require access to the Document objects or a parent document store.
    # For now, we assume retriever handles it or we pass small chunks to generator.
    # If retriever is not modified, `results` will contain small chunks.
    # If it IS modified, `results` might already contain parent chunk content.
    
    final_context_for_generator = []
    if use_parent_chunks_flag:
        logger.info("Small-to-Big: Attempting to use parent chunk context.")
        # This is a conceptual placeholder. The actual implementation requires
        # retriever.py to fetch parent docs based on retrieved child chunk metadata.
        # For now, we demonstrate the principle if retriever returned child chunks
        # and we had access to the full Document objects.
        
        # This simple pass-through assumes retriever.py has already handled parent lookup
        # OR that we are passing small chunks and generator doesn't expect parent content yet.
        # A proper implementation here would involve:
        # 1. Loading the original Document JSONs that contain both child and parent chunks.
        # 2. For each `res` in `results` (which is a child chunk dict):
        #    - Find its `doc_id` and `parent_chunk_id`.
        #    - Load the corresponding `Document` object.
        #    - Find the parent chunk with `parent_chunk_id` from `document.parent_chunks`.
        #    - Add the parent chunk's text to `final_context_for_generator`.
        # This is too complex for main.py without a proper parent document store access.
        # So, we'll assume retriever will handle this or this is a future step for retriever.py
        
        # TEMPORARY: For now, we pass results as is. Retriever needs modification.
        logger.warning("Small-to-Big parent chunk expansion logic in main.py is conceptual. Retriever should ideally handle this.")
        final_context_for_generator = results # Pass small chunks for now
    else:
        final_context_for_generator = results

    if not final_context_for_generator and len(results) > 0: # Fallback if parent expansion yielded nothing but we had initial results
        final_context_for_generator = results

    # Print results if requested
    if args.print_results:
        print(f"\nQuery: {query}")
        # Distinguish if these are small or parent chunks if possible
        print(f"Context documents provided to generator ({('Parent Chunks' if use_parent_chunks_flag and final_context_for_generator is not results else 'Retrieved Chunks')}): {len(final_context_for_generator)}\n")
        
        for i, result in enumerate(final_context_for_generator):
            print(f"Context {i+1} (Original similarity: {result.get('similarity', 'N/A'):.4f}):")
            print(f"  Collection: {result.get('collection', 'N/A')}")
            if 'metadata' in result:
                print(f"  Metadata: {result.get('metadata', {})}")
            
            content_to_show = result.get('content', result.get('text', 'No content')) # Adapt to potential key change
            if args.highlight and hasattr(retriever, 'highlight_keywords'):
                highlighted = retriever.highlight_keywords(content_to_show, query.split(), context_size=100)
                print(f"  Content: {highlighted}")
            else:
                print(f"  Content: {content_to_show[:300]}...") # Show more context
            print("-" * 80)
    
    # Generate response if requested
    if args.generate:
        if not final_context_for_generator:
            logger.warning("No context documents to send to generator. LLM will answer without RAG context.")
            # Optionally, provide a message or let the generator handle empty context
        
        logger.info(f"Generating response using {len(final_context_for_generator)} context documents.")
        generator = ResponseGenerator(config)
        response = generator.generate(query, final_context_for_generator) # Pass potentially expanded context
        
        print(f"\nQuery: {query}")
        print(f"\nResponse:\n{response}")


def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(description="RAG System Command Line Interface")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)
    
    # --- Parse Command ---
    parse_parser = subparsers.add_parser("parse", help="Process source documents (PDF, CSV, TXT) into structured JSON chunks.")
    parse_parser.add_argument("--file", "-f", help="Path to a single source file (PDF, CSV, TXT) to process.")
    parse_parser.add_argument("--directory", "-d", help="Path to a directory of source files to process.")
    parse_parser.add_argument("--output", "-o", help="Output directory for processed JSON chunk files. Defaults to input directory.")
    parse_parser.add_argument("--embed", "-e", action="store_true", help="Also generate embeddings and save (implies --store-db is separate).")
    parse_parser.add_argument("--store-db", "-s", action="store_true", help="Store document (its small chunks) in vector database after parsing (and embedding if --embed is also used).")
    parse_parser.add_argument("--config", "-c", help="Path to a configuration file (e.g., default_config.yaml).")
    
    # --- Embed Command ---
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings for already parsed JSON document chunk files.")
    embed_parser.add_argument("--file", "-f", help="Path to a single parsed JSON chunk file (output of 'parse' command).")
    embed_parser.add_argument("--directory", "-d", help="Path to a directory of parsed JSON chunk files.")
    embed_parser.add_argument("--output", "-o", help="Output directory for JSON files with added embeddings. Defaults to input JSON's directory.")
    embed_parser.add_argument("--store-db", "-s", action="store_true", default=True, help="Store embeddings in vector database (default: True). Use --no-store-db to disable.")
    embed_parser.add_argument("--config", "-c", help="Path to a configuration file.")
    
    # --- Query Command ---
    query_parser = subparsers.add_parser("query", help="Query the RAG system using existing vector database.")
    query_parser.add_argument("--query", "-q", required=True, help="Query string to process.")
    query_parser.add_argument("--collections", "-cols", help="Comma-separated list of collections to search. Defaults to all known by retriever.")
    query_parser.add_argument("--top-k", "-k", type=int, help="Number of initial results to retrieve.")
    query_parser.add_argument("--threshold", "-t", type=float, help="Similarity threshold for retrieval (0.0 to 1.0).")
    query_parser.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval (vector + keyword).")
    query_parser.add_argument("--parent-chunks", "-pc", action=argparse.BooleanOptionalAction, default=True, help="Use parent chunks for context if available (Small-to-Big strategy). Default is --parent-chunks (True). Use --no-parent-chunks to disable.")
    query_parser.add_argument("--print-results", "-p", action="store_true", help="Print detailed retrieval results (context provided to LLM).")
    query_parser.add_argument("--highlight", action="store_true", help="Highlight query keywords in retrieval results (if -p is used).")
    query_parser.add_argument("--generate", "-g", action="store_true", default=True, help="Generate a response using LLM (default: True). Use --no-generate to only retrieve.")
    query_parser.add_argument("--config", "-c", help="Path to a configuration file.")
    
    # --- Interactive Mode Command ---
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive query mode.")
    interactive_parser.add_argument("--collections", "-cols", help="Comma-separated list of collections to search. Defaults to all.")
    interactive_parser.add_argument("--top-k", "-k", type=int, help="Default number of results for retrieval.")
    interactive_parser.add_argument("--threshold", "-t", type=float, help="Default similarity threshold.")
    interactive_parser.add_argument("--hybrid", action="store_true", help="Default to hybrid retrieval.")
    interactive_parser.add_argument("--parent-chunks", "-pc", action=argparse.BooleanOptionalAction, default=True, help="Default to using parent chunks (Small-to-Big). Use --no-parent-chunks to disable.")
    interactive_parser.add_argument("--config", "-c", help="Path to a configuration file.")
    
    args = parser.parse_args()
    
    # Setup logging (can be overridden by config file later if it also sets up logging)
    # Ensure this is called early, but Config object might reconfigure it.
    # Consider passing log level from args too.
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    setup_logger(level=log_level) # Setup with default or ENV level
    
    if not hasattr(args, 'command') or not args.command: # Should be caught by `required=True` in add_subparsers
        parser.print_help()
        sys.exit(1)
    
    try:
        config_path = args.config if hasattr(args, 'config') and args.config else None
        config = Config(config_path) # Config might also set up logging
        
        # Re-apply log level from config if it exists, otherwise keep initial setup
        # This allows config file to override ENV or default.
        final_log_level_str = config.get_general('log_level', log_level_str).upper()
        final_log_level = getattr(logging, final_log_level_str, log_level)
        if logging.getLogger().getEffectiveLevel() != final_log_level:
             setup_logger(level=final_log_level) # Re-setup if different
             logger.info(f"Logger reconfigured to level: {final_log_level_str}")
        
        logger.info(f"Executing command: {args.command}")

        if args.command == "parse":
            if not args.file and not args.directory:
                parse_parser.error("Either --file or --directory must be specified for 'parse' command.")
            process_files(args, config)
        elif args.command == "embed":
            if not args.file and not args.directory:
                embed_parser.error("Either --file or --directory must be specified for 'embed' command.")
            process_embeddings(args, config)
        elif args.command == "query":
            process_query(args, config)
        elif args.command == "interactive":
            interactive_mode(args, config)
    
    except Exception as e:
        logger.critical(f"Critical error executing command '{args.command}': {e}", exc_info=True)
        # logger.error(traceback.format_exc()) # Redundant if exc_info=True
        sys.exit(1)

def interactive_mode(args, config: Config):
    import types # For SimpleNamespace if needed for args
    
    logger.info("Starting interactive RAG mode...")

    # Convert args to a mutable namespace if it's not already one (e.g. from subparsers)
    # This allows us to toggle settings like print_results, hybrid, parent_chunks
    if not isinstance(args, argparse.Namespace):
         current_args_dict = vars(args) if hasattr(args, '__dict__') else {}
         mutable_args = argparse.Namespace(**current_args_dict)
    else:
        mutable_args = args # It's already a Namespace

    # Set defaults for interactive mode if not provided by CLI args for 'interactive' command
    # These will be the initial state, can be changed by commands within interactive mode.
    if not hasattr(mutable_args, 'collections') or mutable_args.collections is None:
        mutable_args.collections_str = None # Store as string for 'use' command
    else:
        mutable_args.collections_str = mutable_args.collections

    if not hasattr(mutable_args, 'top_k') or mutable_args.top_k is None:
        mutable_args.top_k = config.retrieval.get('top_k', 5)
    if not hasattr(mutable_args, 'threshold') or mutable_args.threshold is None:
        mutable_args.threshold = config.retrieval.get('similarity_threshold', 0.7)
    if not hasattr(mutable_args, 'hybrid'): # if 'hybrid' was not a flag for 'interactive' command
        mutable_args.hybrid = False # Default to vector search
    # parent_chunks should be handled by BooleanOptionalAction, so it will have a True/False value
    if not hasattr(mutable_args, 'parent_chunks'):
         mutable_args.parent_chunks = True # Default as per its argparse definition

    # For internal toggling of printing details
    mutable_args.print_retrieval_details = False 
    mutable_args.highlight_keywords_interactive = True


    retriever = DocumentRetriever(config)
    generator = ResponseGenerator(config)

    # Initial collection setup
    available_collections = retriever.collections
    current_search_collections = []
    if mutable_args.collections_str:
        specified = [c.strip() for c in mutable_args.collections_str.split(',')]
        current_search_collections = [c for c in specified if c in available_collections]
        if not current_search_collections:
            logger.warning(f"None of specified collections ({specified}) found. Defaulting to all available.")
            current_search_collections = available_collections
    else:
        current_search_collections = available_collections

    if not current_search_collections and not retriever.offline_mode:
        logger.error("No collections available to search in interactive mode. Exiting.")
        print("Error: No collections available. Please add documents or check Milvus/vector DB status.")
        return
    
    print("=" * 80)
    print(" RAG System Interactive Mode ".center(80, "="))
    print("=" * 80)
    if retriever.offline_mode:
        print("INFO: Running in OFFLINE mode. Retrieval will use dummy data.")
    
    def print_current_settings():
        print("\nCurrent Settings:")
        print(f"  Collections to search: {current_search_collections if current_search_collections else 'ALL AVAILABLE (if any)'}")
        print(f"  Top-K: {mutable_args.top_k}")
        print(f"  Similarity Threshold: {mutable_args.threshold}")
        print(f"  Retrieval Mode: {'Hybrid' if mutable_args.hybrid else 'Vector'}")
        print(f"  Use Parent Chunks (Small-to-Big): {'ON' if mutable_args.parent_chunks else 'OFF'}")
        print(f"  Show Retrieval Details: {'ON' if mutable_args.print_retrieval_details else 'OFF'}")
        print(f"  Highlight Keywords: {'ON' if mutable_args.highlight_keywords_interactive else 'OFF'}")
        print("-" * 80)

    print_current_settings()
    print("Type 'help' for commands, or 'exit'/'quit'/'q' to leave.")
    print("-" * 80)

    while True:
        try:
            query = input("Query> ").strip()

            if not query:
                continue
            if query.lower() in ['exit', 'quit', 'q']:
                print("Exiting interactive mode.")
                break
            
            command_processed = False
            if query.lower() == 'help':
                print("\nInteractive Commands:")
                print("  help                 - Show this help message.")
                print("  exit/quit/q          - Exit interactive mode.")
                print("  settings             - Show current retrieval settings.")
                print("  list collections     - List all available collections in the DB.")
                print("  use <col1,col2,...>  - Set collections to search (comma-separated). 'use all' for all available.")
                print("  topk <N>             - Set Top-K for retrieval (e.g., 'topk 5').")
                print("  threshold <0.N>      - Set similarity threshold (e.g., 'threshold 0.75').")
                print("  mode hybrid/vector   - Set retrieval mode.")
                print("  s2b on/off           - Toggle Small-to-Big (parent chunks) ON or OFF.")
                print("  details on/off       - Toggle display of detailed retrieval results ON or OFF.")
                print("  highlight on/off     - Toggle keyword highlighting in details ON or OFF.")
                print("Anything else is treated as a query to the RAG system.")
                command_processed = True
            elif query.lower() == 'settings':
                print_current_settings()
                command_processed = True
            elif query.lower() == 'list collections':
                available_collections = retriever.list_collections() # Refresh
                print(f"Available collections in DB: {available_collections if available_collections else 'None found'}")
                command_processed = True
            elif query.lower().startswith('use '):
                cols_str = query[4:].strip()
                available_collections = retriever.list_collections() # Refresh
                if cols_str.lower() == 'all':
                    current_search_collections = available_collections
                    print(f"Now searching all available collections: {current_search_collections}")
                elif cols_str:
                    specified = [c.strip() for c in cols_str.split(',')]
                    valid_new_search_cols = [c for c in specified if c in available_collections]
                    if valid_new_search_cols:
                        current_search_collections = valid_new_search_cols
                        print(f"Now searching in collections: {current_search_collections}")
                    else:
                        print(f"Error: None of the specified collections ({specified}) are available. Available: {available_collections}")
                else:
                    print("Error: 'use' command needs collection names (e.g., 'use my_collection') or 'use all'.")
                command_processed = True
            elif query.lower().startswith('topk '):
                try:
                    val = int(query[5:].strip())
                    if val > 0: mutable_args.top_k = val; print(f"Top-K set to {val}.")
                    else: print("Error: Top-K must be positive.")
                except ValueError: print("Error: Invalid number for Top-K.")
                command_processed = True
            elif query.lower().startswith('threshold '):
                try:
                    val = float(query[10:].strip())
                    if 0.0 <= val <= 1.0: mutable_args.threshold = val; print(f"Threshold set to {val}.")
                    else: print("Error: Threshold must be between 0.0 and 1.0.")
                except ValueError: print("Error: Invalid number for threshold.")
                command_processed = True
            elif query.lower().startswith('mode '):
                mode = query[5:].strip().lower()
                if mode == 'hybrid': mutable_args.hybrid = True; print("Retrieval mode set to Hybrid.")
                elif mode == 'vector': mutable_args.hybrid = False; print("Retrieval mode set to Vector.")
                else: print("Error: Unknown mode. Use 'hybrid' or 'vector'.")
                command_processed = True
            elif query.lower().startswith('s2b '): # Small-to-Big
                val = query[4:].strip().lower()
                if val == 'on': mutable_args.parent_chunks = True; print("Small-to-Big (parent chunks) ON.")
                elif val == 'off': mutable_args.parent_chunks = False; print("Small-to-Big (parent chunks) OFF.")
                else: print("Error: Use 's2b on' or 's2b off'.")
                command_processed = True
            elif query.lower().startswith('details '):
                val = query[8:].strip().lower()
                if val == 'on': mutable_args.print_retrieval_details = True; print("Show retrieval details ON.")
                elif val == 'off': mutable_args.print_retrieval_details = False; print("Show retrieval details OFF.")
                else: print("Error: Use 'details on' or 'details off'.")
                command_processed = True
            elif query.lower().startswith('highlight '):
                val = query[10:].strip().lower()
                if val == 'on': mutable_args.highlight_keywords_interactive = True; print("Keyword highlighting ON.")
                elif val == 'off': mutable_args.highlight_keywords_interactive = False; print("Keyword highlighting OFF.")
                else: print("Error: Use 'highlight on' or 'highlight off'.")
                command_processed = True

            if command_processed:
                print("-" * 80)
                continue

            # --- Process the actual query ---
            logger.info(f"Interactive query: {query}")
            print(f"\nThinking...")

            # Retrieve documents
            # CRITICAL: Update retriever methods to accept and use 'use_parent_chunks'
            # For now, this flag is for display and future retriever logic.
            # The actual fetching of parent chunks needs to be implemented in retriever.py
            retrieved_docs_for_context = []
            if mutable_args.hybrid:
                retrieved_docs_for_context = retriever.hybrid_retrieve(
                    query, 
                    mutable_args.top_k, 
                    mutable_args.threshold, 
                    current_search_collections
                    # Pass mutable_args.parent_chunks here if/when retriever supports it
                )
            else:
                retrieved_docs_for_context = retriever.retrieve(
                    query, 
                    mutable_args.top_k, 
                    mutable_args.threshold, 
                    current_search_collections
                     # Pass mutable_args.parent_chunks here if/when retriever supports it
                )
            
            logger.info(f"Retrieved {len(retrieved_docs_for_context)} documents for query '{query}'.")

            if mutable_args.print_retrieval_details:
                print("\n--- Retrieval Details ---")
                if not retrieved_docs_for_context:
                    print("No documents retrieved.")
                for i, doc_data in enumerate(retrieved_docs_for_context):
                    print(f"  Detail {i+1}: Score: {doc_data.get('similarity', 'N/A'):.4f}, Collection: {doc_data.get('collection', 'N/A')}")
                    metadata = doc_data.get('metadata', {})
                    print(f"    Source: {metadata.get('source_file', metadata.get('source', 'N/A'))}, Page: {metadata.get('page_num', 'N/A')}")
                    if metadata.get('article_title'): print(f"    Article: {metadata['article_title']}")
                    if metadata.get('item_marker'): print(f"    Item: {metadata['item_marker']}")
                    
                    content_preview = doc_data.get('content', doc_data.get('text', 'N/A'))
                    if mutable_args.highlight_keywords_interactive and hasattr(retriever, 'highlight_keywords'):
                        highlighted_preview = retriever.highlight_keywords(content_preview, query.split(), context_size=30)
                        print(f"    Preview: {highlighted_preview}")
                    else:
                        print(f"    Preview: {content_preview[:150]}...")
                print("--- End Retrieval Details ---")

            if not retrieved_docs_for_context:
                print("\nNo relevant documents found to answer your query.")
            else:
                print(f"\nGenerating response based on {len(retrieved_docs_for_context)} context(s)...")
                # The `generator.generate` method expects a list of dicts,
                # where each dict has "content" and "metadata" (and optionally "similarity", "collection").
                # The current `retriever.retrieve` already returns this format.
                llm_response = generator.generate(query, retrieved_docs_for_context)
                print("\n--- Generated Response ---")
                print(llm_response)
            print("-" * 80)

        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}", exc_info=True)
            print(f"\nAn error occurred: {e}")
            print("-" * 80)


if __name__ == "__main__":
    main()
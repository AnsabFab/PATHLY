import os
import json
import datetime
import uuid
import asyncio
import faiss
import numpy as np
import pickle
import streamlit as st
import pandas as pd
import requests
from typing import Dict, List, Optional, Any
from duckduckgo_search import DDGS
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Constants
EMOTION_PROMPT = """Analyze the emotional content in this journal entry. Return a JSON object with:
{"primary_emotion": "emotion", "intensity": 1-10, "triggers": ["list"], "growth_opportunities": ["list"]}"""

GROWTH_PLAN_PROMPT = """Based on the emotional analysis and user goals, create a structured growth plan. Include:
{"title": "", "steps": [{"title":"", "description":""}], "expected_outcomes": [""]}"""

RESOURCE_PROMPT = """Synthesize these web resources for the user's emotional growth profile. Include:
{"key_insights": [""], "practical_exercises": [""], "recommended_readings": [{"title":"", "description":""}]}"""

COMMUNITY_PROMPT = """Suggest community interactions based on this profile:
{"suggested_topics": [""], "conversation_starters": [""], "support_strategies": [""]}"""

# Vector DB Setup
VECTOR_DIM = 768  # For nomic-embed-text
FAISS_PATH = "./faiss_index"
META_PATH = "./faiss_metadata.pkl"

# Initialize session state
def init_state():
    if 'user_db' not in st.session_state:
        st.session_state.user_db = {} # In a real app, load/save this from a file/db
    if 'emotion_db' not in st.session_state:
        st.session_state.emotion_db = {} # In a real app, load/save this
    if 'growth_plans_db' not in st.session_state:
        st.session_state.growth_plans_db = {} # In a real app, load/save this
    if 'resource_db' not in st.session_state:
        st.session_state.resource_db = {} # In a real app, load/save this
    if 'community_db' not in st.session_state:
        st.session_state.community_db = [] # In a real app, load/save this
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = load_or_create_faiss_index()
        st.session_state.faiss_metadata = load_or_create_metadata()

# FAISS Functions
def load_or_create_faiss_index():
    if os.path.exists(FAISS_PATH):
        try:
            return faiss.read_index(FAISS_PATH)
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")
            os.remove(FAISS_PATH) # Remove corrupted index
            if os.path.exists(META_PATH):
                os.remove(META_PATH) # Remove potentially mismatched metadata

    # Create new index
    st.warning("Creating new FAISS index.")
    index = faiss.IndexFlatL2(VECTOR_DIM)
    return index

def load_or_create_metadata():
    if os.path.exists(META_PATH):
        # Ensure index exists if metadata does
        if not os.path.exists(FAISS_PATH):
             st.warning("Metadata found without index. Creating new index and metadata.")
             if os.path.exists(META_PATH): os.remove(META_PATH)
             return []
        try:
            with open(META_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading metadata: {e}. Resetting metadata.")
            if os.path.exists(META_PATH): os.remove(META_PATH) # Remove corrupted metadata
            # Attempt to recover index if possible, otherwise create new
            if os.path.exists(FAISS_PATH):
                try:
                    st.session_state.faiss_index = faiss.read_index(FAISS_PATH)
                except:
                     st.error("Failed to reload index after metadata error. Creating new index.")
                     if os.path.exists(FAISS_PATH): os.remove(FAISS_PATH)
                     st.session_state.faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
            else:
                 st.session_state.faiss_index = faiss.IndexFlatL2(VECTOR_DIM)

    st.warning("Creating new metadata file.")
    return []

def save_faiss_index():
    try:
        # Ensure index and metadata are consistent before saving
        if st.session_state.faiss_index.ntotal != len(st.session_state.faiss_metadata):
            st.error(f"FAISS index size ({st.session_state.faiss_index.ntotal}) and metadata size ({len(st.session_state.faiss_metadata)}) mismatch. Saving aborted.")
            # Consider more robust recovery or logging here
            return

        faiss.write_index(st.session_state.faiss_index, FAISS_PATH)
        with open(META_PATH, 'wb') as f:
            pickle.dump(st.session_state.faiss_metadata, f)
        # st.toast("Index saved.") # Optional: uncomment for user feedback
    except Exception as e:
        st.error(f"Error saving FAISS index or metadata: {e}")

def get_ollama_embeddings(texts: List[str], model="nomic-embed-text"):
    """Gets embeddings for a list of texts using a local Ollama instance."""
    embeddings = []
    # Use environment variable or default
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/embeddings"

    # Check if Ollama server is reachable
    try:
        requests.get(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), timeout=2)
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to Ollama at {ollama_url}. Please ensure Ollama is running.")
        return [np.zeros(VECTOR_DIM, dtype=np.float32) for _ in texts] # Return zero vectors

    for text in texts:
        # Handle empty or non-string inputs gracefully
        if not text or not isinstance(text, str):
            # st.warning("Attempted to embed empty or invalid text. Skipping.")
            embeddings.append(np.zeros(VECTOR_DIM, dtype=np.float32)) # Append a zero vector
            continue

        try:
            response = requests.post(
                ollama_url,
                json={"model": model, "prompt": text.strip()}, # Ensure text is stripped
                headers={"Content-Type": "application/json"},
                timeout=30 # Add a timeout
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()

            if "embedding" in result and isinstance(result["embedding"], list):
                 embedding_array = np.array(result["embedding"], dtype=np.float32)
                 # Validate embedding dimension
                 if embedding_array.shape[0] == VECTOR_DIM:
                     embeddings.append(embedding_array)
                 else:
                    st.warning(f"Received embedding with incorrect dimension ({embedding_array.shape[0]} vs {VECTOR_DIM}). Using zero vector.")
                    embeddings.append(np.zeros(VECTOR_DIM, dtype=np.float32))

            else:
                st.warning(f"Embedding not found in Ollama response for text snippet: '{text[:50]}...'. Response: {result}")
                embeddings.append(np.zeros(VECTOR_DIM, dtype=np.float32))

        except requests.exceptions.RequestException as e:
            st.error(f"Ollama embedding request failed: {e}")
            embeddings.append(np.zeros(VECTOR_DIM, dtype=np.float32))
        except json.JSONDecodeError as e:
             st.error(f"Failed to decode Ollama JSON response: {e}")
             embeddings.append(np.zeros(VECTOR_DIM, dtype=np.float32))
        except Exception as e:
            # Catch any other unexpected errors during embedding generation
            st.error(f"Unexpected error getting embedding for '{text[:50]}...': {e}")
            embeddings.append(np.zeros(VECTOR_DIM, dtype=np.float32))

    # Ensure the number of embeddings matches the number of input texts
    if len(embeddings) != len(texts):
         st.error(f"Mismatch between input texts ({len(texts)}) and generated embeddings ({len(embeddings)}). Padding with zeros.")
         # Pad with zero vectors if necessary, though ideally the loop handles this
         while len(embeddings) < len(texts):
             embeddings.append(np.zeros(VECTOR_DIM, dtype=np.float32))
         embeddings = embeddings[:len(texts)] # Truncate if too many (shouldn't happen)


    return embeddings


# Database Functions (Using Session State - simple persistence)
def save_user_data(user_id: str, data: Dict):
    """Saves user profile data."""
    st.session_state.user_db[user_id] = data
    # In a real app, you would save st.session_state.user_db to a file/database here

def save_emotion_entry(user_id: str, emotion_data: Dict):
    """Saves a new emotion journal entry."""
    if user_id not in st.session_state.emotion_db:
        st.session_state.emotion_db[user_id] = []
    emotion_data['timestamp'] = datetime.datetime.now().isoformat()
    emotion_data['id'] = str(uuid.uuid4()) # Generate unique ID
    st.session_state.emotion_db[user_id].append(emotion_data)
    # In a real app, save st.session_state.emotion_db here
    return emotion_data['id']

def get_user_emotion_history(user_id: str) -> List[Dict]:
    """Retrieves all emotion entries for a user."""
    return st.session_state.emotion_db.get(user_id, [])

def get_emotion_entry(user_id: str, emotion_id: str) -> Optional[Dict]:
    """Retrieves a specific emotion entry by its ID."""
    for entry in get_user_emotion_history(user_id):
        if entry.get('id') == emotion_id:
            return entry
    return None

def save_growth_plan(user_id: str, emotion_id: str, plan_data: Dict):
    """Saves a growth plan associated with an emotion entry."""
    if user_id not in st.session_state.growth_plans_db:
        st.session_state.growth_plans_db[user_id] = {}
    st.session_state.growth_plans_db[user_id][emotion_id] = plan_data
    # In a real app, save st.session_state.growth_plans_db here

def get_growth_plan(user_id: str, emotion_id: str) -> Optional[Dict]:
    """Retrieves the growth plan for a specific emotion entry."""
    return st.session_state.growth_plans_db.get(user_id, {}).get(emotion_id)

def save_resource(user_id: str, emotion_id: str, resource_data: Dict):
    """Saves synthesized resources associated with an emotion entry."""
    if user_id not in st.session_state.resource_db:
        st.session_state.resource_db[user_id] = {}
    st.session_state.resource_db[user_id][emotion_id] = resource_data
    # In a real app, save st.session_state.resource_db here

def get_resources(user_id: str, emotion_id: str) -> Optional[Dict]:
    """Retrieves the synthesized resources for a specific emotion entry."""
    return st.session_state.resource_db.get(user_id, {}).get(emotion_id)

# LLM Interaction
def call_llm(prompt: str, system_prompt: str, user_input: str, model="llama3:8b") -> Dict:
    """Calls the Ollama LLM and attempts to parse JSON response."""
    try:
        import ollama # Import here to make dependency optional if Ollama isn't used/installed
    except ImportError:
        st.error("Ollama library not installed. Please install it: pip install ollama")
        return {"error": "Ollama library not installed."}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{prompt}\n\n{user_input}"}
    ]

    try:
        # Check Ollama connection before making the call
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        requests.get(ollama_base_url, timeout=2)

        response = ollama.chat(model=model, messages=messages)
        response_content = response['message']['content']

        # Attempt to extract JSON object from the response
        try:
            # Find the first '{' and the last '}'
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1

            if json_start != -1 and json_end != -1 and json_start < json_end:
                json_str = response_content[json_start:json_end]
                # Validate if it's actually JSON before parsing
                if json_str.startswith('{') and json_str.endswith('}'):
                    parsed_json = json.loads(json_str)
                    return parsed_json
                else:
                     st.warning(f"Extracted text between {{}} doesn't look like valid JSON: {json_str}")
                     return {"raw_response": response_content, "warning": "Could not reliably extract JSON."}
            else:
                 st.warning(f"Could not find JSON object markers '{{' and '}}' in response: {response_content}")
                 return {"raw_response": response_content, "warning": "JSON markers not found."}
        except json.JSONDecodeError as json_err:
            st.warning(f"Failed to parse JSON from LLM response: {json_err}. Raw response: {response_content}")
            return {"raw_response": response_content, "warning": f"JSON parsing failed: {json_err}"}
        except Exception as parse_err: # Catch other potential errors during parsing/extraction
             st.error(f"Error processing LLM response content: {parse_err}")
             return {"raw_response": response_content, "error": f"Error processing response: {parse_err}"}

    except requests.exceptions.ConnectionError:
         st.error(f"Cannot connect to Ollama at {ollama_base_url}. Please ensure Ollama is running.")
         return {"error": f"Failed to connect to Ollama at {ollama_base_url}"}
    except Exception as e:
        # Catch errors from ollama.chat or other issues
        st.error(f"Failed to get response from Ollama model '{model}': {e}")
        return {"error": f"Failed to get response from {model}: {e}"}

def analyze_emotion(journal_entry: str) -> Dict:
    """Analyzes emotion in a journal entry using the LLM."""
    return call_llm("Analyze this journal entry:", EMOTION_PROMPT, journal_entry)

def generate_growth_plan(emotion_analysis: Dict, user_goals: Optional[Dict] = None) -> Dict:
    """Generates a growth plan based on emotion analysis and goals."""
    # Provide default goals if none are given
    input_data = {
        "emotion_analysis": emotion_analysis,
        "user_goals": user_goals or {"general_goal": "Improve emotional awareness and regulation"}
    }
    return call_llm("Create a growth plan based on the following:", GROWTH_PLAN_PROMPT, json.dumps(input_data, indent=2))

def synthesize_resources(emotion_analysis: Dict, growth_plan: Optional[Dict], web_content: List[str]) -> Dict:
    """Synthesizes web content into actionable resources using the LLM."""
    if not web_content:
        st.warning("No web content provided for synthesis.")
        return {"key_insights": ["No content available to synthesize."], "practical_exercises": [], "recommended_readings": []}

    # Limit the amount of content sent to the LLM to avoid exceeding context limits
    MAX_CONTENT_TOKENS = 15000 # Adjust based on model limits (approximate)
    current_content = ""
    token_count = 0
    included_content = []

    for content_piece in web_content:
         # Simple token estimation (split by space)
         piece_tokens = len(content_piece.split())
         if token_count + piece_tokens < MAX_CONTENT_TOKENS:
             included_content.append(content_piece)
             token_count += piece_tokens
         else:
             st.warning(f"Content limit reached for LLM synthesis. Only using first {len(included_content)} pieces.")
             break


    input_data = {
        "emotion_analysis": emotion_analysis,
        "growth_plan": growth_plan or {}, # Include plan if available
        "web_content": "\n---\n".join(included_content) # Join the selected content
    }
    return call_llm("Synthesize these resources related to the user profile:", RESOURCE_PROMPT, json.dumps(input_data, indent=2))

# Web Search and Crawl
def get_web_urls(search_term: str, num_results: int = 3) -> List[str]:
    """Searches the web using DuckDuckGo and filters results."""
    try:
        # Enhance search query for more relevant results
        enhanced_search = f"{search_term} emotional regulation coping strategies self-help"
        st.info(f"Searching web for: '{enhanced_search}'...")
        # Use DDGS().text for organic results
        results = DDGS().text(keywords=enhanced_search, max_results=num_results * 2) # Fetch more initially

        if not results:
            st.warning(f"No search results found for '{enhanced_search}'.")
            return []

        # Extract and filter URLs
        urls = [result["href"] for result in results if result.get("href")]
        filtered_urls = []
        seen_domains = set()

        for url in urls:
            # Basic URL validation and filtering
            if not url or not url.startswith(('http://', 'https://')):
                continue
            if url.lower().endswith((".pdf", ".doc", ".docx", ".ppt", ".pptx")):
                 # st.write(f"Skipping document link: {url}") # Optional logging
                 continue

            try:
                domain = urlparse(url).netloc
                if domain and domain not in seen_domains:
                    # Optional: Add more aggressive filtering (e.g., block social media, forums if desired)
                    # if any(x in domain for x in ['facebook.com', 'twitter.com', 'reddit.com']): continue
                    filtered_urls.append(url)
                    seen_domains.add(domain)
            except Exception as parse_err:
                 st.warning(f"Could not parse URL '{url}': {parse_err}. Skipping.")
                 continue # Skip invalid URLs

        # Check robots.txt for the selected unique domain URLs
        allowed_urls = check_robots_txt(filtered_urls[:num_results]) # Limit to desired number after filtering
        st.write(f"Found {len(allowed_urls)} allowed URLs.")
        return allowed_urls

    except Exception as e:
        st.error(f"Failed to fetch search results using DuckDuckGo: {str(e)}")
        return []

def check_robots_txt(urls: List[str]) -> List[str]:
    """Checks if crawling is allowed by robots.txt for a list of URLs."""
    allowed_urls = []
    user_agent = "PathlyApp/1.0 (+http://yourappdomain.com/bot)" # Be a good citizen

    for url in urls:
        try:
            parsed_url = urlparse(url)
            scheme, netloc = parsed_url.scheme, parsed_url.netloc
            if not scheme or not netloc:
                st.warning(f"Skipping URL with invalid scheme/netloc: {url}")
                continue

            robots_url = f"{scheme}://{netloc}/robots.txt"
            rp = RobotFileParser(robots_url)
            rp.read()

            if rp.can_fetch(user_agent, url):
                allowed_urls.append(url)
            else:
                st.warning(f"Crawling disallowed by robots.txt for: {url}")
        except Exception as e:
             # If robots.txt can't be fetched or parsed, cautiously allow (or disallow)
             st.warning(f"Could not check robots.txt for {url} ({e}). Assuming allowed (use caution).")
             allowed_urls.append(url) # Or choose to disallow by default: continue

    return allowed_urls

async def crawl_webpages(urls: List[str], query: str):
    """Asynchronously crawls webpages using crawl4ai."""
    if not urls:
        return []

    # Configure crawl4ai filter and generator
    bm25_filter = BM25ContentFilter(user_query=query, bm25_threshold=0.5) # Adjust threshold as needed
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)

    # Configure crawler settings
    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        # More specific tags to exclude common noise
        excluded_tags=["nav", "footer", "header", "form", "img", "script", "style", "aside", "figure", "figcaption", "button", "input", "select", "textarea"],
        only_text=True,
        exclude_social_media_links=True,
        cache_mode=CacheMode.BYPASS, # Or use CacheMode.NORMAL for faster re-runs
        remove_overlay_elements=True,
        page_load_timeout=30000, # 30 seconds timeout per page
        max_depth=0 # Only crawl the specified URLs, no deeper links
    )
    # Use headless mode for background operation
    browser_config = BrowserConfig(headless=True, text_mode=True) # text_mode can sometimes help

    crawled_data = []
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Run crawls concurrently
            results = await crawler.arun_many(urls, config=crawler_config)

            # Process results, ensuring markdown exists and is meaningful
            for r in results:
                if r and r.markdown_v2 and r.markdown_v2.fit_markdown and len(r.markdown_v2.fit_markdown) > 100: # Filter very short/empty results
                    crawled_data.append(r)
                elif r:
                     st.warning(f"Crawled {r.url} but content was filtered out or too short.")
            return crawled_data
    except Exception as e:
        st.error(f"Web crawling encountered an error: {e}")
        # Log detailed error if needed: print(traceback.format_exc())
        return [] # Return empty list on failure

def add_to_vector_database(results):
    """Chunks crawled content, gets embeddings, and adds to FAISS."""
    texts_to_embed = []
    metadata = []

    if not results:
        st.info("No valid crawled content to add to the vector database.")
        return []

    st.info(f"Processing {len(results)} crawled pages for vectorization...")

    for result in results:
        # Double-check content validity
        if not result or not result.markdown_v2 or not result.markdown_v2.fit_markdown:
            continue

        content = result.markdown_v2.fit_markdown
        url = result.url or "unknown_source"
        title = result.title or "No Title Provided"

        # Simple Chunking Strategy (by paragraphs, then by size)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        MIN_CHUNK_LEN = 50 # Min characters for a chunk
        MAX_CHUNK_LEN = 800 # Target max characters per chunk
        OVERLAP = 100 # Overlap chars for larger chunks

        for para in paragraphs:
            if not para or len(para) < MIN_CHUNK_LEN:
                continue # Skip empty or very short paragraphs

            if len(para) <= MAX_CHUNK_LEN:
                texts_to_embed.append(para)
                metadata.append({"text": para, "source": url, "title": title})
            else:
                # Split larger paragraphs with overlap
                start = 0
                while start < len(para):
                    end = start + MAX_CHUNK_LEN
                    chunk = para[start:end].strip()
                    if chunk: # Ensure chunk is not just whitespace
                       texts_to_embed.append(chunk)
                       metadata.append({"text": chunk, "source": url, "title": title})
                    # Move start for the next chunk, considering overlap
                    start += MAX_CHUNK_LEN - OVERLAP
                    if start >= len(para) - MIN_CHUNK_LEN: # Avoid tiny last chunk
                        break # Stop if remaining part is too small


    if not texts_to_embed:
        st.warning("No suitable text chunks found after processing crawled content.")
        return []

    st.info(f"Generating embeddings for {len(texts_to_embed)} text chunks...")
    embeddings = get_ollama_embeddings(texts_to_embed)

    # Filter out zero vectors and their corresponding metadata if embedding failed
    valid_embeddings = []
    valid_metadata = []
    original_indices = [] # Keep track of original index for metadata association
    for i, emb in enumerate(embeddings):
         if np.any(emb): # Check if the vector is not all zeros
             valid_embeddings.append(emb)
             valid_metadata.append(metadata[i])
             original_indices.append(len(st.session_state.faiss_metadata) + len(valid_metadata) - 1) # Calculate future index in combined metadata

    if not valid_embeddings:
         st.error("Failed to generate any valid embeddings for the crawled content.")
         return []

    st.info(f"Adding {len(valid_embeddings)} valid vectors to FAISS index...")
    try:
        # Add valid embeddings to FAISS index
        embedding_matrix = np.array(valid_embeddings, dtype=np.float32)
        st.session_state.faiss_index.add(embedding_matrix)

        # Append corresponding valid metadata
        st.session_state.faiss_metadata.extend(valid_metadata)

        # Verify consistency before saving
        if st.session_state.faiss_index.ntotal == len(st.session_state.faiss_metadata):
             save_faiss_index() # Save updated index and metadata
             st.success(f"Successfully added {len(valid_embeddings)} items to vector DB. Total items: {st.session_state.faiss_index.ntotal}")
             # Return the text content that was successfully added
             return [m["text"] for m in valid_metadata]
        else:
            # This case should ideally be prevented by pre-save check in save_faiss_index
            # But as a fallback, try to recover or log error.
             st.error("CRITICAL: Index and metadata count mismatch AFTER adding vectors. State might be corrupted. Saving aborted.")
             # Attempt to rollback? (Difficult with FAISS add)
             # For now, just log the error and return empty.
             # Consider rebuilding index/metadata if this occurs frequently.
             return []

    except Exception as e:
        st.error(f"Error adding vectors to FAISS or saving: {e}")
        # Log detailed error if needed
        return [] # Return empty list on failure


def query_vector_database(query_text: str, n_results: int = 5) -> List[Dict]:
    """Queries the FAISS index for relevant text chunks."""
    results_with_meta = []
    if not query_text:
        st.warning("Query text is empty.")
        return []
    if st.session_state.faiss_index.ntotal == 0:
        st.info("Vector database is empty. Cannot perform query.")
        return []

    st.info(f"Generating embedding for query: '{query_text[:100]}...'")
    query_embedding_list = get_ollama_embeddings([query_text])

    # Check if embedding generation succeeded
    if not query_embedding_list or not np.any(query_embedding_list[0]):
         st.error("Failed to generate embedding for the query text.")
         return []

    query_embedding = query_embedding_list[0].reshape(1, -1).astype(np.float32) # Ensure correct shape and type

    try:
        k = min(n_results, st.session_state.faiss_index.ntotal) # Number of neighbors to search
        st.info(f"Searching top {k} neighbors in vector database...")
        distances, indices = st.session_state.faiss_index.search(query_embedding, k)

        if indices.size == 0 or indices[0][0] == -1: # Check for empty or invalid results
             st.warning("No relevant results found in the vector database for this query.")
             return []

        # Retrieve metadata for the found indices
        retrieved_indices = indices[0]
        for i, idx in enumerate(retrieved_indices):
            if 0 <= idx < len(st.session_state.faiss_metadata):
                meta = st.session_state.faiss_metadata[idx]
                meta['distance'] = float(distances[0][i]) # Add distance score
                results_with_meta.append(meta)
            else:
                st.warning(f"Index {idx} from FAISS search is out of bounds for metadata (size {len(st.session_state.faiss_metadata)}). Skipping.")

        # Optional: Sort results by distance (lower is better for L2)
        results_with_meta.sort(key=lambda x: x.get('distance', float('inf')))

        st.success(f"Found {len(results_with_meta)} results.")
        return results_with_meta # Return list of dictionaries including metadata

    except Exception as e:
        st.error(f"Error querying FAISS index: {e}")
        # Log detailed error if needed
        return []


async def search_and_crawl_resources(emotion_analysis: Dict):
    """Orchestrates web search, crawling, and indexing for resources."""
    emotion = emotion_analysis.get('primary_emotion', 'emotional challenge')
    triggers = emotion_analysis.get('triggers', [])
    opportunities = emotion_analysis.get('growth_opportunities', [])

    # Build more targeted search queries
    search_queries = [
        f"how to cope with feeling {emotion}",
        f"strategies for managing {emotion}",
        f"understanding triggers for {emotion}",
        f"healthy ways to deal with {emotion}"
    ]
    if triggers:
        search_queries.append(f"dealing with {emotion} triggered by {triggers[0]}") # Use first trigger for specific search
    if opportunities:
         search_queries.append(f"personal growth exercises for {opportunities[0]}") # Use first opportunity

    all_urls = set()
    MAX_QUERIES = 2 # Limit number of search engine queries
    URLS_PER_QUERY = 2 # Target URLs per query

    # Perform searches and gather unique, allowed URLs
    for query in search_queries[:MAX_QUERIES]:
        urls = get_web_urls(query, num_results=URLS_PER_QUERY)
        all_urls.update(urls)
        await asyncio.sleep(0.5) # Small delay between searches


    unique_urls = list(all_urls)[:MAX_QUERIES * URLS_PER_QUERY] # Limit total URLs to crawl

    if not unique_urls:
        st.warning("No suitable URLs found after web search and filtering.")
        return []

    # Define a query for the BM25 relevance filter during crawling
    crawl_relevance_query = f"{emotion} coping strategies {' '.join(triggers)} {' '.join(opportunities)}"

    st.info(f"Starting crawl for {len(unique_urls)} relevant resources...")
    crawl_results = await crawl_webpages(unique_urls, query=crawl_relevance_query)

    if not crawl_results:
        st.warning("Crawling finished, but no suitable content was extracted or passed filters.")
        return []

    st.info("Processing and indexing crawled content...")
    # Add crawled content to the vector database
    added_chunks_text = add_to_vector_database(crawl_results)

    return added_chunks_text # Return the list of text chunks that were added


# UI Components
def render_login_page():
    """Displays the login and sign-up forms."""
    st.title("🌱 Pathly: Emotion to Action")
    st.subheader("Transform Emotional Experiences into Personal Growth")

    # Use columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Login")
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submitted = st.form_submit_button("Login")

            if login_submitted:
                if not login_username or not login_password:
                    st.error("Please enter both username and password.")
                elif login_username in st.session_state.user_db:
                    # In a real app, use secure password hashing (e.g., bcrypt)
                    # For this demo, simple comparison
                    if st.session_state.user_db[login_username].get('password') == login_password:
                        st.session_state.current_user = login_username
                        st.session_state.authenticated = True
                        st.session_state.current_view = "main" # Navigate to main dashboard
                        st.success("Login successful!")
                        st.rerun() # Rerun to reflect the new state
                    else:
                        st.error("Invalid password.")
                else:
                    st.error("Username not found. Please sign up.")

    with col2:
        st.subheader("Sign Up")
        with st.form("signup_form"):
            signup_username = st.text_input("Choose Username", key="signup_username")
            signup_password = st.text_input("Choose Password", type="password", key="signup_password")
            signup_confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
            signup_submitted = st.form_submit_button("Sign Up")

            if signup_submitted:
                if not signup_username or not signup_password or not signup_confirm_password:
                    st.error("Please fill in all sign-up fields.")
                elif signup_password != signup_confirm_password:
                    st.error("Passwords do not match.")
                elif signup_username in st.session_state.user_db:
                    st.error("Username already taken. Please choose another.")
                else:
                    # Basic validation (add more as needed, e.g., password strength)
                    if len(signup_password) < 6:
                        st.error("Password must be at least 6 characters long.")
                    else:
                        # In a real app, hash the password before storing
                        hashed_password = signup_password # Placeholder for hashing
                        st.session_state.user_db[signup_username] = {
                            'password': hashed_password, # Store hashed password
                            'joined_date': datetime.datetime.now().isoformat(),
                            'goals': {} # Initialize empty goals
                        }
                        # Persist user_db (e.g., save to file/db)
                        # save_all_user_data(st.session_state.user_db) # Example function call
                        st.success("Account created successfully! You can now login.")
                        st.balloons()

def render_sidebar():
    """Renders the sidebar navigation for authenticated users."""
    st.sidebar.title("Navigation")
    if st.session_state.current_user:
        st.sidebar.write(f"👋 Hello, {st.session_state.current_user}!")

        # Define available pages/views
        pages = {
            "Dashboard": "main",
            "Journal": "journal",
            "Resources": "resources" # Combined resource view/search
            # Add other pages like "Profile", "Community" later
        }

        # Determine the current page index for the radio button
        current_view = st.session_state.get('current_view', 'main')
        page_keys = list(pages.keys())
        try:
            current_index = list(pages.values()).index(current_view)
        except ValueError:
            current_index = 0 # Default to Dashboard if view is somehow invalid

        # Use st.radio for navigation
        selected_page_name = st.sidebar.radio("Go to:", options=page_keys, index=current_index)
        selected_view = pages[selected_page_name]


        # Update view state if selection changed
        if selected_view != current_view:
            st.session_state.current_view = selected_view
            # Reset specific context if navigating away from detail views
            if selected_view in ["main", "journal", "resources"]:
                 if 'current_emotion_id' in st.session_state:
                     del st.session_state['current_emotion_id']
                 if 'emotion_analysis_context' in st.session_state:
                      del st.session_state['emotion_analysis_context']
            st.rerun() # Rerun to load the new view

        st.sidebar.divider()

        # Logout button
        if st.sidebar.button("Logout"):
            # Clear relevant session state keys on logout
            keys_to_clear = ['authenticated', 'current_user', 'current_view', 'current_emotion_id', 'emotion_analysis_context']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_view = "login" # Go back to login
            st.rerun()

def render_main_dashboard():
    """Displays the main dashboard view."""
    st.title("🌱 Your Growth Dashboard")
    user_id = st.session_state.current_user
    emotion_history = get_user_emotion_history(user_id)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### Welcome back, {user_id}!")
        st.markdown("Here's a summary of your recent journey. Use the journal to capture your feelings and insights.")
    with col2:
        st.write("") # Spacer
        if st.button("📝 New Journal Entry", type="primary", use_container_width=True):
            st.session_state.current_view = "journal"
            # Clear any previous emotion context when starting fresh
            if 'current_emotion_id' in st.session_state: del st.session_state['current_emotion_id']
            if 'emotion_analysis_context' in st.session_state: del st.session_state['emotion_analysis_context']
            st.rerun()

    st.divider()

    if not emotion_history:
        st.info("You haven't added any journal entries yet. Click 'New Journal Entry' to get started!")
    else:
        st.subheader("Recent Emotional Journey")

        # Sort entries by timestamp (newest first) and display the top 3-5
        sorted_history = sorted(emotion_history, key=lambda x: x.get('timestamp', ''), reverse=True)
        for entry in sorted_history[:5]:
            entry_id = entry.get('id')
            timestamp = entry.get('timestamp', 'Unknown Time')
            try:
                 dt_obj = datetime.datetime.fromisoformat(timestamp)
                 display_time = dt_obj.strftime("%Y-%m-%d %H:%M") # Format timestamp nicely
            except:
                 display_time = timestamp[:16] # Fallback formatting

            analysis = entry.get('analysis', {})
            primary_emotion = analysis.get('primary_emotion', 'Entry')
            intensity = analysis.get('intensity', None)
            title = f"**{primary_emotion}**{' (Intensity: ' + str(intensity) + ')' if intensity else ''} - {display_time}"

            with st.expander(title, expanded=False):
                st.caption("Journal Entry:")
                st.write(entry.get('journal_entry', 'No content.'))

                # Display key analysis points briefly
                if analysis and 'error' not in analysis:
                    st.caption("Analysis Snippet:")
                    if 'triggers' in analysis and analysis['triggers']:
                        st.write(f"Potential Triggers: {', '.join(analysis['triggers'][:2])}{'...' if len(analysis['triggers']) > 2 else ''}")
                    if 'growth_opportunities' in analysis and analysis['growth_opportunities']:
                         st.write(f"Growth Areas: {', '.join(analysis['growth_opportunities'][:2])}{'...' if len(analysis['growth_opportunities']) > 2 else ''}")
                elif 'error' in analysis:
                    st.warning(f"Analysis Error: {analysis['error']}")
                elif 'raw_response' in analysis:
                     st.warning("Analysis generated raw text, not structured data.")

                if st.button("View Full Analysis & Actions", key=f"view_{entry_id}"):
                    st.session_state.current_emotion_id = entry_id
                    st.session_state.current_view = "emotion_analysis" # Navigate to the detailed analysis view
                    st.rerun()

def render_journal_page():
    """Displays the page for creating a new journal entry."""
    st.title("📝 Emotional Journal")
    st.write("Take a moment to reflect. What are you feeling right now? What happened? Be as detailed as you like.")

    # Use a form to handle the text area and button together
    with st.form("journal_entry_form"):
        journal_entry = st.text_area("Your thoughts and feelings:", height=250, key="journal_input",
                                     placeholder="e.g., Today I felt overwhelmed during the team meeting when...")
        submitted = st.form_submit_button("Analyze My Emotions", type="primary")

        if submitted:
            if not journal_entry or len(journal_entry.strip()) < 10:
                st.warning("Please write a bit more in your journal entry before analyzing.")
            else:
                with st.spinner("Analyzing your emotions... This might take a moment."):
                    analysis = analyze_emotion(journal_entry.strip()) # Analyze the stripped entry

                    # Save the entry and the analysis result
                    emotion_id = save_emotion_entry(st.session_state.current_user, {
                        'journal_entry': journal_entry.strip(),
                        'analysis': analysis # Store whatever the LLM returned (might be analysis, error, or raw)
                    })

                    st.session_state.current_emotion_id = emotion_id
                    st.session_state.current_view = "emotion_analysis" # Go to analysis view
                    st.rerun() # Rerun to show the analysis page

    st.divider()
    st.subheader("Past Journal Entries")

    history = get_user_emotion_history(st.session_state.current_user)
    if not history:
        st.info("No past entries yet.")
    else:
        # Display past entries similar to dashboard but maybe more compact
        sorted_history = sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)
        for entry in sorted_history:
             entry_id = entry.get('id')
             timestamp = entry.get('timestamp', 'Unknown Time')
             try:
                 dt_obj = datetime.datetime.fromisoformat(timestamp)
                 display_time = dt_obj.strftime("%Y-%m-%d %H:%M")
             except:
                 display_time = timestamp[:16]

             analysis = entry.get('analysis', {})
             primary_emotion = analysis.get('primary_emotion', 'Entry')
             preview_text = entry.get('journal_entry', '')[:100] + ('...' if len(entry.get('journal_entry', '')) > 100 else '')


             col1, col2 = st.columns([3, 1])
             with col1:
                 st.write(f"**{primary_emotion}** ({display_time}): {preview_text}")
             with col2:
                 if st.button("View Details", key=f"list_{entry_id}", use_container_width=True):
                     st.session_state.current_emotion_id = entry_id
                     st.session_state.current_view = "emotion_analysis"
                     st.rerun()
             st.markdown("---") # Add a small separator

def render_emotion_analysis(emotion_id: str):
    """Displays the detailed analysis of a specific emotion entry."""
    st.title("🧠 Emotion Analysis & Actions")

    user_id = st.session_state.current_user
    emotion_data = get_emotion_entry(user_id, emotion_id)

    if not emotion_data:
        st.error("Error: Could not load the selected journal entry.")
        if st.button("Back to Journal"):
             st.session_state.current_view = "journal"
             if 'current_emotion_id' in st.session_state: del st.session_state['current_emotion_id']
             st.rerun()
        return

    st.subheader("Your Journal Entry")
    st.markdown(f"> {emotion_data.get('journal_entry', 'Entry content not found.')}") # Display entry in blockquote
    st.caption(f"Entry logged on: {emotion_data.get('timestamp', 'Unknown')[:19].replace('T', ' ')}")


    st.divider()
    st.subheader("AI-Powered Analysis")
    analysis = emotion_data.get('analysis', {})

    # Check for analysis errors or raw response
    if not analysis:
         st.warning("No analysis data found for this entry.")
    elif 'error' in analysis:
        st.error(f"Analysis failed: {analysis['error']}")
        # Option to retry analysis?
        # if st.button("Retry Analysis"): ...
    elif 'raw_response' in analysis and 'warning' in analysis:
         st.warning(f"Analysis incomplete ({analysis['warning']}). Showing raw response:")
         st.text(analysis['raw_response'])
    elif 'raw_response' in analysis: # Handle cases where only raw response exists without specific warning
         st.warning("Analysis returned raw text instead of structured data:")
         st.text(analysis['raw_response'])
    else:
        # Display structured analysis if successful
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Primary Emotion")
            emotion = analysis.get('primary_emotion', 'N/A')
            st.info(f"**{emotion.capitalize()}**")
            if 'intensity' in analysis and analysis['intensity'] is not None:
                 try:
                    intensity_val = int(analysis['intensity'])
                    st.metric(label="Intensity (1-10)", value=intensity_val)
                    # st.progress(min(intensity_val / 10.0, 1.0)) # Alternative display
                 except (ValueError, TypeError):
                      st.write(f"Intensity: {analysis['intensity']} (could not parse as number)")

        with col2:
            triggers = analysis.get('triggers')
            if triggers and isinstance(triggers, list):
                 st.markdown("#### Potential Triggers")
                 if triggers:
                      for trigger in triggers:
                          st.write(f"- {trigger}")
                 else:
                     st.write("_None identified._")

        growth_opps = analysis.get('growth_opportunities')
        if growth_opps and isinstance(growth_opps, list):
             st.markdown("#### Growth Opportunities")
             if growth_opps:
                 for opp in growth_opps:
                     st.write(f"- {opp}")
             else:
                 st.write("_None identified._")

    st.divider()
    st.subheader("Next Steps")

    col1, col2 = st.columns(2)

    # --- Growth Plan Section ---
    with col1:
        st.markdown("##### **Personal Growth Plan**")
        growth_plan = get_growth_plan(user_id, emotion_id)
        if growth_plan and 'error' not in growth_plan:
            st.success("Growth plan already generated.")
            plan_title = growth_plan.get('title', 'Your Plan')
            if st.button(f"View '{plan_title}'", key="view_plan_button", use_container_width=True):
                st.session_state.current_view = "growth_plan"
                st.rerun()
        elif growth_plan and 'error' in growth_plan:
             st.warning(f"Previous plan generation failed: {growth_plan['error']}")
             if st.button("💡 Try Generating Plan Again", key="retry_plan_button", type="primary", use_container_width=True, disabled=('error' in analysis or 'raw_response' in analysis)):
                 with st.spinner("Creating your growth plan..."):
                     # Ensure analysis is valid before generating plan
                     if 'error' not in analysis and 'raw_response' not in analysis:
                         plan = generate_growth_plan(analysis, st.session_state.user_db.get(user_id, {}).get('goals'))
                         save_growth_plan(user_id, emotion_id, plan)
                         if 'error' not in plan:
                             st.session_state.current_view = "growth_plan"
                         st.rerun()
                     else:
                         st.error("Cannot generate plan from failed or incomplete analysis.")
        else:
            # Button to generate plan if none exists and analysis is valid
            if st.button("💡 Create Growth Plan", key="create_plan_button", type="primary", use_container_width=True, disabled=('error' in analysis or 'raw_response' in analysis)):
                with st.spinner("Creating your growth plan..."):
                    if 'error' not in analysis and 'raw_response' not in analysis:
                         plan = generate_growth_plan(analysis, st.session_state.user_db.get(user_id, {}).get('goals'))
                         save_growth_plan(user_id, emotion_id, plan)
                         if 'error' not in plan:
                              st.session_state.current_view = "growth_plan"
                         st.rerun()
                    else:
                         st.error("Cannot generate plan from failed or incomplete analysis.")
            elif ('error' in analysis or 'raw_response' in analysis):
                 st.caption("Cannot generate plan until analysis is successful.")

    # --- Resource Section ---
    with col2:
        st.markdown("##### **Relevant Resources**")
        resources = get_resources(user_id, emotion_id)
        if resources and 'error' not in resources:
             st.success("Resources found for this entry.")
             if st.button("📚 View Resources", key="view_res_button", use_container_width=True):
                 st.session_state.current_view = "view_resources"
                 st.rerun()
        elif resources and 'error' in resources:
             st.warning(f"Previous resource search failed: {resources['error']}")
             if st.button("🔎 Try Finding Resources Again", key="retry_res_button", type="primary", use_container_width=True, disabled=('error' in analysis or 'raw_response' in analysis)):
                 # Ensure analysis is valid before searching
                 if 'error' not in analysis and 'raw_response' not in analysis:
                     st.session_state.emotion_analysis_context = analysis
                     st.session_state.current_view = "search_resources" # Go to search trigger page
                     st.rerun()
                 else:
                     st.error("Cannot search resources based on failed or incomplete analysis.")
        else:
             # Button to initiate resource search if none exist and analysis is valid
             if st.button("🔎 Find Related Resources", key="find_res_button", type="primary", use_container_width=True, disabled=('error' in analysis or 'raw_response' in analysis)):
                  if 'error' not in analysis and 'raw_response' not in analysis:
                      st.session_state.emotion_analysis_context = analysis # Store context for search page
                      st.session_state.current_view = "search_resources" # Go to search trigger page
                      st.rerun()
                  else:
                      st.error("Cannot search resources based on failed or incomplete analysis.")
             elif ('error' in analysis or 'raw_response' in analysis):
                 st.caption("Cannot find resources until analysis is successful.")


def render_growth_plan(emotion_id: str):
    """Displays the generated growth plan."""
    st.title("🚀 Your Growth Plan")

    user_id = st.session_state.current_user
    plan = get_growth_plan(user_id, emotion_id)
    emotion_data = get_emotion_entry(user_id, emotion_id) # Get original entry for context

    if not plan:
        st.error("Growth plan not found or not yet generated for this entry.")
        if st.button("Back to Analysis"):
            st.session_state.current_view = "emotion_analysis"
            st.rerun()
        return

    if emotion_data and 'analysis' in emotion_data:
         analysis = emotion_data['analysis']
         emotion = analysis.get('primary_emotion', 'the emotion')
         st.caption(f"This plan addresses your experience with: **{emotion}**")

    st.divider()

    # Check for errors in the plan data itself
    if 'error' in plan:
         st.error(f"Failed to generate growth plan: {plan['error']}")
    elif 'raw_response' in plan:
          st.warning("Growth plan generation returned raw text, not structured data:")
          st.text(plan['raw_response'])
    else:
        # Display the structured plan
        st.header(plan.get('title', 'Personal Growth Plan'))

        steps = plan.get('steps')
        if steps and isinstance(steps, list):
            st.subheader("Action Steps")
            for i, step in enumerate(steps, 1):
                 step_title = step.get('title', f'Step {i}')
                 step_desc = step.get('description', 'No description provided.')
                 with st.expander(f"**Step {i}: {step_title}**", expanded=True):
                      st.write(step_desc)
        else:
             st.warning("No action steps found in the generated plan.")


        outcomes = plan.get('expected_outcomes')
        if outcomes and isinstance(outcomes, list):
            st.subheader("Expected Outcomes")
            for outcome in outcomes:
                st.write(f"- {outcome}")
        else:
             st.info("No specific outcomes listed in the plan.")


    st.divider()
    if st.button("⬅️ Back to Analysis"):
        st.session_state.current_view = "emotion_analysis"
        st.rerun()

def render_search_resources(emotion_id: str):
    """Handles the process of initiating web search and crawl."""
    st.title("🔎 Finding Growth Resources")

    user_id = st.session_state.current_user
    # Retrieve the analysis context stored in session state
    analysis = st.session_state.get('emotion_analysis_context')

    if not analysis:
         # Attempt to fetch it from the emotion entry if context is missing (e.g., page reload)
         emotion_data = get_emotion_entry(user_id, emotion_id)
         if emotion_data and 'analysis' in emotion_data and 'error' not in emotion_data['analysis']:
             analysis = emotion_data['analysis']
             st.session_state.emotion_analysis_context = analysis # Restore context
         else:
              st.error("Cannot initiate search: Emotion analysis data is missing or invalid.")
              if st.button("Back to Analysis"):
                  st.session_state.current_view = "emotion_analysis"
                  if 'emotion_analysis_context' in st.session_state: del st.session_state['emotion_analysis_context']
                  st.rerun()
              return

    emotion = analysis.get('primary_emotion', 'your current emotion')
    st.write(f"Pathly will now search the web for resources related to coping with **{emotion}** based on your analysis.")
    st.info("This involves searching, crawling relevant pages, and processing the content. It may take a minute or two.")

    if st.button("🚀 Start Resource Search", type="primary"):
        # Run the async search and crawl process
        with st.spinner("Searching, crawling, and processing resources... Please wait."):
            try:
                # Get or create an event loop for the async operations
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError: # 'RuntimeError: There is no current event loop...'
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Run the async function and wait for completion
                crawled_chunks = loop.run_until_complete(search_and_crawl_resources(analysis))
                # loop.close() # Closing loop might cause issues if other async tasks are needed later

                if crawled_chunks:
                    st.success("Found and processed relevant content. Now synthesizing...")
                    # Synthesize the found resources using the LLM
                    with st.spinner("Synthesizing key insights and exercises..."):
                         resource_data = synthesize_resources(
                             analysis,
                             get_growth_plan(user_id, emotion_id), # Pass existing plan if available
                             crawled_chunks # Pass the text chunks added to the DB
                         )
                         # Save the synthesized resources
                         save_resource(user_id, emotion_id, resource_data)
                         st.success("Resources synthesized and saved!")
                         # Navigate to the view resources page
                         st.session_state.current_view = "view_resources"
                         if 'emotion_analysis_context' in st.session_state: del st.session_state['emotion_analysis_context'] # Clean up context
                         st.rerun()
                else:
                    st.warning("Search completed, but no suitable resources were found or processed after crawling.")
                    # Save an empty/error state for resources? Or just let user retry?
                    save_resource(user_id, emotion_id, {"error": "No suitable resources found after search and crawl."})
                    st.session_state.current_view = "emotion_analysis" # Go back to analysis
                    if 'emotion_analysis_context' in st.session_state: del st.session_state['emotion_analysis_context']
                    st.rerun()

            except Exception as e:
                st.error(f"An unexpected error occurred during the resource search process: {e}")
                # Log detailed error if needed
                # save_resource(user_id, emotion_id, {"error": f"Unexpected error: {e}"}) # Save error state
                # st.session_state.current_view = "emotion_analysis" # Go back
                # if 'emotion_analysis_context' in st.session_state: del st.session_state['emotion_analysis_context']
                # st.rerun()


    st.divider()
    if st.button("Cancel and Go Back"):
        st.session_state.current_view = "emotion_analysis"
        if 'emotion_analysis_context' in st.session_state: del st.session_state['emotion_analysis_context'] # Clean up
        st.rerun()


def render_resources():
    """Displays the general resource search interface and links to specific resources."""
    st.title("📚 Growth Resources Library")
    st.write("Search the collected resources or view resources linked to specific journal entries.")

    user_id = st.session_state.current_user
    emotion_id = st.session_state.get('current_emotion_id') # Check if navigating from a specific entry

    # --- Section for Resources linked to the CURRENT Emotion ID (if any) ---
    if emotion_id:
        emotion_data = get_emotion_entry(user_id, emotion_id)
        if emotion_data:
             analysis = emotion_data.get('analysis', {})
             primary_emotion = analysis.get('primary_emotion', 'selected entry')
             st.subheader(f"Resources for: '{primary_emotion}' Entry")
             resources = get_resources(user_id, emotion_id)

             if resources and 'error' not in resources:
                  st.success("Synthesized resources are available for this entry.")
                  if st.button(f"View Resources for '{primary_emotion}'", key="view_specific_res"):
                      st.session_state.current_view = "view_resources" # Should already be set if coming from analysis, but safe fallback
                      st.rerun()
             elif resources and 'error' in resources:
                  st.warning(f"Resource finding failed previously: {resources['error']}")
                  if st.button("Try Finding Resources Again", key="retry_specific_res"):
                      if 'error' not in analysis and 'raw_response' not in analysis:
                          st.session_state.emotion_analysis_context = analysis
                          st.session_state.current_view = "search_resources"
                          st.rerun()
                      else:
                          st.error("Cannot search based on failed/incomplete analysis.")
             else:
                  st.info("No resources have been generated for this specific entry yet.")
                  if st.button("Find Resources Now", key="find_specific_res"):
                       if 'error' not in analysis and 'raw_response' not in analysis:
                           st.session_state.emotion_analysis_context = analysis
                           st.session_state.current_view = "search_resources"
                           st.rerun()
                       else:
                           st.error("Cannot search based on failed/incomplete analysis.")
             st.divider()


    # --- General Search Section ---
    st.subheader("Search All Collected Resources")
    st.caption(f"Your library currently contains {st.session_state.faiss_index.ntotal} indexed resource chunks.")

    query = st.text_input("Enter keywords to search (e.g., 'mindfulness', 'anxiety coping')", key="resource_search_query")

    if st.button("Search Library", key="search_library_button", disabled=not query):
        with st.spinner("Searching indexed resources..."):
            results = query_vector_database(query.strip(), n_results=5) # Get top 5 results

            if results:
                st.success(f"Found {len(results)} relevant resource snippets:")
                for i, result_meta in enumerate(results):
                     text = result_meta.get('text', 'No text found')
                     source = result_meta.get('source', 'Unknown source')
                     title = result_meta.get('title', 'Unknown title')
                     distance = result_meta.get('distance', None)

                     expander_title = f"Result {i+1}: {title[:60]}{'...' if len(title)>60 else ''}"
                     if distance is not None:
                          expander_title += f" (Relevance Score: {distance:.2f})" # Lower is more relevant for L2

                     with st.expander(expander_title, expanded=(i==0)): # Expand first result
                          st.markdown(text)
                          st.caption(f"Source: [{source}]({source})") # Make source a clickable link
            else:
                st.warning("No matching resources found in your library for that query.")
    elif st.session_state.faiss_index.ntotal == 0:
         st.info("Your resource library is currently empty. Analyze journal entries and find resources to build it up!")


def render_view_resources():
    """Displays the synthesized resources for a specific emotion entry."""
    st.title("📚 Synthesized Resources")

    user_id = st.session_state.current_user
    emotion_id = st.session_state.get('current_emotion_id')

    if not emotion_id:
        st.error("No specific emotion entry selected. Cannot display resources.")
        if st.button("Go to Resource Library"):
            st.session_state.current_view = "resources" # Navigate to the general resource page
            st.rerun()
        return

    resources = get_resources(user_id, emotion_id)
    emotion_data = get_emotion_entry(user_id, emotion_id) # Get original entry for context

    if emotion_data and 'analysis' in emotion_data:
        emotion = emotion_data['analysis'].get('primary_emotion', 'the selected emotion')
        st.caption(f"Resources related to your journal entry about: **{emotion}**")

    st.divider()

    if not resources:
        st.warning("No synthesized resources found for this entry.")
        # Offer to find resources again
        if st.button("Try Finding Resources"):
            analysis = emotion_data.get('analysis') if emotion_data else None
            if analysis and 'error' not in analysis and 'raw_response' not in analysis:
                st.session_state.emotion_analysis_context = analysis
                st.session_state.current_view = "search_resources"
                st.rerun()
            else:
                 st.error("Cannot search resources without valid analysis data.")
        if st.button("Back to Analysis"):
             st.session_state.current_view = "emotion_analysis"
             st.rerun()
        return

    # Check for errors or raw responses in the resource data
    if 'error' in resources:
        st.error(f"Failed to synthesize resources: {resources['error']}")
    elif 'raw_response' in resources:
        st.warning("Resource synthesis returned raw text:")
        st.text(resources['raw_response'])
    else:
        # Display structured resources
        insights = resources.get('key_insights')
        if insights and isinstance(insights, list):
            st.subheader("💡 Key Insights")
            for insight in insights:
                st.write(f"- {insight}")
        else:
             st.info("No key insights were synthesized.")

        exercises = resources.get('practical_exercises')
        if exercises and isinstance(exercises, list):
            st.subheader("🧘 Practical Exercises")
            for i, exercise in enumerate(exercises, 1):
                st.write(f"{i}. {exercise}")
        else:
            st.info("No practical exercises were synthesized.")

        readings = resources.get('recommended_readings')
        if readings and isinstance(readings, list):
            st.subheader("📖 Recommended Reading/Viewing")
            for reading in readings:
                title = reading.get('title')
                desc = reading.get('description')
                if title:
                    st.markdown(f"**{title}**")
                    if desc:
                        st.write(desc)
                elif desc: # Handle case where only description exists
                     st.write(desc)

        else:
            st.info("No specific readings were recommended.")

    st.divider()
    if st.button("⬅️ Back to Analysis"):
        st.session_state.current_view = "emotion_analysis"
        st.rerun()


# Main Application Logic
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Pathly - Emotion to Action", layout="wide", initial_sidebar_state="auto")

    # Initialize session state databases and variables
    init_state()

    # Initialize session state variables if they don't exist
    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    if 'current_user' not in st.session_state: st.session_state.current_user = None
    # Default view is login if not authenticated, main dashboard otherwise
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "login" if not st.session_state.authenticated else "main"
    if 'current_emotion_id' not in st.session_state: st.session_state.current_emotion_id = None
    if 'emotion_analysis_context' not in st.session_state: st.session_state.emotion_analysis_context = None

    # --- Routing ---
    if not st.session_state.authenticated:
        # If not authenticated, force the view to login
        st.session_state.current_view = "login"
        render_login_page()
    else:
        # If authenticated, show sidebar and route to the correct view
        render_sidebar() # Render sidebar navigation

        # Get current view and context ID
        view = st.session_state.current_view
        emotion_id = st.session_state.get('current_emotion_id')

        # Main content area based on the current view
        if view == "main":
            render_main_dashboard()
        elif view == "journal":
            render_journal_page()
        elif view == "emotion_analysis":
            if emotion_id:
                render_emotion_analysis(emotion_id)
            else:
                st.warning("No emotion entry selected. Please select one from the Journal or Dashboard.")
                st.session_state.current_view = "journal" # Redirect to journal
                st.rerun()
        elif view == "growth_plan":
            if emotion_id:
                render_growth_plan(emotion_id)
            else:
                st.warning("No emotion entry selected for viewing growth plan.")
                st.session_state.current_view = "journal" # Redirect
                st.rerun()
        elif view == "search_resources":
             if emotion_id:
                 render_search_resources(emotion_id)
             else:
                 st.warning("No emotion entry selected to base resource search on.")
                 st.session_state.current_view = "journal" # Redirect
                 st.rerun()
        elif view == "resources": # General resource library page
            render_resources()
        elif view == "view_resources": # View synthesized resources for a specific entry
            if emotion_id:
                 render_view_resources()
            else:
                 st.warning("No specific emotion entry selected to view resources.")
                 st.session_state.current_view = "resources" # Redirect to general library
                 st.rerun()
        else:
            # Fallback to dashboard if view is unknown or state is inconsistent
            st.error(f"Unknown view: {view}. Returning to dashboard.")
            st.session_state.current_view = "main"
            if 'current_emotion_id' in st.session_state: del st.session_state['current_emotion_id'] # Clear context
            st.rerun()

# Standard Python entry point
if __name__ == "__main__":
    main()

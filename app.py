import asyncio
import os
import json
import datetime
import tempfile
from typing import Dict, List, Optional
import uuid
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import streamlit as st
import pandas as pd
import ollama
from duckduckgo_search import DDGS
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# System prompts for different app functionalities
EMOTION_ANALYSIS_PROMPT = """
You are an AI assistant specializing in emotional analysis and personal growth.
Analyze the emotional content in the user's journal entry, focusing on:
1. Identifying primary emotions (joy, sadness, anger, fear, etc.)
2. Recognizing emotion intensity (1-10 scale)
3. Detecting emotional patterns and triggers
4. Observing potential growth opportunities

Format your response as a JSON object with these keys:
- primary_emotion: The dominant emotion expressed
- intensity: A numerical value from 1-10
- triggers: A list of potential triggers identified
- patterns: Any emotional patterns detected
- growth_opportunities: 3 specific ways the user might grow from this experience
- action_steps: 3 suggested concrete actions the user could take

Base your analysis solely on the provided journal entry. Be empathetic yet objective in your assessment.
"""

GROWTH_PLAN_PROMPT = """
You are an AI coach specializing in transforming emotional experiences into personal growth opportunities.
Based on the user's emotional profile and goals, create a structured growth plan that includes:

1. Short-term actions (next 24-48 hours)
2. Medium-term practices (1-2 weeks)
3. Long-term behavior changes (1-3 months)

Format your response as a JSON object with these keys:
- short_term_actions: List of 3 immediate actions
- medium_term_practices: List of 3 practices to develop over weeks
- long_term_changes: List of 3 behavior patterns to cultivate
- reflection_prompts: List of 3 questions for daily reflection
- success_metrics: List of 3 ways to measure progress

Make all suggestions specific, actionable, and tailored to the user's emotional state and goals.
"""

RESOURCE_SYNTHESIS_PROMPT = """
You are an AI assistant specializing in synthesizing web resources for emotional growth.
Based on the user's emotional state and growth goals, synthesize the provided web content into actionable resources.

Format your response as a JSON object with these keys:
- key_insights: List of 3-5 most relevant insights from the resources
- practical_exercises: List of 2-3 practical exercises mentioned in the resources
- recommended_readings: List of any specific books, articles, or resources mentioned
- expert_advice: Summary of expert advice found in the resources
- action_plan: 3 steps to implement these insights based on the user's emotional state

Maintain a compassionate, supportive tone while focusing on factual, evidence-based information.
"""

COMMUNITY_SUGGESTION_PROMPT = """
You are an AI community facilitator specializing in emotional growth.
Based on the user's emotional profile, goals, and growth plan, suggest relevant community resources:

1. Types of experiences that might benefit from community sharing
2. Community topics that align with user's growth areas
3. Potential community support needs

Format your response as a JSON object with these keys:
- sharing_opportunities: List of 3 aspects of the user's journey that could benefit from sharing
- recommended_topics: List of 3 community discussion topics aligned with user's growth areas
- support_needs: List of 3 ways community members might support this journey

Ensure all suggestions maintain user privacy while facilitating meaningful connections.
"""

# --- Database Functions ---
def initialize_databases():
    """Initialize the databases for storing user data, emotions, and community content."""
    # Use Streamlit session state for simple in-memory storage
    if 'user_db' not in st.session_state:
        st.session_state.user_db = {} # {username: {'password': '', 'joined_date': '', 'premium': False, 'streak': 0, 'points': 0, 'goals': {}}}

    if 'emotion_db' not in st.session_state:
        st.session_state.emotion_db = {} # {username: [{'id': '', 'timestamp': '', 'journal_entry': '', 'analysis': {}}]}

    if 'community_db' not in st.session_state:
        st.session_state.community_db = [] # [{'id': '', 'user_id': '', 'timestamp': '', 'title': '', 'content': '', 'likes': 0, 'comments': []}]

    if 'growth_plans_db' not in st.session_state:
        st.session_state.growth_plans_db = {} # {username: {emotion_id: {plan_data}}}

    if 'resource_db' not in st.session_state:
        st.session_state.resource_db = {} # {username: {emotion_id: {resource_data}}}

def get_vector_collection(collection_name="emotion_growth"):
    """Creates or retrieves a vector collection."""
    # Ensure Ollama URL is correct, especially if running in Docker
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_ef = OllamaEmbeddingFunction(
        url=f"{ollama_url}/api/embeddings",
        model_name="nomic-embed-text:latest", # Make sure this model is pulled in Ollama
    )

    # Create a persistent client (stores data on disk)
    # Ensure the directory exists and is writable
    db_path = "./emotion-action-db"
    os.makedirs(db_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=db_path, settings=Settings(anonymized_telemetry=False)
    )

    try:
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=ollama_ef,
            metadata={"hnsw:space": "cosine"}, # Use cosine distance
        )
        return collection, chroma_client
    except Exception as e:
        st.error(f"Error initializing vector database: {e}")
        st.warning("Vector database features (resource search) might be limited.")
        return None, None


def normalize_url(url):
    """Normalizes a URL to be used as a ChromaDB ID."""
    parsed = urlparse(url)
    # Keep scheme and netloc, replace path separators, limit length
    normalized_url = f"{parsed.scheme}_{parsed.netloc}{parsed.path}".replace("/", "_").replace("-", "_").replace(".", "_")
    return normalized_url[:250] # Limit length for ID compatibility


def save_user_data(user_id: str, data: Dict):
    """Save user data to the session state database."""
    if 'user_db' in st.session_state:
        st.session_state.user_db[user_id] = data

def save_emotion_entry(user_id: str, emotion_data: Dict):
    """Save an emotion entry to the session state database."""
    if user_id not in st.session_state.emotion_db:
        st.session_state.emotion_db[user_id] = []

    emotion_data['timestamp'] = datetime.datetime.now().isoformat()
    emotion_data['id'] = str(uuid.uuid4())

    st.session_state.emotion_db[user_id].append(emotion_data)
    return emotion_data['id']

def save_growth_plan(user_id: str, emotion_id: str, plan_data: Dict):
    """Save a growth plan to the session state database."""
    if user_id not in st.session_state.growth_plans_db:
        st.session_state.growth_plans_db[user_id] = {}

    st.session_state.growth_plans_db[user_id][emotion_id] = plan_data

def get_user_emotion_history(user_id: str) -> List[Dict]:
    """Get the emotion history for a user from session state."""
    if 'emotion_db' not in st.session_state or user_id not in st.session_state.emotion_db:
        return []
    return st.session_state.emotion_db[user_id]

def get_emotion_entry(user_id: str, emotion_id: str) -> Optional[Dict]:
    """Get a specific emotion entry by ID."""
    history = get_user_emotion_history(user_id)
    for entry in history:
        if entry.get('id') == emotion_id:
            return entry
    return None

def get_growth_plan(user_id: str, emotion_id: str) -> Optional[Dict]:
    """Get a specific growth plan from session state."""
    if 'growth_plans_db' not in st.session_state or user_id not in st.session_state.growth_plans_db:
        return None
    return st.session_state.growth_plans_db[user_id].get(emotion_id)

def save_community_post(user_id: str, post_data: Dict):
    """Save a community post to the session state database."""
    post_data['user_id'] = user_id
    post_data['timestamp'] = datetime.datetime.now().isoformat()
    post_data['id'] = str(uuid.uuid4())
    post_data['likes'] = 0
    post_data['comments'] = [] # format: {'id': '', 'user_id': '', 'comment': '', 'timestamp': ''}

    if 'community_db' not in st.session_state:
        st.session_state.community_db = []
    st.session_state.community_db.append(post_data)
    return post_data['id']

def get_community_posts(limit: int = 20) -> List[Dict]:
    """Get community posts, sorted by recency."""
    if 'community_db' not in st.session_state:
        return []
    posts = sorted(
        st.session_state.community_db,
        key=lambda x: x['timestamp'],
        reverse=True
    )
    return posts[:limit]

def add_comment_to_post(post_id: str, user_id: str, comment: str):
    """Add a comment to a community post in session state."""
    if 'community_db' in st.session_state:
        for post in st.session_state.community_db:
            if post['id'] == post_id:
                if 'comments' not in post:
                    post['comments'] = []
                post['comments'].append({
                    'user_id': user_id,
                    'comment': comment,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'id': str(uuid.uuid4())
                })
                break

def like_post(post_id: str, user_id: str):
    """Like a community post (simplified: just increments count)."""
    # In a real app, you'd track *who* liked it to prevent multiple likes.
    if 'community_db' in st.session_state:
        for post in st.session_state.community_db:
            if post['id'] == post_id:
                post['likes'] = post.get('likes', 0) + 1
                break

def save_resource(user_id: str, emotion_id: str, resource_data: Dict):
    """Save resources (synthesis results) for a specific emotion entry."""
    if user_id not in st.session_state.resource_db:
        st.session_state.resource_db[user_id] = {}

    st.session_state.resource_db[user_id][emotion_id] = resource_data

def get_resources(user_id: str, emotion_id: str) -> Optional[Dict]:
    """Get resources (synthesis results) for a specific emotion entry."""
    if 'resource_db' not in st.session_state or user_id not in st.session_state.resource_db:
        return None
    return st.session_state.resource_db[user_id].get(emotion_id)

# --- LLM Interaction Functions ---
def call_llm(prompt: str, system_prompt: str, user_input: str) -> Dict:
    """Call LLM with the given system prompt and user input."""
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"{prompt}\n\n{user_input}", # Combine prompt and input
        },
    ]

    try:
        # Ensure OLLAMA_HOST is set if running in Docker or remotely
        # ollama_client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        # response = ollama_client.chat(model="llama3:8b", messages=messages) # Use a specific model
        response = ollama.chat(model="llama3:8b", messages=messages)

        response_content = response['message']['content']

        # Attempt to find JSON within the response (LLMs sometimes add extra text)
        try:
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response_content[json_start:json_end]
                return json.loads(json_str)
            else:
                # If no JSON object found, return raw text
                st.warning("LLM response was not valid JSON. Displaying raw text.")
                return {"raw_response": response_content}
        except json.JSONDecodeError:
            st.warning("LLM response could not be parsed as JSON. Displaying raw text.")
            return {"raw_response": response_content}
        except Exception as e:
            st.error(f"Error parsing LLM response: {e}")
            return {"raw_response": response_content}

    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        return {"error": f"Failed to get response from LLM: {e}"}


def analyze_emotion(journal_entry: str) -> Dict:
    """Analyze the emotion content in a journal entry."""
    return call_llm(
        prompt="Analyze this journal entry for emotional content based on the specified JSON format:",
        system_prompt=EMOTION_ANALYSIS_PROMPT,
        user_input=journal_entry
    )

def generate_growth_plan(emotion_analysis: Dict, user_goals: Dict) -> Dict:
    """Generate a growth plan based on emotion analysis and user goals."""
    input_data = {
        "emotion_analysis": emotion_analysis,
        "user_goals": user_goals if user_goals else {"general_goal": "Improve emotional regulation and well-being."} # Provide default if no goals set
    }

    return call_llm(
        prompt="Create a growth plan based on this emotional analysis and goals, following the specified JSON format:",
        system_prompt=GROWTH_PLAN_PROMPT,
        user_input=json.dumps(input_data, indent=2)
    )

def synthesize_resources(emotion_analysis: Dict, growth_plan: Optional[Dict], web_content: List[str]) -> Dict:
    """Synthesize web resources based on emotion analysis and growth plan."""
    if not web_content:
        return {"error": "No web content provided for synthesis."}

    input_data = {
        "emotion_analysis": emotion_analysis,
        "growth_plan": growth_plan if growth_plan else "No specific growth plan available.",
        "web_content": "\n---\n".join(web_content) # Join content snippets
    }

    return call_llm(
        prompt="Synthesize these web resources for emotional growth based on the user's profile, following the specified JSON format:",
        system_prompt=RESOURCE_SYNTHESIS_PROMPT,
        user_input=json.dumps(input_data, indent=2)
    )

def get_community_suggestions(emotion_analysis: Dict, growth_plan: Optional[Dict]) -> Dict:
    """Get community interaction suggestions based on emotion analysis and growth plan."""
    input_data = {
        "emotion_analysis": emotion_analysis,
        "growth_plan": growth_plan if growth_plan else "No specific growth plan available."
    }

    return call_llm(
        prompt="Suggest community interactions based on this profile, following the specified JSON format:",
        system_prompt=COMMUNITY_SUGGESTION_PROMPT,
        user_input=json.dumps(input_data, indent=2)
    )

# --- Web Search and Crawl Functions ---
def get_web_urls(search_term: str, num_results: int = 5) -> List[str]:
    """Performs a web search and returns filtered URLs."""
    try:
        # Add emotional growth related terms to search query
        enhanced_search = f"{search_term} emotional regulation coping strategies personal development"

        # Exclude low-quality or irrelevant domains
        discard_sites = ["youtube.com", "amazon.com", "pinterest.com", "facebook.com", "instagram.com", "twitter.com", "tiktok.com", "reddit.com/r/"]
        for site in discard_sites:
            enhanced_search += f" -site:{site}"

        print(f"Searching DuckDuckGo for: {enhanced_search}") # Log search query
        results = DDGS().text(enhanced_search, max_results=num_results * 2) # Get more initially
        urls = [result["href"] for result in results if result.get("href")]
        print(f"Initial URLs found: {urls}")

        # Basic filtering (e.g., remove PDFs, duplicates)
        filtered_urls = []
        seen_domains = set()
        for url in urls:
            if url.lower().endswith(".pdf"):
                continue
            domain = urlparse(url).netloc
            if domain not in seen_domains:
                filtered_urls.append(url)
                seen_domains.add(domain)

        print(f"Filtered URLs (before robots check): {filtered_urls[:num_results]}")
        # Check robots.txt before returning
        allowed_urls = check_robots_txt(filtered_urls[:num_results]) # Limit to desired number after filtering
        print(f"Allowed URLs (after robots check): {allowed_urls}")
        return allowed_urls

    except Exception as e:
        error_msg = f"❌ Failed to fetch results from the web: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return []

def check_robots_txt(urls: List[str]) -> List[str]:
    """Checks robots.txt files to determine which URLs are allowed to be crawled."""
    allowed_urls = []
    rp_cache = {} # Cache RobotFileParser instances per domain

    for url in urls:
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme
        netloc = parsed_url.netloc
        if not scheme or not netloc:
            print(f"Skipping invalid URL for robots check: {url}")
            continue

        robots_url = f"{scheme}://{netloc}/robots.txt"

        rp = rp_cache.get(robots_url)
        if rp is None:
            try:
                rp = RobotFileParser(robots_url)
                rp.read()
                rp_cache[robots_url] = rp # Cache the parser
                print(f"Read robots.txt for {netloc}")
            except Exception as e:
                print(f"Could not read or parse robots.txt for {netloc}: {e}. Assuming allowed.")
                # If robots.txt is missing or there's an error, cautiously assume allowed for now
                allowed_urls.append(url)
                rp_cache[robots_url] = "error" # Mark as errored to avoid retrying
                continue
        elif rp == "error": # Skip if previously errored
             allowed_urls.append(url)
             continue


        try:
             # Use a common user agent string or the one used by the crawler
            user_agent = "crawl4ai/python" # Or match the crawler's user agent
            if rp.can_fetch(user_agent, url):
                allowed_urls.append(url)
                print(f"Allowed crawling: {url}")
            else:
                print(f"Disallowed by robots.txt: {url}")
        except Exception as e:
            print(f"Error checking robots.txt permission for {url}: {e}. Assuming allowed.")
            allowed_urls.append(url)


    return allowed_urls


async def crawl_webpages(urls: List[str], query: str):
    """Asynchronously crawls multiple webpages and extracts relevant content."""
    if not urls:
        return []

    # Configure BM25 filter based on the search query
    bm25_filter = BM25ContentFilter(user_query=query, bm25_threshold=1.0) # Adjust threshold as needed
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)

    # Define crawler configuration
    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header", "form", "img", "a", "script", "style", "aside"], # More aggressive exclusions
        only_text=True,
        exclude_social_media_links=True,
        keep_data_attributes=False,
        cache_mode=CacheMode.BYPASS, # Use CacheMode.NORMAL for faster re-runs during dev
        remove_overlay_elements=True,
        user_agent="Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)", # Be a good citizen
        page_timeout=25000,  # Increased timeout (25 seconds)
        wait_for_network_idle=True, # Wait longer for dynamic content
        network_idle_timeout=5000, # 5 seconds idle
    )

    # Configure browser settings (headless is crucial for servers)
    browser_config = BrowserConfig(
        headless=True,
        text_mode=True, # Helps with some anti-scraping
        light_mode=True # Can improve performance
    )

    results = []
    print(f"Starting crawl for {len(urls)} URLs...")
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Use await crawler.arun_many for concurrency
            crawl_results = await crawler.arun_many(urls, config=crawler_config)
            results.extend(crawl_results)
            print(f"Crawling finished. Got {len(results)} results.")
    except Exception as e:
        st.error(f"An error occurred during web crawling: {e}")
        print(f"Crawling error: {e}")

    # Filter results that actually have content
    valid_results = [res for res in results if res and res.markdown_v2 and res.markdown_v2.fit_markdown]
    print(f"Valid crawl results with markdown: {len(valid_results)}")
    return valid_results


def add_to_vector_database(results, collection_name="web_resources"):
    """Adds crawl results to a vector database for semantic search."""
    collection, client = get_vector_collection(collection_name)
    if not collection:
        st.error("Vector database collection not available. Cannot add resources.")
        return [] # Return empty list indicating no documents were added

    documents, metadatas, ids = [], [], []

    for result in results:
        # Ensure we have valid markdown content
        if not result or not result.markdown_v2 or not result.markdown_v2.fit_markdown:
            print(f"Skipping result for {result.url if result else 'N/A'} due to missing markdown.")
            continue

        content = result.markdown_v2.fit_markdown
        if not content.strip(): # Skip if content is empty after stripping whitespace
            print(f"Skipping result for {result.url} due to empty content.")
            continue

        # Normalize URL for use as part of ID
        normalized_url = normalize_url(result.url)

        # Simple chunking (split by paragraphs, then by sentences if needed, then fixed size)
        # Goal: Meaningful chunks, not too large for embedding model context window
        chunks = []
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if len(para) < 50: continue # Skip very short paragraphs
            if len(para) <= 1000: # Max chunk size (adjust based on embedding model)
                 if para.strip(): chunks.append(para.strip())
            else:
                # Further split large paragraphs (e.g., by sentences or fixed length)
                # Basic fixed length split for simplicity here
                for i in range(0, len(para), 800): # Overlapping slightly
                    chunk = para[i:i+1000].strip()
                    if chunk: chunks.append(chunk)


        print(f"Generated {len(chunks)} chunks for URL: {result.url}")

        for idx, chunk in enumerate(chunks):
            # Ensure chunk is not just whitespace
            if chunk:
                doc_id = f"{normalized_url}_{idx}"
                # Check length just in case
                if len(doc_id) > 512: # Check ChromaDB ID length limit (adjust if needed)
                    doc_id = doc_id[:512]

                documents.append(chunk)
                metadatas.append({"source": result.url, "title": result.title or "No Title"}) # Add title if available
                ids.append(doc_id)


    if documents:
        print(f"Adding {len(documents)} documents to vector collection '{collection_name}'...")
        try:
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            print(f"Successfully added {len(documents)} documents.")
            # If using PersistentClient, ensure data is saved (though upsert usually handles this)
            # client.persist() # Might not be necessary depending on client version/config
        except Exception as e:
            st.error(f"Failed to add documents to vector database: {e}")
            print(f"ChromaDB Upsert Error: {e}")
            # Optionally, print details of failed documents if possible
            # for i in range(len(documents)):
            #     print(f"ID: {ids[i]}, Meta: {metadatas[i]}, Doc: {documents[i][:100]}...")

    return documents # Return the added documents (or chunks)


async def search_and_crawl_resources(emotion_analysis: Dict):
    """Search the web based on emotion analysis, crawl pages, and add to vector DB."""
    emotion = emotion_analysis.get('primary_emotion', 'emotional challenge')
    triggers = emotion_analysis.get('triggers', [])
    growth_opportunities = emotion_analysis.get('growth_opportunities', [])

    # Create more targeted search queries
    search_queries = [
        f"how to cope with feeling {emotion}",
        f"strategies for managing {emotion}",
        f"understanding triggers for {emotion}",
        f"personal growth after feeling {emotion}",
    ]
    if triggers:
        search_queries.append(f"dealing with {emotion} triggered by {triggers[0]}")
    if growth_opportunities:
        search_queries.append(f"{growth_opportunities[0]} techniques")

    all_urls = set() # Use a set to automatically handle duplicates
    url_limit_per_query = 2
    total_url_limit = 6

    for query in search_queries:
        if len(all_urls) >= total_url_limit:
            break
        urls = get_web_urls(query, num_results=url_limit_per_query)
        for url in urls:
            if len(all_urls) < total_url_limit:
                all_urls.add(url)
            else:
                break

    unique_urls = list(all_urls)
    if not unique_urls:
        st.warning("Could not find any relevant web resources.")
        return []

    st.info(f"Found {len(unique_urls)} unique URLs to crawl...")

    # Define a relevant query for the crawler's content filter
    crawl_query = f"{emotion} coping strategies {' '.join(triggers)} {' '.join(growth_opportunities)}"

    # Crawl the webpages
    crawl_results = await crawl_webpages(unique_urls, query=crawl_query)

    if not crawl_results:
        st.warning("Crawling did not yield usable content from the found URLs.")
        return []

    # Add results to vector database
    st.info("Adding crawled content to resource database...")
    web_content_chunks = add_to_vector_database(crawl_results, collection_name="web_resources")

    st.success(f"Successfully processed {len(crawl_results)} web pages and added {len(web_content_chunks)} content chunks to resources.")
    return web_content_chunks # Return the chunks added

def query_vector_database(query_text: str, n_results: int = 5, collection_name="web_resources") -> List[str]:
    """Queries the vector database for relevant content chunks."""
    collection, _ = get_vector_collection(collection_name)
    if not collection:
        st.error("Vector database collection not available for querying.")
        return []

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents'] # Only need the document content for synthesis
        )
        # Extract documents from the nested structure
        if results and 'documents' in results and results['documents']:
            return results['documents'][0] # Query returns a list of results for each query text
        else:
            return []
    except Exception as e:
        st.error(f"Error querying vector database: {e}")
        return []

# --- UI Components ---

def render_login_page():
    """Render the login/signup page."""
    st.title("🌱 Emotion to Action")
    st.subheader("Transform Emotional Experiences into Personal Growth")

    st.markdown("""
        **Welcome to Emotion to Action!**

        This app helps you transform intense emotional experiences into opportunities for personal growth.
        Our AI-powered system guides you through:

        - Capturing and analyzing your emotions
        - Creating personalized growth plans
        - Finding targeted resources for emotional management
        - Connecting with a supportive community

        Get started by creating an account or logging in below.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if login_username in st.session_state.user_db:
                # TODO: Implement proper password hashing and verification
                if st.session_state.user_db[login_username].get('password') == login_password:
                    st.session_state.current_user = login_username
                    st.session_state.authenticated = True
                    st.session_state.current_view = "main" # Navigate to dashboard on login
                    st.rerun()
                else:
                    st.error("Invalid password")
            else:
                st.error("Username not found")

    with col2:
        st.subheader("Sign Up")
        signup_username = st.text_input("Choose Username", key="signup_username")
        signup_password = st.text_input("Choose Password", type="password", key="signup_password")
        signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")

        if st.button("Sign Up"):
            if not signup_username or not signup_password:
                 st.error("Username and password cannot be empty.")
            elif signup_username in st.session_state.user_db:
                st.error("Username already taken")
            elif signup_password != signup_confirm:
                st.error("Passwords do not match")
            else:
                # TODO: Implement proper password hashing before storing
                st.session_state.user_db[signup_username] = {
                    'password': signup_password, # Store hashed password in real app
                    'joined_date': datetime.datetime.now().isoformat(),
                    'premium': False,
                    'streak': 0,
                    'points': 0,
                    'goals': {} # Initialize empty goals dict
                }
                st.success("Account created! You can now login.")

def render_sidebar():
    """Render the sidebar with navigation options."""
    st.sidebar.title("Navigation")

    # Ensure user is logged in and data exists
    if 'current_user' in st.session_state and st.session_state.current_user and st.session_state.current_user in st.session_state.user_db:
        user_data = st.session_state.user_db[st.session_state.current_user]
        st.sidebar.write(f"👋 Hello, {st.session_state.current_user}!")
        st.sidebar.write(f"🔥 Streak: {user_data.get('streak', 0)} days")
        st.sidebar.write(f"⭐ Points: {user_data.get('points', 0)}")

        # Define page options and corresponding view states
        pages = {
            "Dashboard": "main",
            "Journal": "journal",
            "Community": "community",
            "Resources": "resources",
            "Profile": "profile"
        }

        # Find the current page name based on the view state
        current_page_name = next((name for name, view in pages.items() if st.session_state.get('current_view') == view), "Dashboard")

        # Use radio buttons for navigation
        selected_page = st.sidebar.radio(
            "Go to:",
            options=list(pages.keys()),
            index=list(pages.keys()).index(current_page_name), # Set default based on current view
            key="navigation_radio"
        )

        # Update the current view based on selection
        if pages[selected_page] != st.session_state.get('current_view'):
             st.session_state.current_view = pages[selected_page]
             # Clear specific states when navigating away from certain pages
             if 'current_emotion_id' in st.session_state:
                 del st.session_state['current_emotion_id']
             if 'current_resource_query' in st.session_state:
                 del st.session_state['current_resource_query']
             st.rerun()


        # Logout button
        st.sidebar.divider()
        if st.sidebar.button("Logout"):
            # Clear relevant session state keys on logout
            keys_to_clear = ['authenticated', 'current_user', 'current_view', 'current_emotion_id', 'current_resource_query']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    else:
        st.sidebar.info("Please log in or sign up.")


def render_main_dashboard():
    """Render the main dashboard."""
    st.title("🌱 Your Growth Dashboard")

    user_id = st.session_state.current_user
    user_data = st.session_state.user_db.get(user_id, {})
    emotion_history = get_user_emotion_history(user_id)

    # Welcome message with quick actions
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
        ### Welcome back, {user_id}!
        You're on a {user_data.get('streak', 0)}-day streak of emotional awareness.
        What would you like to do today?
        """)

    with col2:
        if st.button("📝 New Journal Entry"):
            st.session_state.current_view = "journal"
            st.rerun()
        if st.button("👥 Explore Community"):
            st.session_state.current_view = "community"
            st.rerun()

    st.divider()

    # Recent emotions
    st.subheader("Recent Emotional Journey")

    if not emotion_history:
        st.info("You haven't recorded any emotions yet. Start journaling to see your emotional journey!")
    else:
        # Display recent emotions in a timeline
        recent_emotions = sorted(emotion_history, key=lambda x: x.get('timestamp', ''), reverse=True)[:3]

        for idx, entry in enumerate(recent_emotions):
            analysis = entry.get('analysis', {})
            emotion_id = entry.get('id')
            timestamp_str = entry.get('timestamp', 'Unknown time')
            try:
                # Attempt to parse and format the date nicely
                 timestamp_dt = datetime.datetime.fromisoformat(timestamp_str)
                 display_date = timestamp_dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                 display_date = timestamp_str[:16] # Fallback to simple slicing


            with st.container(border=True):
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.write(f"**{analysis.get('primary_emotion', 'N/A')}**")
                    st.write(f"Intensity: {analysis.get('intensity', 'N/A')}/10")
                    st.caption(display_date)

                with col2:
                    st.write(entry.get('journal_entry', 'No entry text.')[:150] + "..." if len(entry.get('journal_entry', '')) > 150 else entry.get('journal_entry', ''))

                    # Action buttons specific to this entry
                    button_key_base = f"dash_{emotion_id}_{idx}"
                    b_col1, b_col2, b_col3 = st.columns(3)
                    with b_col1:
                        if st.button("View Analysis", key=f"{button_key_base}_analysis", type="secondary"):
                            st.session_state.current_emotion_id = emotion_id
                            st.session_state.current_view = "emotion_analysis"
                            st.rerun()
                    with b_col2:
                        growth_plan = get_growth_plan(user_id, emotion_id)
                        if growth_plan:
                            if st.button("View Plan", key=f"{button_key_base}_viewplan", type="secondary"):
                                st.session_state.current_emotion_id = emotion_id
                                st.session_state.current_view = "growth_plan"
                                st.rerun()
                        else:
                            if st.button("Create Plan", key=f"{button_key_base}_createplan", type="secondary"):
                                st.session_state.current_emotion_id = emotion_id
                                st.session_state.current_view = "create_growth_plan"
                                st.rerun()
                    with b_col3:
                         if st.button("Find Resources", key=f"{button_key_base}_findres", type="secondary"):
                            st.session_state.current_emotion_id = emotion_id
                            st.session_state.current_view = "search_resources" # Go to initiate search
                            st.rerun()


            st.write("---") # Visual separator between entries

    st.divider()

    # Community highlights
    st.subheader("Community Highlights")

    community_posts = get_community_posts(3)

    if not community_posts:
        st.info("No community posts yet. Be the first to share!")
    else:
        for post in community_posts:
             with st.container(border=True):
                st.write(f"**{post.get('title', 'No Title')}**")
                st.caption(f"Posted by: {post.get('user_id', 'Unknown')} | ❤️ {post.get('likes', 0)} | 💬 {len(post.get('comments', []))}")
                st.write(post.get('content', '')[:100] + "...") # Show snippet
                if st.button("View Post", key=f"view_post_{post.get('id')}", type="secondary"):
                     st.session_state.current_view = "community"
                     st.session_state.selected_post_id = post.get('id') # Store which post to maybe highlight
                     st.rerun()


        if st.button("See All Community Posts"):
            st.session_state.current_view = "community"
            st.rerun()

def render_journal_page():
    """Render the journaling page."""
    st.title("📝 Emotional Journal")

    st.write("""
    Take a moment to reflect on your emotions.
    What are you feeling right now? What triggered these feelings?
    Writing about your emotional experiences helps develop self-awareness and emotional intelligence.
    """)

    journal_entry = st.text_area(
        "What are you feeling right now? Describe your emotional experience in detail.",
        height=250,
        key="journal_entry_input"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Analyze My Emotions", type="primary"):
            if not journal_entry or not journal_entry.strip():
                st.warning("Please write about your emotional experience first.")
                return # Stop execution if no entry

            with st.spinner("🧠 Analyzing your emotions with AI... This may take a moment."):
                emotion_analysis = analyze_emotion(journal_entry)

                if "error" in emotion_analysis or "raw_response" in emotion_analysis:
                    st.error("Could not analyze emotions effectively. The AI might be unavailable or the response was not structured correctly.")
                    st.json(emotion_analysis) # Show the problematic response
                else:
                    # Save the emotion entry
                    emotion_id = save_emotion_entry(
                        st.session_state.current_user,
                        {
                            'journal_entry': journal_entry,
                            'analysis': emotion_analysis
                        }
                    )

                    # Update user streak and points (basic gamification)
                    user_data = st.session_state.user_db[st.session_state.current_user]
                    # Basic streak logic: Check if last entry was yesterday
                    # This is simplified; a real app needs more robust date checking
                    user_data['streak'] = user_data.get('streak', 0) + 1 # Increment streak for now
                    user_data['points'] = user_data.get('points', 0) + 10
                    save_user_data(st.session_state.current_user, user_data)

                    # Set session state for navigation to the analysis page
                    st.session_state.current_emotion_id = emotion_id
                    st.session_state.current_view = "emotion_analysis"
                    st.success("Emotion analysis complete!")
                    st.rerun() # Rerun to navigate to the analysis page

    with col2:
        if st.button("Cancel"):
            # Optionally clear the text area or just navigate away
            # st.session_state.journal_entry_input = "" # Uncomment to clear
            st.session_state.current_view = "main"
            st.rerun()

    st.divider()

    # Past journal entries
    st.subheader("Past Journal Entries")

    user_id = st.session_state.current_user
    emotion_history = get_user_emotion_history(user_id)

    if not emotion_history:
        st.info("You haven't recorded any emotions yet.")
    else:
        # Display past journal entries, newest first
        past_entries = sorted(emotion_history, key=lambda x: x.get('timestamp',''), reverse=True)

        for idx, entry in enumerate(past_entries):
            analysis = entry.get('analysis', {})
            emotion_id = entry.get('id')
            timestamp_str = entry.get('timestamp', 'Unknown time')
            try:
                timestamp_dt = datetime.datetime.fromisoformat(timestamp_str)
                display_date = timestamp_dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                display_date = timestamp_str[:16]

            expander_title = f"{analysis.get('primary_emotion', 'Entry')} - {display_date}"
            with st.expander(expander_title):
                st.caption("Journal Entry:")
                st.markdown(f"> {entry.get('journal_entry', 'No text.')}") # Use markdown blockquote
                st.write("---")

                # Action buttons inside expander
                button_key_base = f"past_{emotion_id}_{idx}"
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("View Analysis", key=f"{button_key_base}_analysis"):
                        st.session_state.current_emotion_id = emotion_id
                        st.session_state.current_view = "emotion_analysis"
                        st.rerun()

                with col2:
                    growth_plan = get_growth_plan(user_id, emotion_id)
                    if growth_plan:
                        if st.button("View Plan", key=f"{button_key_base}_viewplan"):
                            st.session_state.current_emotion_id = emotion_id
                            st.session_state.current_view = "growth_plan"
                            st.rerun()
                    else:
                        if st.button("Create Plan", key=f"{button_key_base}_createplan"):
                            st.session_state.current_emotion_id = emotion_id
                            st.session_state.current_view = "create_growth_plan"
                            st.rerun()
                with col3:
                      if st.button("Find Resources", key=f"{button_key_base}_findres"):
                          st.session_state.current_emotion_id = emotion_id
                          st.session_state.current_view = "search_resources"
                          st.rerun()

def render_emotion_analysis(emotion_id: str):
    """Render the emotion analysis page."""
    user_id = st.session_state.current_user
    emotion_data = get_emotion_entry(user_id, emotion_id)

    if not emotion_data:
        st.error("Emotion entry not found or you do not have permission to view it.")
        if st.button("Back to Journal"):
             st.session_state.current_view = "journal"
             # Clear potentially invalid emotion ID
             if 'current_emotion_id' in st.session_state: del st.session_state['current_emotion_id']
             st.rerun()
        return

    analysis = emotion_data.get('analysis', {})
    journal_entry = emotion_data.get('journal_entry', 'No journal entry text.')

    st.title("🧠 Emotion Analysis")

    # Show journal entry
    st.subheader("Your Journal Entry")
    with st.container(border=True):
         st.markdown(f"> {journal_entry}")


    st.divider()

    # Show analysis results
    st.subheader("AI Analysis Results")

    if "error" in analysis or "raw_response" in analysis or not analysis:
         st.warning("Analysis data is missing or incomplete.")
         st.json(analysis) # Show what we have
         primary_emotion = "N/A" # Set defaults for dependent actions
         triggers = []
    else:
        primary_emotion = analysis.get('primary_emotion', 'Unknown')
        intensity = analysis.get('intensity', 'N/A')
        triggers = analysis.get('triggers', [])
        patterns = analysis.get('patterns', [])
        growth_opportunities = analysis.get('growth_opportunities', [])
        action_steps = analysis.get('action_steps', [])

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Primary Emotion", value=primary_emotion)
            st.metric(label="Intensity (1-10)", value=str(intensity))

            st.write("**Potential Triggers:**")
            if triggers:
                for trigger in triggers:
                    st.write(f"- {trigger}")
            else:
                st.caption("None identified")

        with col2:
            st.write("**Emotional Patterns:**")
            if patterns:
                for pattern in patterns:
                    st.write(f"- {pattern}")
            else:
                st.caption("None identified")

        st.write("**Growth Opportunities:**")
        if growth_opportunities:
            for opportunity in growth_opportunities:
                st.write(f"- {opportunity}")
        else:
            st.caption("None identified")

        st.write("**Suggested Action Steps:**")
        if action_steps:
            for step in action_steps:
                st.write(f"- {step}")
        else:
             st.caption("None identified")

    st.divider()

    # Actions
    st.subheader("Next Steps")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Check if a plan already exists for this entry
        growth_plan = get_growth_plan(user_id, emotion_id)
        if growth_plan:
             if st.button("View Growth Plan"):
                st.session_state.current_view = "growth_plan"
                st.rerun()
        else:
            if st.button("💡 Create Growth Plan", type="primary"):
                st.session_state.current_view = "create_growth_plan"
                st.rerun()


    with col2:
         # Check if resources already exist for this entry
         resources = get_resources(user_id, emotion_id)
         if resources:
             if st.button("View Resources"):
                 st.session_state.current_view = "view_resources" # New view to display saved resources
                 st.rerun()
         else:
             # Only enable resource search if analysis was successful
            if primary_emotion != "N/A":
                if st.button("🔎 Find Resources"):
                    st.session_state.current_view = "search_resources"
                    st.rerun()
            else:
                st.button("🔎 Find Resources", disabled=True)


    with col3:
        if st.button("Back to Journal"):
            st.session_state.current_view = "journal"
            # Clear the specific emotion ID as we are navigating away from its context
            if 'current_emotion_id' in st.session_state: del st.session_state['current_emotion_id']
            st.rerun()


# --- Additional UI Rendering Functions (Placeholder Implementations) ---

def render_create_growth_plan(emotion_id: str):
    """Render the page to create a growth plan."""
    st.title("💡 Create Your Growth Plan")

    user_id = st.session_state.current_user
    emotion_data = get_emotion_entry(user_id, emotion_id)

    if not emotion_data or 'analysis' not in emotion_data:
        st.error("Cannot create plan: Emotion analysis data is missing.")
        if st.button("Back"):
            st.session_state.current_view = "emotion_analysis" # Or 'journal'
            st.rerun()
        return

    analysis = emotion_data['analysis']
    st.subheader("Based on your analysis:")
    st.write(f"- **Emotion:** {analysis.get('primary_emotion', 'N/A')}")
    st.write(f"- **Triggers:** {', '.join(analysis.get('triggers',[])) or 'N/A'}")
    st.write(f"- **Growth Opportunities:**")
    for opp in analysis.get('growth_opportunities', []): st.write(f"  - {opp}")

    st.divider()
    st.subheader("Define Your Goals (Optional)")
    st.write("What do you want to achieve? This helps tailor the plan.")

    # Load existing goals or provide inputs
    user_data = st.session_state.user_db.get(user_id, {})
    current_goals = user_data.get('goals', {})

    goal1 = st.text_input("Goal 1:", value=current_goals.get("goal1", ""), key="gp_goal1")
    goal2 = st.text_input("Goal 2:", value=current_goals.get("goal2", ""), key="gp_goal2")
    goal3 = st.text_input("Goal 3:", value=current_goals.get("goal3", ""), key="gp_goal3")

    user_goals = {"goal1": goal1, "goal2": goal2, "goal3": goal3}

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✨ Generate Growth Plan", type="primary"):
            with st.spinner("Generating your personalized growth plan..."):
                # Save goals to user profile before generating
                user_data['goals'] = {k: v for k, v in user_goals.items() if v} # Save non-empty goals
                save_user_data(user_id, user_data)

                # Call LLM to generate plan
                plan_data = generate_growth_plan(analysis, user_data['goals'])

                if "error" in plan_data or "raw_response" in plan_data:
                     st.error("Failed to generate growth plan.")
                     st.json(plan_data)
                else:
                    # Save the generated plan
                    save_growth_plan(user_id, emotion_id, plan_data)

                    # Update user points
                    user_data['points'] = user_data.get('points', 0) + 20 # More points for a plan
                    save_user_data(user_id, user_data)

                    st.success("Growth plan generated!")
                    # Navigate to view the plan
                    st.session_state.current_view = "growth_plan"
                    st.rerun()

    with col2:
        if st.button("Cancel"):
            st.session_state.current_view = "emotion_analysis" # Go back to analysis
            st.rerun()

def render_growth_plan(emotion_id: str):
    """Render the generated growth plan."""
    st.title("🚀 Your Growth Plan")

    user_id = st.session_state.current_user
    plan_data = get_growth_plan(user_id, emotion_id)
    emotion_data = get_emotion_entry(user_id, emotion_id) # Get context

    if not plan_data:
        st.error("Growth plan not found for this entry.")
        if st.button("Back to Analysis"):
             st.session_state.current_view = "emotion_analysis"
             st.rerun()
        return

    if emotion_data and 'analysis' in emotion_data:
        st.caption(f"Plan related to feeling: **{emotion_data['analysis'].get('primary_emotion', 'N/A')}** on {emotion_data.get('timestamp', '')[:10]}")

    st.divider()

    if "error" in plan_data or "raw_response" in plan_data:
        st.warning("Plan data seems incomplete or improperly formatted.")
        st.json(plan_data)
    else:
        st.subheader("🗓️ Short-term Actions (Next 24-48 hours)")
        actions = plan_data.get('short_term_actions', [])
        if actions:
            for i, action in enumerate(actions):
                st.checkbox(f"{i+1}. {action}", key=f"st_action_{i}") # Use checkboxes for tracking
        else:
            st.caption("No short-term actions defined.")

        st.subheader("🧘 Medium-term Practices (Next 1-2 weeks)")
        practices = plan_data.get('medium_term_practices', [])
        if practices:
            for i, practice in enumerate(practices):
                st.checkbox(f"{i+1}. {practice}", key=f"mt_practice_{i}")
        else:
            st.caption("No medium-term practices defined.")

        st.subheader("🌱 Long-term Changes (Next 1-3 months)")
        changes = plan_data.get('long_term_changes', [])
        if changes:
             for i, change in enumerate(changes):
                 st.checkbox(f"{i+1}. {change}", key=f"lt_change_{i}")
        else:
             st.caption("No long-term changes defined.")

        st.subheader("❓ Reflection Prompts")
        prompts = plan_data.get('reflection_prompts', [])
        if prompts:
             for prompt in prompts:
                 st.write(f"- {prompt}")
        else:
            st.caption("No reflection prompts defined.")

        st.subheader("📊 Success Metrics")
        metrics = plan_data.get('success_metrics', [])
        if metrics:
             for metric in metrics:
                 st.write(f"- {metric}")
        else:
             st.caption("No success metrics defined.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
         if st.button("Back to Analysis"):
             st.session_state.current_view = "emotion_analysis"
             st.rerun()
    with col2:
         # Placeholder for editing or updating plan status
         st.button("Mark as Complete (Future Feature)", disabled=True)


def render_search_resources(emotion_id: str):
    """Page to initiate web search and crawling for resources."""
    st.title("🔎 Find Growth Resources")

    user_id = st.session_state.current_user
    emotion_data = get_emotion_entry(user_id, emotion_id)

    if not emotion_data or 'analysis' not in emotion_data:
        st.error("Cannot search for resources: Emotion analysis data is missing.")
        if st.button("Back"):
            st.session_state.current_view = "emotion_analysis"
            st.rerun()
        return

    analysis = emotion_data['analysis']
    st.write(f"Searching for resources related to: **{analysis.get('primary_emotion', 'your recent experience')}**")
    st.write(f"Considering triggers: {', '.join(analysis.get('triggers',[])) or 'N/A'}")

    st.info("This involves searching the web and processing relevant pages. It may take a minute or two.")

    # Button to start the async process
    if st.button("Start Resource Search", type="primary"):
        with st.spinner("Searching the web and crawling relevant pages... Please wait."):
            # --- Async Execution ---
            # Streamlit runs synchronously, so we need to run the async function
            # using asyncio.run() or get_event_loop().run_until_complete()
            try:
                # Get or create an event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                # Run the async function and wait for its completion
                crawled_chunks = loop.run_until_complete(search_and_crawl_resources(analysis))
                loop.close()

                if crawled_chunks:
                     st.session_state.last_crawled_chunks = crawled_chunks # Store for potential synthesis
                     st.success("Web search and crawling complete! Content added to resource database.")
                     # Now, offer to synthesize or view general resources
                     st.session_state.current_view = "resources" # Navigate to the main resources page
                     st.rerun()
                else:
                     st.warning("Search completed, but no usable content was found or added.")
                     # Stay on this page or navigate back
                     if st.button("Try Again or Go Back"):
                          st.session_state.current_view = "emotion_analysis"
                          st.rerun()

            except Exception as e:
                st.error(f"An error occurred during the resource search process: {e}")
                print(f"Error in search_and_crawl_resources execution: {e}")


    if st.button("Cancel"):
         st.session_state.current_view = "emotion_analysis"
         st.rerun()

def render_resources():
    """Render the main resources page, allowing search and synthesis."""
    st.title("📚 Growth Resources")
    st.write("Explore resources based on your needs or synthesize findings related to a specific journal entry.")

    user_id = st.session_state.current_user
    emotion_id = st.session_state.get('current_emotion_id') # Check if we came from a specific entry

    col1, col2 = st.columns([3, 1])

    with col1:
        # Option 1: Synthesize resources for the current emotion entry (if applicable)
        if emotion_id:
            emotion_data = get_emotion_entry(user_id, emotion_id)
            plan_data = get_growth_plan(user_id, emotion_id)
            if emotion_data:
                st.subheader(f"Synthesize Resources for '{emotion_data['analysis'].get('primary_emotion', 'Entry')}'")
                # Check if synthesis already exists
                existing_synthesis = get_resources(user_id, emotion_id)
                if existing_synthesis:
                    st.success("Synthesis already generated for this entry.")
                    if st.button("View Existing Synthesis", key="view_existing_synth"):
                        st.session_state.current_view = "view_resources"
                        st.rerun()
                else:
                    # Query vector DB based on emotion analysis
                    query_text = f"{emotion_data['analysis'].get('primary_emotion', '')} coping strategies {', '.join(emotion_data['analysis'].get('triggers',[]))}"
                    st.write(f"Searching stored resources related to: '{query_text}'")

                    if st.button("Synthesize Relevant Resources", type="primary", key="synthesize_now"):
                        with st.spinner("Querying resource database and synthesizing..."):
                            retrieved_docs = query_vector_database(query_text, n_results=10) # Get top 10 chunks
                            if retrieved_docs:
                                st.info(f"Found {len(retrieved_docs)} relevant content snippets.")
                                synthesis_result = synthesize_resources(emotion_data['analysis'], plan_data, retrieved_docs)

                                if "error" in synthesis_result or "raw_response" in synthesis_result:
                                    st.error("Failed to synthesize resources.")
                                    st.json(synthesis_result)
                                else:
                                    # Save the synthesis
                                    save_resource(user_id, emotion_id, synthesis_result)
                                    user_data = st.session_state.user_db[user_id]
                                    user_data['points'] = user_data.get('points', 0) + 15
                                    save_user_data(user_id, user_data)
                                    st.success("Resources synthesized successfully!")
                                    st.session_state.current_view = "view_resources" # Navigate to view it
                                    st.rerun()
                            else:
                                st.warning("Could not find relevant resources in the database to synthesize. Try broadening the search or adding more resources.")

    with col2:
         # Option 2: General Search in Resources
         st.subheader("Search All Resources")
         search_query = st.text_input("Enter keywords (e.g., 'managing anxiety', 'building confidence')", key="resource_search_query")
         if st.button("Search", key="general_resource_search"):
             if search_query:
                 st.session_state.current_resource_query = search_query # Store query
                 st.session_state.current_view = "view_resources" # Go to display search results
                 st.rerun()
             else:
                 st.warning("Please enter a search term.")

    st.divider()
    st.caption("Note: Resources are gathered from web crawls and stored for semantic search. Synthesis uses AI to summarize relevant findings.")


def render_view_resources():
    """Displays synthesized resources for an entry or search results."""
    st.title("📚 View Resources")

    user_id = st.session_state.current_user
    emotion_id = st.session_state.get('current_emotion_id')
    search_query = st.session_state.get('current_resource_query')

    resource_data = None
    display_mode = "synthesis" # Default: show synthesis for an emotion_id

    if search_query:
         display_mode = "search"
         st.subheader(f"Search Results for: '{search_query}'")
         with st.spinner("Searching resource database..."):
             resource_data = query_vector_database(search_query, n_results=15) # List of text chunks
    elif emotion_id:
        resource_data = get_resources(user_id, emotion_id) # Dict from synthesis LLM call
        if resource_data:
            emotion_data = get_emotion_entry(user_id, emotion_id)
            st.subheader(f"Synthesized Resources for '{emotion_data['analysis'].get('primary_emotion', 'Entry')}'")
        else:
            st.warning("No synthesized resources found for this entry. You might need to generate them first.")
            display_mode = "none"
    else:
        st.info("Navigate here from a specific journal entry's analysis or via the resource search.")
        display_mode = "none"


    if display_mode == "synthesis" and resource_data:
         if "error" in resource_data or "raw_response" in resource_data:
             st.warning("Resource data is incomplete or improperly formatted.")
             st.json(resource_data)
         else:
             st.markdown("**Key Insights:**")
             insights = resource_data.get('key_insights', [])
             if insights:
                 for insight in insights: st.write(f"- {insight}")
             else: st.caption("None provided.")

             st.markdown("**Practical Exercises:**")
             exercises = resource_data.get('practical_exercises', [])
             if exercises:
                  for exercise in exercises: st.write(f"- {exercise}")
             else: st.caption("None provided.")

             st.markdown("**Recommended Readings/Resources:**")
             readings = resource_data.get('recommended_readings', [])
             if readings:
                 for reading in readings: st.write(f"- {reading}")
             else: st.caption("None provided.")

             st.markdown("**Expert Advice Summary:**")
             advice = resource_data.get('expert_advice', '')
             if advice: st.write(advice)
             else: st.caption("None provided.")

             st.markdown("**Action Plan Integration:**")
             plan = resource_data.get('action_plan', [])
             if plan:
                  for step in plan: st.write(f"- {step}")
             else: st.caption("None provided.")

    elif display_mode == "search" and resource_data:
         if resource_data: # resource_data is a list of text chunks here
             st.info(f"Found {len(resource_data)} relevant snippets from crawled web pages.")
             for i, doc in enumerate(resource_data):
                 with st.expander(f"Snippet {i+1}", expanded=(i<3)): # Expand first few
                     st.markdown(doc)
                     # You could try to retrieve metadata (like source URL) if stored and included in query results
                     # metadata = ... # retrieve metadata if available
                     # st.caption(f"Source: {metadata.get('source', 'Unknown')}")
             st.caption("These are raw text snippets matching your query. For a structured summary, use the 'Synthesize Resources' feature from a journal entry.")
         else:
             st.warning("No relevant resources found matching your search query in the database.")

    st.divider()
    if st.button("Back to Resources Hub"):
        st.session_state.current_view = "resources"
        # Clear specific context
        if 'current_resource_query' in st.session_state: del st.session_state['current_resource_query']
        # Keep emotion_id if we came from synthesis, otherwise clear it? Maybe clear always when going back to hub.
        if 'current_emotion_id' in st.session_state and display_mode == "search": del st.session_state['current_emotion_id']
        st.rerun()


def render_community_page():
    """Render the community interaction page."""
    st.title("👥 Community Hub")
    st.write("Share your experiences, offer support, and learn from others on their growth journeys.")

    user_id = st.session_state.current_user

    # Section 1: AI Suggestions for Sharing (Optional, based on last entry)
    last_emotion_entry = get_user_emotion_history(user_id)[-1] if get_user_emotion_history(user_id) else None
    if last_emotion_entry:
        with st.expander("AI Suggestions for Community Interaction"):
            analysis = last_emotion_entry.get('analysis')
            plan = get_growth_plan(user_id, last_emotion_entry.get('id'))
            if analysis:
                # Use a session state flag to avoid calling LLM on every rerun
                if 'community_suggestions' not in st.session_state or st.session_state.get('suggestion_emotion_id') != last_emotion_entry.get('id'):
                     with st.spinner("Getting AI suggestions..."):
                          st.session_state.community_suggestions = get_community_suggestions(analysis, plan)
                          st.session_state.suggestion_emotion_id = last_emotion_entry.get('id') # Track which entry suggestions relate to

                suggestions = st.session_state.community_suggestions
                if suggestions and not ("error" in suggestions or "raw_response" in suggestions):
                    st.write("**Consider sharing about:**")
                    for opp in suggestions.get('sharing_opportunities', []): st.write(f"- {opp}")
                    st.write("**Relevant discussion topics:**")
                    for topic in suggestions.get('recommended_topics', []): st.write(f"- {topic}")
                    st.write("**How community might help:**")
                    for need in suggestions.get('support_needs', []): st.write(f"- {need}")
                elif suggestions:
                     st.warning("Could not get structured suggestions.")
                     st.json(suggestions) # Show raw/error
                else:
                     st.info("No specific suggestions available right now.")
            else:
                st.info("Analyze a recent journal entry to get personalized community suggestions.")


    st.divider()

    # Section 2: Create New Post
    st.subheader("Share Your Thoughts or Ask a Question")
    with st.form("new_post_form", clear_on_submit=True):
        post_title = st.text_input("Post Title")
        post_content = st.text_area("Your message:")
        submitted = st.form_submit_button("Post to Community")
        if submitted:
            if post_title and post_content:
                post_id = save_community_post(user_id, {'title': post_title, 'content': post_content})
                st.success("Post submitted successfully!")
                # Update user points
                user_data = st.session_state.user_db[user_id]
                user_data['points'] = user_data.get('points', 0) + 5
                save_user_data(user_id, user_data)
                st.rerun() # Refresh to show the new post
            else:
                st.warning("Please provide both a title and content for your post.")

    st.divider()

    # Section 3: View Community Posts
    st.subheader("Recent Community Activity")
    posts = get_community_posts(limit=20)

    if not posts:
        st.info("The community feed is empty. Be the first to post!")
    else:
        for post in posts:
            post_id = post.get('id')
            timestamp_str = post.get('timestamp', 'Unknown time')
            try:
                 display_date = datetime.datetime.fromisoformat(timestamp_str).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                 display_date = timestamp_str[:16]


            with st.container(border=True):
                st.markdown(f"### {post.get('title', 'No Title')}")
                st.caption(f"Posted by: **{post.get('user_id', 'Unknown')}** on {display_date}")
                st.write(post.get('content', ''))
                st.write("---") # Separator within the post container

                # Likes and Comments Section
                col1, col2, col3 = st.columns([1, 1, 5])
                with col1:
                    if st.button(f"❤️ Like ({post.get('likes', 0)})", key=f"like_{post_id}"):
                        like_post(post_id, user_id) # Pass user_id for potential future logic
                        st.rerun() # Refresh to show updated like count
                with col2:
                     comment_count = len(post.get('comments', []))
                     st.write(f"💬 Comments ({comment_count})") # Just display count for now

                # Expander for Comments
                with st.expander("View / Add Comments"):
                    comments = sorted(post.get('comments', []), key=lambda c: c.get('timestamp',''))
                    if comments:
                        for comment in comments:
                            comment_ts_str = comment.get('timestamp', '')
                            try:
                                comment_date = datetime.datetime.fromisoformat(comment_ts_str).strftime("%Y-%m-%d %H:%M")
                            except ValueError:
                                comment_date = comment_ts_str[:16]

                            st.markdown(f"**{comment.get('user_id', 'Unknown')}** ({comment_date}):")
                            st.write(f"> {comment.get('comment', '')}")
                            st.caption("---") # Separator between comments
                    else:
                        st.caption("No comments yet.")

                    # Add new comment form within expander
                    new_comment = st.text_input("Add your comment:", key=f"comment_input_{post_id}")
                    if st.button("Submit Comment", key=f"submit_comment_{post_id}"):
                        if new_comment:
                            add_comment_to_post(post_id, user_id, new_comment)
                            # Update points for commenting
                            user_data = st.session_state.user_db[user_id]
                            user_data['points'] = user_data.get('points', 0) + 2
                            save_user_data(user_id, user_data)
                            st.rerun() # Refresh to show the new comment
                        else:
                            st.warning("Comment cannot be empty.")



def render_profile_page():
    """Render the user profile page."""
    st.title("👤 Your Profile")

    user_id = st.session_state.current_user
    user_data = st.session_state.user_db.get(user_id, {})

    if not user_data:
        st.error("User data not found.")
        return

    # Display User Stats
    st.subheader("Your Stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Username", user_id)
    with col2:
        st.metric("Journaling Streak", f"{user_data.get('streak', 0)} Days")
    with col3:
        st.metric("Growth Points", f"{user_data.get('points', 0)} ⭐")

    st.caption(f"Member since: {user_data.get('joined_date', 'N/A')[:10]}")
    st.caption(f"Premium Status: {'Active' if user_data.get('premium', False) else 'Not Active'}") # Placeholder

    st.divider()

    # Manage Growth Goals
    st.subheader("Your Growth Goals")
    st.write("Set or update your personal growth goals. These help personalize AI suggestions.")

    current_goals = user_data.get('goals', {})
    # Use a form to manage goal updates
    with st.form("goals_form"):
        goal1 = st.text_input("Goal 1:", value=current_goals.get("goal1", ""), key="prof_goal1")
        goal2 = st.text_input("Goal 2:", value=current_goals.get("goal2", ""), key="prof_goal2")
        goal3 = st.text_input("Goal 3:", value=current_goals.get("goal3", ""), key="prof_goal3")
        # Add more goals if needed

        submitted = st.form_submit_button("Save Goals")
        if submitted:
            new_goals = {
                "goal1": goal1.strip(),
                "goal2": goal2.strip(),
                "goal3": goal3.strip(),
                # Add other goals here
            }
            # Filter out empty goals before saving
            user_data['goals'] = {k: v for k, v in new_goals.items() if v}
            save_user_data(user_id, user_data)
            st.success("Goals updated successfully!")
            # No rerun needed unless you want immediate visual confirmation outside the form

    st.divider()

    # Placeholder for other profile settings
    st.subheader("Settings (Future Features)")
    st.checkbox("Enable Email Notifications", disabled=True)
    st.selectbox("Preferred AI Model (if applicable)", ["Default", "Model A", "Model B"], disabled=True)
    if st.button("Change Password", disabled=True):
        pass # Implement password change logic securely
    if st.button("Delete Account", type="secondary", disabled=True):
        st.warning("Account deletion is permanent and cannot be undone.")
        # Implement account deletion logic


# --- Main Application Logic ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="EmotionToAction")

    # Initialize databases (session state) if they don't exist
    initialize_databases()

    # Initialize session state variables if not present
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "login" # Start at login page
    if 'current_emotion_id' not in st.session_state:
         st.session_state.current_emotion_id = None # ID of the emotion entry being worked on
    if 'current_resource_query' not in st.session_state:
          st.session_state.current_resource_query = None # For general resource search


    # --- Routing ---
    if not st.session_state.authenticated:
        st.session_state.current_view = "login" # Force login view if not authenticated
        render_login_page()
    else:
        # If authenticated, show sidebar and route to the correct view
        render_sidebar()

        view = st.session_state.current_view
        emotion_id = st.session_state.get('current_emotion_id')

        if view == "main":
            render_main_dashboard()
        elif view == "journal":
            render_journal_page()
        elif view == "emotion_analysis":
            if emotion_id:
                render_emotion_analysis(emotion_id)
            else:
                st.warning("No emotion entry selected. Please select one from your journal.")
                st.session_state.current_view = "journal"
                st.rerun()
        elif view == "create_growth_plan":
             if emotion_id:
                 render_create_growth_plan(emotion_id)
             else:
                 st.warning("Cannot create plan without a selected emotion entry.")
                 st.session_state.current_view = "journal"
                 st.rerun()
        elif view == "growth_plan":
             if emotion_id:
                 render_growth_plan(emotion_id)
             else:
                 st.warning("Cannot view plan without a selected emotion entry.")
                 st.session_state.current_view = "journal"
                 st.rerun()
        elif view == "search_resources":
             if emotion_id:
                 render_search_resources(emotion_id)
             else:
                  st.warning("Cannot search resources without a selected emotion entry.")
                  st.session_state.current_view = "journal"
                  st.rerun()
        elif view == "resources":
             render_resources()
        elif view == "view_resources":
             render_view_resources() # Handles both synthesis display and search results
        elif view == "community":
            render_community_page()
        elif view == "profile":
            render_profile_page()
        elif view == "login": # Should not happen if authenticated, but as fallback
             render_login_page()
        else:
            # Fallback to dashboard if view is unknown
            st.warning(f"Unknown view: {view}. Navigating to Dashboard.")
            st.session_state.current_view = "main"
            render_main_dashboard()


if __name__ == "__main__":
    # Consider environment variables for configuration (e.g., Ollama URL)
    # os.environ['OLLAMA_HOST'] = 'http://host.docker.internal:11434' # Example for Docker
    main()

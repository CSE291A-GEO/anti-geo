import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from readabilipy import simple_json_from_html_string
import trafilatura
import nltk
from dotenv import load_dotenv
import google.generativeai as genai
import json
import os
import pickle
import uuid

print("DEBUG: All imports completed")

load_dotenv()
print("DEBUG: .env loaded")

genai.configure(api_key=os.environ.get('GEMINI_API_KEY', ''))
print("DEBUG: Gemini configured")

# download NLTK punkt tokenizer
print("DEBUG: About to download NLTK punkt")
try:
    nltk.download('punkt_tab')
    print("DEBUG: NLTK punkt downloaded")
except Exception as e:
    print(f"DEBUG: NLTK download failed: {e}")
    print("DEBUG: Continuing without NLTK...")


def clean_source_gpt35(source : str) -> str:
    print(f"DEBUG: clean_source_gpt35 called with source length: {len(source)}")
    
    # Use smaller chunks to avoid rate limits (500 characters each)
    chunk_size = 500
    chunks = [source[i:i+chunk_size] for i in range(0, len(source), chunk_size)]
    print(f"DEBUG: Split into {len(chunks)} chunks")
    
    cleaned_chunks = []
    
    for chunk_idx, chunk in enumerate(chunks):
        print(f"DEBUG: Processing chunk {chunk_idx + 1}/{len(chunks)}")
        
        response = None
        for attempt in range(2):  # Only 2 attempts per chunk
            try:
                print(f"DEBUG: Gemini API attempt {attempt + 1} for chunk {chunk_idx + 1}")
                prompt = (
                    f"Clean and refine this text excerpt from a website (chunk {chunk_idx + 1} of {len(chunks)}). Remove any unwanted content such as headers, sidebars, and navigation menus. Retain only the main content and ensure that the text is well-formatted and free of HTML tags, special characters, and any other irrelevant information. Apply markdown formatting when outputting the answer.\n\nHere is the text excerpt:\n```html_text\n"
                    + chunk.strip() + "```"
                )
                model_client = genai.GenerativeModel("gemini-2.5-flash")
                response = model_client.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.0,
                        'top_p': 1.0,
                        'max_output_tokens': 500,  # Smaller output for chunks
                    }
                )
                print(f"DEBUG: Gemini API attempt {attempt + 1} successful for chunk {chunk_idx + 1}")
                break
            except Exception as e:
                print(f'Error while cleaning chunk {chunk_idx + 1} with Gemini API {e}')
                if attempt < 1:  # Only sleep once
                    import time
                    time.sleep(1)  # Shorter sleep
        
        if response is None:
            print(f"DEBUG: All Gemini API attempts failed for chunk {chunk_idx + 1}, using original chunk")
            cleaned_chunks.append(chunk)
        else:
            cleaned_text = (response.text or '').strip()
            print(f"DEBUG: Chunk {chunk_idx + 1} cleaned, length: {len(cleaned_text)}")
            cleaned_chunks.append(cleaned_text)
        
        # Add small delay between chunks to avoid rate limits
        if chunk_idx < len(chunks) - 1:
            import time
            time.sleep(0.5)
    
    # Combine all cleaned chunks
    result = "\n\n".join(cleaned_chunks)
    print(f"DEBUG: Final cleaned text length: {len(result)}")
    return result

def clean_source_text(text: str) -> str:
    return (
        text.strip()
        .replace("\n\n\n", "\n\n")
        .replace("\n\n", " ")
        .replace("  ", " ")
        .replace("\t", "")
        .replace("\n", "")
    )

import time
from pdb import set_trace as bp
import os


def summarize_text_identity(source, query) -> str:
    return source[:8000]


def search_handler(req, source_count = 8):
    query = req
    print(f"DEBUG: Starting search_handler with query: {query}")
    print("DEBUG: Step 1 - Function started")

    # Use Google Custom Search API for web search
    print("DEBUG: Step 2 - Checking environment variables")
    api_key = os.environ.get('GOOGLE_API_KEY')
    cse_id = os.environ.get('GOOGLE_CSE_ID')
    
    print(f"DEBUG: Step 3 - API key exists: {bool(api_key)}")
    print(f"DEBUG: Step 3 - CSE ID exists: {bool(cse_id)}")
    
    if not api_key or not cse_id:
        print("ERROR: Google Custom Search API credentials not found!")
        print("Please set GOOGLE_API_KEY and GOOGLE_CSE_ID in your .env file")
        print("\nTo set up:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Enable Custom Search API")
        print("3. Create API key")
        print("4. Go to https://cse.google.com/cse/")
        print("5. Create custom search engine (use '*' to search entire web)")
        print("6. Add credentials to .env file")
        print("\nNote: Gemini Pro plan doesn't include programmatic web search API access.")
        print("You still need Custom Search API for web scraping functionality.")
        return {'sources': []}
    
    print("DEBUG: Step 4 - Using Google Custom Search API")
    
    # Make API request with timeout
    print("DEBUG: Step 5 - About to make API request")
    try:
        api_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': api_key,
            'cx': cse_id,
            'q': query,
            'num': min(source_count, 10)  # API limit is 10 per request
        }
        
        print(f"DEBUG: Step 6 - Making API request to {api_url}")
        print(f"DEBUG: Step 6 - Query: {query}")
        print(f"DEBUG: Step 6 - Using CSE ID: {cse_id}")
        
        print("DEBUG: Step 7 - About to call requests.get")
        response = requests.get(api_url, params=params, timeout=10)  # 10 second timeout
        print(f"DEBUG: Step 8 - Got response, status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"ERROR: API request failed with status {response.status_code}")
            print(f"ERROR: Response: {response.text}")
            return {'sources': []}
            
        print("DEBUG: Step 9 - Parsing JSON response")
        data = response.json()
        print(f"DEBUG: Step 10 - API returned {len(data.get('items', []))} results")
        
        # Extract links from API response
        print("DEBUG: Step 11 - Extracting links from API response")
        links = []
        for item in data.get('items', []):
            link = item.get('link')
            if link:
                links.append(link)
                print(f"DEBUG: Added link from API: {link}")
        
        print(f"DEBUG: Step 12 - Total links from API: {len(links)}")
        
    except requests.exceptions.Timeout:
        print("ERROR: API request timed out after 10 seconds")
        return {'sources': []}
    except requests.exceptions.RequestException as e:
        print(f"ERROR: API request failed: {e}")
        return {'sources': []}
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return {'sources': []}

    print("DEBUG: Step 13 - Starting link filtering")
    # Filter links (same logic as before)
    exclude_list = ["google", "facebook", "twitter", "instagram", "youtube", "tiktok","quora"]
    filtered_links = []
    
    for link in links:
        try:
            hostname = urlparse(link).hostname
            print(f"DEBUG: Checking link {link}, hostname: {hostname}")
            if hostname and len(hostname.split('.')) >= 2 and hostname.split('.')[1] not in exclude_list:
                filtered_links.append(link)
                print(f"DEBUG: Link passed filter: {link}")
            else:
                print(f"DEBUG: Link filtered out: {link}")
        except Exception as e:
            print(f"DEBUG: Error parsing link {link}: {e}")
            pass
    
    print(f"DEBUG: Step 14 - After filtering, {len(filtered_links)} links remain")

    final_links = filtered_links[:source_count]
    print(f"DEBUG: Step 15 - Final links to process: {len(final_links)}")

    # SCRAPE TEXT FROM LINKS
    sources = []

    for i, link in enumerate(final_links):
        print(f'DEBUG: Processing link {i+1}/{len(final_links)}: {link}')
        source_text = None
        try:
            for attempt in range(5):
                print(f"DEBUG: Trafilatura attempt {attempt + 1} for {link}")
                downloaded = trafilatura.fetch_url(link)
                source_text = trafilatura.extract(downloaded)
                if source_text is not None:
                    print(f"DEBUG: Trafilatura extraction successful, length: {len(source_text)}")
                    break
                
                print(f'DEBUG: Trafilatura attempt {attempt + 1} failed for {link}')
                time.sleep(4)
            
            if source_text is None:
                print(f"DEBUG: All trafilatura attempts failed for {link}, skipping")
                continue
                
            print(f"DEBUG: Making requests.get call for {link}")
            response = requests.get(link, timeout=15)
            print(f"DEBUG: requests.get successful, status: {response.status_code}")
        except Exception as e:
            print(f"DEBUG: Exception during link processing {link}: {e}")
            continue
            
        print('DEBUG: Link Loaded successfully')
        html = response.text
        print(f"DEBUG: HTML content length: {len(html)}")
        
        html_text = None
        page_title = None
        
        try:
            print("DEBUG: Attempting readabilipy extraction")
            readable = simple_json_from_html_string(html)
            html_text = readable['content']
            page_title = readable.get('title', 'No title')
            print(f"DEBUG: Readabilipy successful, content length: {len(html_text)}, title: {page_title}")
        except Exception as e:
            print(f"DEBUG: Readabilipy failed: {e}")
            try:
                print("DEBUG: Attempting fallback extraction")
                from readabilipy.extractors import extract_title
                page_title = extract_title(html)
                html_text = str(html)
                print(f"DEBUG: Fallback successful, title: {page_title}")
            except Exception as e2:
                print(f"DEBUG: Fallback also failed: {e2}")
                continue
                
        if html_text is None or len(html_text) < 400:
            print(f"DEBUG: Content too short ({len(html_text) if html_text else 0}), skipping")
            continue
            
        print(f"DEBUG: Content length: {len(html_text)}")

        soup = BeautifulSoup(html_text, 'html.parser')

        if source_text:
            print(f"DEBUG: Processing source_text, length: {len(source_text)}")
            source_text = clean_source_text(source_text)
            print('DEBUG: Going to call Gemini API')
            raw_source = source_text
            
            # Try Gemini API with improved error handling
            # try:
            #     print('DEBUG: Attempting Gemini API call')
            #     cleaned_source = clean_source_gpt35(source_text[:8000])
            #     source_text = cleaned_source
            #     print('DEBUG: Gemini API Called successfully')
            # except Exception as e:
            #     print(f'DEBUG: Gemini API failed: {e}, using original text')
            #     # Use original text if Gemini fails
            #     source_text = source_text
            
            # Skip Gemini API for now - use raw scraped content
            print('DEBUG: Skipping Gemini API, using raw scraped content')
            # source_text is already cleaned by clean_source_text()
            
            summary_text = summarize_text_identity(source_text, query)
            sources.append({'url': link, 'text': f'Title: {page_title}\nSummary:' + summary_text, 'raw_source' : raw_source, 'source' : source_text, 'summary' : summary_text})
        else:
            print("DEBUG: No source_text available, skipping processing")
            
        if len(sources) == source_count:
            print(f"DEBUG: Reached target source count ({source_count}), breaking")
            break
            
    print(f"DEBUG: Final sources count: {len(sources)}")
    return {'sources': sources}
    
if __name__ == '__main__':
    print("DEBUG: Main execution started")
    import sys
    print("DEBUG: About to call search_handler first time")
    search_handler('What is Generative Engine Optimization?')
    print("DEBUG: First search_handler call completed")
    import json
    print("DEBUG: About to call search_handler second time")
    print(json.dumps(search_handler(sys.argv[1]), indent = 2))
    print("DEBUG: Script execution completed")
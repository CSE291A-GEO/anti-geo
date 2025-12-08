"""
Generate real world dataset from AI mode search results.
This script loads queries from combined_queries_100_new.csv, gets top 10 results
from Google AI mode search, and scrapes the website content.
Uses async/threading with 100 workers for parallel processing.
"""

import json
import csv
import os
import time
import re
import requests
from bs4 import BeautifulSoup
from serpapi.google_search import GoogleSearch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configuration
SERP_API_KEY = "ed18fadc95e7c23220e9aaf54f7638fb7d3388d067ff35fa503aac35c1d2c297"
QUERIES_CSV_FILE = "combined_queries_100_new.csv"
OUTPUT_FILE = "real_world_dataset.json"
SEARCHED_QUERIES_FILE = "searched_queries.txt"
MAX_RESULTS = 10  # Top 10 results from AI mode search
MAX_WORKERS = 100  # Number of parallel workers

# Thread-safe file writing
file_lock = threading.Lock()


def load_queries_from_csv(csv_path=QUERIES_CSV_FILE):
    """
    Reads queries from CSV file.

    Parameters:
    - csv_path: path to the CSV file

    Returns:
    - List of queries
    """
    queries = []
    with open(csv_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            query = row.get('query', '').strip()
            if query:
                queries.append(query)
    return queries


def get_ai_mode_top_results(ai_mode_results, max_results=MAX_RESULTS):
    """
    Extract top N references from AI mode search results.
    
    Args:
        ai_mode_results (dict): AI mode search results from SERPAPI
        max_results (int): Maximum number of results to return
        
    Returns:
        list: List of reference dictionaries with 'link' and 'title' keys
    """
    references = []
    if ai_mode_results and 'references' in ai_mode_results:
        refs = ai_mode_results['references'][:max_results]
        for ref in refs:
            references.append({
                'link': ref.get('link', ''),
                'title': ref.get('title', '')
            })
    return references


def scrape_website_content(url):
    """
    Scrape and extract text content from a website URL.
    
    Args:
        url (str): URL to scrape
        
    Returns:
        str: Extracted text content, or empty string if failed
    """
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
            script_or_style.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up multiple spaces/newlines
        clean_text = re.sub(r'\s+', ' ', text)
        
        return clean_text
    except Exception as e:
        return ""  # Return empty string on failure


def process_single_query(query, query_index, total_queries):
    """
    Process a single query: API call -> get top 10 -> scrape websites -> return result.
    
    Args:
        query (str): Query string
        query_index (int): Index of query (for logging)
        total_queries (int): Total number of queries (for logging)
        
    Returns:
        dict: Result dictionary with 'query' and 'website_contents', or None if failed
    """
    print(f"[{query_index + 1}/{total_queries}] Processing: {query[:60]}...", flush=True)
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # API call
            print(f"  [{query_index + 1}/{total_queries}] Querying Google AI Mode...", flush=True)
            ai_mode_params = {
                "engine": "google_ai_mode",
                "q": query,
                "api_key": SERP_API_KEY
            }
            
            ai_mode_search = GoogleSearch(ai_mode_params)
            ai_mode_results = ai_mode_search.get_dict()
            
            # Get top 10 references
            top_references = get_ai_mode_top_results(ai_mode_results, MAX_RESULTS)
            print(f"  [{query_index + 1}/{total_queries}] Found {len(top_references)} references", flush=True)
            
            # Scrape website content for each reference
            website_contents = []
            for j, ref in enumerate(top_references):
                url = ref.get('link', '')
                if url:
                    print(f"  [{query_index + 1}/{total_queries}] Scraping {j + 1}/{len(top_references)}: {url[:50]}...", flush=True)
                    content = scrape_website_content(url)
                    if content:
                        website_contents.append(content)
                    time.sleep(0.5)  # Small delay between requests
                else:
                    website_contents.append("")  # Empty content if no URL
            
            result_entry = {
                "query": query,
                "website_contents": website_contents
            }
            
            print(f"  [{query_index + 1}/{total_queries}] ✓ Completed: {len(website_contents)} websites scraped", flush=True)
            return result_entry
            
        except Exception as e:
            retry_count += 1
            print(f"  [{query_index + 1}/{total_queries}] Error (attempt {retry_count}/{max_retries}): {e}", flush=True)
            if retry_count < max_retries:
                wait_time = 15 * retry_count  # Exponential backoff
                print(f"  [{query_index + 1}/{total_queries}] Retrying after {wait_time} seconds...", flush=True)
                time.sleep(wait_time)
            else:
                print(f"  [{query_index + 1}/{total_queries}] ✗ Failed after {max_retries} attempts", flush=True)
                return None
    
    return None


def append_result_to_file(result_entry, output_file):
    """
    Thread-safe append of a single result entry to JSON file.
    
    Args:
        result_entry (dict): Single result dictionary
        output_file (str): Output file path
    """
    with file_lock:
        # Read existing data if file exists
        existing_data = []
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                existing_data = []
        
        # Append new entry
        existing_data.append(result_entry)
        
        # Write back
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)


def append_result_to_csv(result_entry, csv_output_file):
    """
    Thread-safe append of a single result entry to CSV file.
    
    Args:
        result_entry (dict): Single result dictionary
        csv_output_file (str): CSV output file path
    """
    with file_lock:
        file_exists = os.path.exists(csv_output_file)
        
        with open(csv_output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['query'] + [f'website{i}content' for i in range(1, MAX_RESULTS + 1)])
            
            # Write data row
            query = result_entry['query']
            contents = result_entry['website_contents']
            # Pad with empty strings if less than MAX_RESULTS
            while len(contents) < MAX_RESULTS:
                contents.append("")
            writer.writerow([query] + contents[:MAX_RESULTS])


def mark_query_completed(query):
    """
    Thread-safe marking of query as completed.
    
    Args:
        query (str): Query string
    """
    with file_lock:
        with open(SEARCHED_QUERIES_FILE, "a", encoding="utf-8") as q:
            q.write(query + "\n")


def filter_queries(queries, filename=SEARCHED_QUERIES_FILE):
    """
    Filter out queries that have already been scraped.
    
    Args:
        queries (list): List of query strings to filter.
        filename (str): Path to the file containing previously scraped queries.
        
    Returns:
        list: A filtered list of queries excluding those found in the scraped queries file.
    """
    print(f"Filtering out already scraped queries...", flush=True)
    file_queries = set()
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            file_queries = set(line.strip() for line in f if line.strip())
    
    filtered_queries = [q for q in queries if q not in file_queries]
    
    if len(queries) == len(filtered_queries):
        print(f"All queries are new. None were filtered out.", flush=True)
    else:
        print(f"Found {len(queries) - len(filtered_queries)} queries already scraped", flush=True)
    
    return filtered_queries


def generate_dataset():
    """
    Main function to generate the dataset using parallel workers.
    """
    print("="*80, flush=True)
    print("Starting dataset generation with parallel workers", flush=True)
    print("="*80, flush=True)
    
    # Load queries from CSV
    print("Loading queries from CSV...", flush=True)
    queries = load_queries_from_csv()
    print(f"Loaded {len(queries)} queries from {QUERIES_CSV_FILE}", flush=True)
    
    # Filter already searched queries
    filtered_queries = filter_queries(queries)
    print(f"Total queries to process: {len(filtered_queries)}", flush=True)
    
    if len(filtered_queries) == 0:
        print("No new queries to process.", flush=True)
        return
    
    # Initialize output files
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)
        print(f"Created new {OUTPUT_FILE}", flush=True)
    
    csv_file = OUTPUT_FILE.replace('.json', '.csv')
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['query'] + [f'website{i}content' for i in range(1, MAX_RESULTS + 1)])
        print(f"Created new {csv_file}", flush=True)
    
    print(f"\nStarting parallel processing with {MAX_WORKERS} workers...", flush=True)
    print("="*80, flush=True)
    
    # Process queries in parallel
    completed_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_query = {
            executor.submit(process_single_query, query, i, len(filtered_queries)): (query, i)
            for i, query in enumerate(filtered_queries)
        }
        
        # Process results as they complete
        for future in as_completed(future_to_query):
            query, query_index = future_to_query[future]
            try:
                result = future.result()
                if result:
                    # Save result immediately
                    append_result_to_file(result, OUTPUT_FILE)
                    append_result_to_csv(result, csv_file)
                    mark_query_completed(query)
                    completed_count += 1
                    print(f"\n✓ [{completed_count}/{len(filtered_queries)}] Saved: {query[:60]}...", flush=True)
                else:
                    failed_count += 1
                    print(f"\n✗ [{failed_count} failed] Query failed: {query[:60]}...", flush=True)
            except Exception as e:
                failed_count += 1
                print(f"\n✗ [{failed_count} failed] Exception for query '{query[:60]}...': {e}", flush=True)
    
    print("\n" + "="*80, flush=True)
    print("Dataset generation complete!", flush=True)
    print(f"Total queries processed: {len(filtered_queries)}", flush=True)
    print(f"Successfully completed: {completed_count}", flush=True)
    print(f"Failed: {failed_count}", flush=True)
    print(f"Results saved to {OUTPUT_FILE} and {csv_file}", flush=True)
    print("="*80, flush=True)


if __name__ == "__main__":
    generate_dataset()

# Anti-GEO: Dataset Generation & Analysis

This repository hosts the codebase for generating datasets to analyze Generative Engine Optimization (GEO). The primary goal is to identify and study "GEO-optimized" content—sources that appear prominently in AI-generated search summaries (like Google AI Overviews) but are ranked lower or are entirely absent from traditional organic search results.

> **Note:** The `warning_prompts` directory and the `Detect_misused_GEO_using_exaggerations` script were exploratory and paused for now.

## Project Structure

  * **`Anti_GEO_Dataset_Generation.ipynb`**: The core Jupyter Notebook. It handles fetching search results via SerpApi, identifying GEO-optimized sources, and scraping their HTML content.
  * **`anti_geo_dataset.tsv.zip`**: The compressed dataset containing raw search results, including organic rankings, AI Overviews, and AI Mode responses.
  * **`scraped_data.jsonl`**: The output file containing the scraped content from identified GEO-optimized URLs.
  * **`combined_queries_1000.csv`**: Source file containing the search queries used for dataset generation.
  * **`searched_queries.txt`**: A log file tracking processed queries to prevent redundant API calls.
  * **`requirements.txt`**: List of Python dependencies.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/CSE291A-GEO/anti-geo.git
    cd anti-geo
    ```

2.  **Install dependencies:**
    Ensure you have Python 3.x installed. It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

    *The notebook also includes inline commands to install specific packages like `serpapi` and `google-search-results` if missing.*

3.  **API Key Configuration:**
    The dataset generation requires a valid **SerpApi** key to fetch Google Search results.

      * Open `Anti_GEO_Dataset_Generation.ipynb`.
      * Locate the variable `SERP_API_KEY` (usually in the second code cell).
      * Replace the placeholder or existing key with your own valid API key.

## Usage

To generate the dataset and scrape GEO-optimized content, run the `Anti_GEO_Dataset_Generation.ipynb` notebook.

The notebook executes the following workflow:

1.  **Query Loading:** Loads queries from `combined_queries_1000.csv`.
2.  **Data Fetching:** Batches queries and calls SerpApi to retrieve:
      * Organic Google Search results.
      * Google AI Overview (if triggered).
      * Google AI Mode results.
3.  **Data Storage:** Saves the raw results into `anti_geo_dataset.tsv.zip`.
4.  **Analysis (GEO Detection):** Compares AI citations against organic search rankings. Sources cited by AI but missing from the top 10 organic results are flagged as "GEO-optimized."
5.  **Content Scraping:** Fetches and cleans the HTML text from these specific GEO-optimized URLs and appends the data to `scraped_data.jsonl`.

### Running the Notebook

1.  Launch Jupyter:
    ```bash
    jupyter notebook
    ```
2.  Open `Anti_GEO_Dataset_Generation.ipynb`.
3.  Execute the cells sequentially.
      * *Note on Performance:* The scraping function includes a `time.sleep()` delay between requests to be polite to servers. Processing a large number of queries may take significant time.

## Output Data Format

The final scraped data (`scraped_data.jsonl`) contains entries in the following JSON structure:

```json
{
  "query": "original search query",
  "ai_mode": [
    {
      "source_url": "https://example.com/article",
      "ge_rank": 7,       // Rank in the Generative Engine (AI) citation list
      "se_rank": -1,      // Rank in Organic Search (-1 indicates not in top results)
      "html_content": "<html>...</html>",
      "clean_content": "Extracted text content..."
    }
  ],
  "ai_overview": [
    ... // Similar structure for AI Overview results
  ]
}
```

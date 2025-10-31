from dotenv import load_dotenv
# import google.generativeai as genai  # Commented out - switched to Ollama
import ollama
import json
import time
import os
import pickle
import uuid

"""Load env variables"""
load_dotenv()
# genai.configure(api_key=os.environ.get('GEMINI_API_KEY', ''))  # Commented out - switched to Ollama

# Ollama configuration
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5:3b')

# ORIGINAL PROMPT (for reference - works with larger models like GPT/Gemini):
# query_prompt = """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Français, Español, Deutsch, or 日本語 should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.
#
# Question: {query}
#
# Search Results:
# {source_text}
# """

# NEW PROMPT (optimized for smaller Ollama models like qwen2.5:3b):
query_prompt = """CRITICAL REQUIREMENT: You MUST add [1], [2], [3], etc. citations after EVERY sentence!

Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Français, Español, Deutsch, or 日本語 should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible.

CITATION FORMAT (MANDATORY):
- Every sentence in the answer MUST be immediately followed by an in-line citation to the search result(s)
- The cited search result(s) should fully support ALL the information in the sentence
- Search results MUST be cited using [index] format (e.g., [1], [2], [3])
- When citing several search results, use [1][2][3] format rather than [1, 2, 3]
- You can use multiple search results to respond comprehensively while avoiding irrelevant search results

EXAMPLE (showing correct citation format):
Question: What is the capital of France?
Search Results:
### Source 1: Paris is the capital and largest city of France.
### Source 2: France is a country in Western Europe.

CORRECT Answer:
Paris is the capital of France[1]. France is located in Western Europe[2]. Paris is also the largest city in the country[1].

WRONG Answer (missing citations - DO NOT do this):
Paris is the capital of France. France is in Europe.

Now answer this question with [index] citations after EVERY sentence:

Question: {query}

Search Results:
{source_text}

Answer (with mandatory [1], [2], [3] citations after every sentence):"""

# Remove Bedrock client

def generate_answer(query, sources, num_completions, temperature = 0.5, verbose = False, model = None):
    # Use Ollama model
    if model is None:
        model = OLLAMA_MODEL
    
    source_text = '\n\n'.join(['### Source '+str(idx+1)+':\n'+source + '\n\n\n' for idx, source in enumerate(sources)])
    prompt = query_prompt.format(query = query, source_text = source_text)

    responses = []
    for _ in range(num_completions):
        while True:
            try:
                print(f'Running Ollama Model: {model}')
                resp = ollama.generate(
                    model=model,
                    prompt=prompt,
                    options={
                        'temperature': float(temperature),
                        'top_p': 1.0,
                        'num_predict': 1024,  # equivalent to max_output_tokens
                    }
                )
                content = (resp['response'] or '').strip()
                responses.append(content + '\n')
                print('Response Done')
                break
            except Exception as e:
                print('Error in calling Ollama API', e)
                # If model not found, try to pull it
                if 'not found' in str(e).lower() or 'does not exist' in str(e).lower():
                    print(f'Model {model} not found, attempting to pull...')
                    try:
                        ollama.pull(model)
                        print(f'Successfully pulled {model}')
                        continue
                    except:
                        print(f'Failed to pull {model}, using default model')
                        model = 'qwen2.5:3b'
                        continue
                time.sleep(5)  # Shorter wait for local model
                continue

    return responses
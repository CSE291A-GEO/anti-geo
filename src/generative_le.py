from dotenv import load_dotenv
import google.generativeai as genai
import json
import time
import os
import pickle
import uuid

"""Load env variables and configure Gemini"""
load_dotenv()
genai.configure(api_key=os.environ.get('GEMINI_API_KEY', ''))

query_prompt = """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Français, Español, Deutsch, or 日本語 should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {query}

Search Results:
{source_text}
"""

# Remove Bedrock client; using Gemini instead

def generate_answer(query, sources, num_completions, temperature = 0.5, verbose = False, model = 'gemini-2.5-pro'):

    source_text = '\n\n'.join(['### Source '+str(idx+1)+':\n'+source + '\n\n\n' for idx, source in enumerate(sources)])
    prompt = query_prompt.format(query = query, source_text = source_text)

    model_client = genai.GenerativeModel(model)
    responses = []
    for _ in range(num_completions):
        while True:
            try:
                print('Running Google Gemini Model')
                resp = model_client.generate_content(
                    prompt,
                    generation_config={
                        'temperature': float(temperature),
                        'top_p': 1.0,
                        'max_output_tokens': 1024,
                    }
                )
                content = (resp.text or '').strip()
                responses.append(content + '\n')
                print('Response Done')
                break
            except Exception as e:
                print('Error in calling Gemini API', e)
                # Fallback to 2.5 flash model if the specified one is unavailable
                if '404' in str(e) or 'not found' in str(e).lower():
                    model = 'gemini-2.5-flash'
                    model_client = genai.GenerativeModel(model)
                    continue
                time.sleep(15)
                continue

    return responses
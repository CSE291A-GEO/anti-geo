from dotenv import load_dotenv
import google.generativeai as genai
# import ollama  # Commented out - using Gemini
import json
import time
import os
import pickle
import uuid

"""Load env variables"""
load_dotenv()
genai.configure(api_key=os.environ.get('GEMINI_API_KEY', ''))

# Gemini configuration (reverted from Ollama)
# OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5:3b')

query_prompt = """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Français, Español, Deutsch, or 日本語 should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {query}

Search Results:
{source_text}
"""

# Remove Bedrock client

def generate_answer(query, sources, num_completions, temperature = 0.5, verbose = False, model = None):
    # Use Gemini model
    if model is None:
        model = genai.GenerativeModel('gemini-2.5-pro')
    
    source_text = '\n\n'.join(['### Source '+str(idx+1)+':\n'+source + '\n\n\n' for idx, source in enumerate(sources)])
    prompt = query_prompt.format(query = query, source_text = source_text)

    responses = []
    for _ in range(num_completions):
        while True:
            try:
                print(f'Running Gemini Model')
                resp = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=float(temperature),
                        top_p=1.0,
                        max_output_tokens=1024,
                    )
                )
                
                # Try to get text from response
                try:
                    content = (resp.text or '').strip()
                except Exception as text_error:
                    # If response.text fails, try to get from candidates
                    print(f'Warning: response.text failed ({text_error}), trying candidates')
                    if hasattr(resp, 'candidates') and resp.candidates:
                        try:
                            content = resp.candidates[0].content.parts[0].text.strip()
                        except:
                            print('Error: Could not extract text from candidates either')
                            content = ''
                    else:
                        print('Error: No candidates available, response may be blocked')
                        content = ''
                
                if content:
                    responses.append(content + '\n')
                    print('Response Done')
                    break
                else:
                    print('Warning: Empty response, retrying...')
                    time.sleep(10)
                    continue
                    
            except Exception as e:
                print('Error in calling Gemini API', e)
                time.sleep(10)
                continue

    return responses
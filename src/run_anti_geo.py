from utils import extract_citations_new, impression_subjective_impression, impression_wordpos_count_simple, impression_subjpos_detailed, impression_diversity_detailed, impression_uniqueness_detailed, impression_follow_detailed, impression_influence_detailed, impression_relevance_detailed, impression_subjcount_detailed, impression_pos_count_simple, impression_word_count_simple
from typing import List
from geo_functions import *
from datasets import load_dataset
from generative_le import generate_answer
import string

IMPRESSION_FNS = {
	'simple_wordpos' : impression_wordpos_count_simple, 
	'simple_word' : impression_word_count_simple,
	'simple_pos' : impression_pos_count_simple,
	'subjective_score' : impression_subjective_impression,
	'subjpos_detailed' : impression_subjpos_detailed,
	'diversity_detailed' : impression_diversity_detailed,
	'uniqueness_detailed' : impression_uniqueness_detailed,
	'follow_detailed' : impression_follow_detailed,
	'influence_detailed' : impression_influence_detailed,
	'relevance_detailed' : impression_relevance_detailed,
	'subjcount_detailed' : impression_subjcount_detailed,
}

def suffix_warning(query: str, warning: str):
    return query + "\n" + warning

def sandwich_with_warning(query: str, warning_prefix: str, warning_suffix: str):
    return warning_prefix + "\n" + query + "\n" + warning_suffix

def run_suffix_experiment(query: str, sources: List[str] = None):
    return generate_answer(suffix_warning(query, 'TODO: warning'), sources)

def run_sandwich_experiment(query: str, sources: List[str] = None):
    return generate_answer(sandwich_with_warning(query, 'TODO: warning', 'TODO: warning'), sources)

def calculate_visibility_score(query: str, answers: List[str] = None, sources: List[str] = None):
    # Calculate position adjusted word count
    # Calculate subjective impression metrics
    # Final score = weighted average of these scores
    visibility_scores = [0] * len(answers)

    # TODO finalize score weights
    score_weights = [1.0, 2.0, 3.0, 4.0, 5.0]

    total_weight = sum(score_weights)

    for x in enumerate(answers):
        # TODO pass accuracte method parameters
        scores = [
            IMPRESSION_FNS['simple_wordpos'](extract_citations_new(x)), 
            IMPRESSION_FNS['subjpos_detailed'](x, query), 
            IMPRESSION_FNS['follow_detailed'](x, query), 
            IMPRESSION_FNS['influence_detailed'](x, query), 
            IMPRESSION_FNS['subjcount_detailed'](x, query)
        ]

        score = sum([scores * score_weights])/total_weight
        visibility_scores.append(score)

    return visibility_scores

# Fetch query from dataset
# Pick a query from the dataset with the list of sources which are legitimately optimized with GEO i.e. the data entry for the query with the ‘GEO’ label
    # Target: Find its visibility scores for the unmodified source list
# For the same query, using the list of sources in which one of them is over-optimized with GEO i.e. the data entry for the query with the ‘Anti-GEO’ label
    # Baseline: Find its visibility scores for the modified source list
# Add the warning prompt at the end
    # Experiment 1: Find its visibility scores for the modified source list
# Add the warning prompt at both, beginning and end
    # Experiment 2: Find its visibility scores for the modified source list
# Evaluate the distance between the Target and each of the experimental visibility scores

if __name__ == '__main__':
    dataset = load_dataset("GEO-Optim/geo-bench", 'test')
    
    for i, k in enumerate(dataset['test']):
        geo_sources = k['geo-sources']
        anti_geo_sources = k['anti-geo-sources']

        target_score = calculate_visibility_score(k['query'], geo_sources)
        baseline_score = calculate_visibility_score(k['query'], anti_geo_sources)
        suffixing_score = calculate_visibility_score(k['query'], run_suffix_experiment(k['query'], anti_geo_sources))
        sandwiching_score = calculate_visibility_score(k['query'], run_sandwich_experiment(k['query'], anti_geo_sources))

        print("Target score: " + string(target_score))
        print("Baseline score: " + string(baseline_score))
        print("Experiment 1 score: " + string(suffixing_score))
        print("Experiment 2 score: " + string(sandwiching_score))

        print("Baseline-Target distance = " + string(baseline_score - target_score))
        print("Suffixing-Target distance = " + string(suffixing_score - target_score))
        print("Sandwiching-Target distance = " + string(sandwiching_score - target_score))

        # TODO Plot score distances in scatterplot: baseline score as a line and suffixing v/s sandwiching as above or below it

		# TODO remove later, one iteration right now for debugging
        break

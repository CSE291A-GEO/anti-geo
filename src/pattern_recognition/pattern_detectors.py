"""
Rule-Based Pattern Detectors for GEO Detection

This module implements rule-based detection methods for each GEO pattern type,
using feature extraction and scoring heuristics to identify GEO-optimized content.
"""

import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    nltk_available = True
except ImportError:
    nltk_available = False
    # Fallback tokenization
    def sent_tokenize(text):
        return re.split(r'[.!?]+', text)
    def word_tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())


# --- Rule-based Pattern Scoring Functions ---

def score_geo_struct_001(text: str) -> float:
    """
    GEO_STRUCT_001: Excessive Q&A blocks
    Scoring: Q&A Density & Repetition
    
    Core Detection Metrics:
    1. Q&A Density: Calculate the ratio of Q/A pairs to total paragraph count
    2. Repetitiveness: Identify core entity/keyword and calculate average mentions per Q/A pair
    """
    # 1. Q&A Density: Calculate ratio of Q/A pairs to total paragraphs
    qa_pattern = r'(?:^|\n)\s*[Qq]:\s*[^\n]+\s*[Aa]:\s*[^\n]+'
    qa_matches = len(re.findall(qa_pattern, text, re.MULTILINE))
    
    # Also check for questions ending with '?'
    question_pattern = r'[^\n]*\?[^\n]*'
    questions = re.findall(question_pattern, text)
    
    # Count paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    total_paragraphs = len(paragraphs) if paragraphs else 1
    
    # Q&A density score: High ratio → High score
    qa_density = (qa_matches + len(questions)) / max(total_paragraphs, 1)
    qa_density_score = min(qa_density / 2.0, 1.0)  # Normalize, high density = high score
    
    # 2. Repetitiveness: Find core entity/keyword and count per Q/A pair
    # Extract entities (capitalized words/phrases that appear frequently)
    words = word_tokenize(text)
    # Find capitalized multi-word phrases
    capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    if capitalized_phrases:
        entity_counter = Counter(capitalized_phrases)
        if entity_counter:
            most_common_entity = entity_counter.most_common(1)[0][0]
            entity_count = text.count(most_common_entity)
            repetitiveness = entity_count / max(qa_matches, 1) if qa_matches > 0 else entity_count
            repetitiveness_score = min(repetitiveness / 3.0, 1.0)  # High number → High score
        else:
            repetitiveness_score = 0.0
    else:
        repetitiveness_score = 0.0
    
    # Combine scores (weighted average)
    final_score = 0.6 * qa_density_score + 0.4 * repetitiveness_score
    return final_score


def score_geo_struct_002(text: str) -> float:
    """
    GEO_STRUCT_002: Over-Chunking/Simplification
    Scoring: Simplification/Chunking Score
    
    Core Detection Metrics:
    1. Average Sentence Length (ASL): Very low ASL (< 5 words) → High score
    2. Bullet/List Density: Ratio of list items to total tokens
    3. Complexity Check: Use Flesch-Kincaid-like readability score (extremely simple → High score)
    """
    # 1. Average Sentence Length (ASL)
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0
    
    total_words = sum(len(word_tokenize(s)) for s in sentences)
    asl = total_words / len(sentences)
    
    # Very low ASL (< 5 words) = high score
    if asl < 3:
        asl_score = 1.0
    elif asl < 5:
        asl_score = 0.8
    elif asl < 8:
        asl_score = 0.5
    else:
        asl_score = max(0.0, 1.0 - (asl - 8) / 10.0)
    
    # 2. Bullet/List Density: Ratio of list items to total tokens
    list_patterns = [
        r'[•\-\*\+]\s+',  # Bullet points
        r'\d+\.\s+',      # Numbered lists
        r'<li>',          # HTML list items
        r'^\s*[-*+•]\s+', # Markdown lists
    ]
    list_count = sum(len(re.findall(pattern, text)) for pattern in list_patterns)
    total_tokens = len(word_tokenize(text))
    list_density = list_count / max(total_tokens, 1)
    list_density_score = min(list_density * 10, 1.0)  # High ratio → High score
    
    # 3. Complexity Check: Simple Flesch-like readability (simplified)
    # Lower complexity = simpler text = higher GEO score
    avg_words_per_sentence = asl
    avg_syllables_per_word = 1.5  # Simplified estimate (most words have 1-2 syllables)
    # Simplified readability: lower score = simpler = more GEO-like
    # Extremely low (simple) score → High GEO score
    complexity_score = max(0.0, 1.0 - (avg_words_per_sentence * avg_syllables_per_word) / 20.0)
    
    # Combine scores
    final_score = 0.4 * asl_score + 0.3 * list_density_score + 0.3 * complexity_score
    return final_score


def score_geo_struct_003(text: str) -> float:
    """
    GEO_STRUCT_003: Header Stuffing
    Scoring: Header Repetition Score
    
    Core Detection Metrics:
    1. Header N-gram Overlap: Jaccard similarity between adjacent headers (high similarity → High score)
    2. Header Keyword Density: Calculate keyword density of primary entity within headers (very high → High score)
    """
    # Extract headers (markdown style: ##, ###, etc. or HTML: <h2>, <h3>, etc.)
    header_pattern = r'(?:^|\n)\s*(?:#{1,6}\s+|<h[1-6]>)([^\n]+)(?:</h[1-6]>)?'
    headers = re.findall(header_pattern, text, re.MULTILINE | re.IGNORECASE)
    
    if len(headers) < 2:
        return 0.0
    
    # 1. Header N-gram Overlap: Jaccard similarity between adjacent headers
    similarities = []
    for i in range(len(headers) - 1):
        h1_words = set(word_tokenize(headers[i]))
        h2_words = set(word_tokenize(headers[i + 1]))
        if h1_words or h2_words:
            intersection = len(h1_words & h2_words)
            union = len(h1_words | h2_words)
            jaccard = intersection / union if union > 0 else 0.0
            similarities.append(jaccard)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    # High similarity → High score
    
    # 2. Header Keyword Density: Find most common words across headers
    all_header_words = []
    for header in headers:
        all_header_words.extend(word_tokenize(header))
    
    if all_header_words:
        word_counter = Counter(all_header_words)
        # Filter out common stop words
        if nltk_available:
            try:
                stop_words = set(stopwords.words('english'))
            except LookupError:
                stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        else:
            stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Find most frequent non-stopword
        frequent_words = [(w, c) for w, c in word_counter.items() if w not in stop_words and len(w) > 3]
        if frequent_words:
            most_common_word, count = max(frequent_words, key=lambda x: x[1])
            keyword_density = count / len(headers)
            keyword_density_score = min(keyword_density, 1.0)  # Very high density → High score
        else:
            keyword_density_score = 0.0
    else:
        keyword_density_score = 0.0
    
    # Combine scores
    final_score = 0.6 * avg_similarity + 0.4 * keyword_density_score
    return final_score


def score_geo_semantic_004(text: str) -> float:
    """
    GEO_SEMANTIC_004: Entity Over-Attribution
    Scoring: Entity Repetition & Verbosity
    
    Core Detection Metrics:
    1. NER Detection: Find people/organizations with titles/affiliations
    2. Attribution Repetition: Count entity mentions with title/affiliation within 5-sentence window (high frequency → High score)
    3. Assertiveness: Score sentences for words like definitive, guaranteed, superior, only, best quality (high usage → High score)
    """
    # 1. NER Detection: Find entities with titles/affiliations
    # Pattern: "Dr. Name, the title at Organization,"
    entity_pattern = r'(?:Dr\.|Professor|Mr\.|Mrs\.|Ms\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+(?:the\s+)?[^,]+(?:at\s+[A-Z][^,]+)?,'
    entities = re.findall(entity_pattern, text, re.IGNORECASE)
    
    # 2. Attribution Repetition: Count repetitions within 5 sentences
    sentences = sent_tokenize(text)
    if len(sentences) < 5:
        window_size = len(sentences)
    else:
        window_size = 5
    
    repetition_count = 0
    for i in range(len(sentences) - window_size + 1):
        window = ' '.join(sentences[i:i + window_size])
        # Count entity mentions in this window
        window_entities = re.findall(entity_pattern, window, re.IGNORECASE)
        if len(window_entities) > 2:  # More than 2 in 5 sentences = high repetition
            repetition_count += 1
    
    repetition_score = min(repetition_count / max(len(sentences) / window_size, 1), 1.0)
    # High frequency → High score
    
    # 3. Assertiveness: Count assertive words
    assertive_words = [
        'definitive', 'guaranteed', 'superior', 'only', 'best quality', 
        'best', 'superior', 'guarantee', 'proven', 'authentic', 
        'comprehensive', 'no more', 'will not regret', 'most valuable',
        'world-renowned', 'leading expert', 'chief inventor'
    ]
    assertive_count = sum(text.lower().count(word) for word in assertive_words)
    assertive_score = min(assertive_count / 5.0, 1.0)  # High usage → High score
    
    # Combine scores
    entity_score = min(len(entities) / 3.0, 1.0)  # Normalize entity count
    final_score = 0.3 * entity_score + 0.4 * repetition_score + 0.3 * assertive_score
    return final_score


def score_geo_semantic_005(text: str) -> float:
    """
    GEO_SEMANTIC_005: Unnatural Citation Embedding
    Scoring: Data/Quote Extraction Score
    
    Core Detection Metrics:
    1. Precision Number Detection: Find high-precision numbers (e.g., 95.2%, 87.3%) not in dates/prices (isolated facts → High score)
    2. Citation/Quote Isolation: Calculate contextual link of quotes/attributions (highly self-contained → High score)
    3. Source Credibility Check: Check for commonly faked source names (matches → High score)
    """
    # 1. Precision Number Detection: Find high-precision numbers (e.g., 95.2%, 87.3%)
    # Exclude dates and prices
    precision_pattern = r'\b\d{1,3}\.\d+%?'  # Numbers with decimal points
    numbers = re.findall(precision_pattern, text)
    
    # Filter out likely dates (years like 1999.5, 2023.1) and prices
    date_pattern = r'(19|20)\d{2}\.\d+'
    filtered_numbers = [n for n in numbers if not re.match(date_pattern, n)]
    
    # Count isolated precision numbers (unsupported facts)
    precision_score = min(len(filtered_numbers) / 5.0, 1.0)
    
    # 2. Citation/Quote Isolation: Find quotes and attribution phrases
    quote_pattern = r'["\'][^"\']+["\']|According to [^,\.]+|Research (?:from|by|indicates)|Studies show|Findings (?:from|indicate)'
    citations = re.findall(quote_pattern, text, re.IGNORECASE)
    
    # Check if citations are self-contained (isolated facts)
    # Simple heuristic: citations that are short sentences (low dependence on surrounding text)
    isolated_citations = 0
    for citation in citations:
        citation_sentences = sent_tokenize(citation)
        if len(citation_sentences) == 1 and len(word_tokenize(citation)) < 20:
            isolated_citations += 1
    
    isolation_score = min(isolated_citations / 3.0, 1.0)  # Highly self-contained → High score
    
    # 3. Source Credibility Check: Common fake source names
    fake_sources = [
        "Google's latest report",
        "Archaeological Survey of India",
        "DakshinaChitra Heritage Museum",
        "Indian Academy of Pediatrics",
        "National Motor Vehicle Safety Institute",
        "latest report",
        "recent studies",
        "research indicates",
        "statistics show",
        "data reveals",
        "findings indicate"
    ]
    fake_source_count = sum(1 for source in fake_sources if source.lower() in text.lower())
    fake_source_score = min(fake_source_count / 3.0, 1.0)  # Matches → High score
    
    # Combine scores
    final_score = 0.4 * precision_score + 0.3 * isolation_score + 0.3 * fake_source_score
    return final_score


def calculate_rule_based_geo_score(text: str, pattern_id: str) -> float:
    """
    Calculate rule-based GEO score for a specific pattern.
    
    Args:
        text: The text to analyze
        pattern_id: The pattern ID (e.g., 'GEO_STRUCT_001')
    
    Returns:
        Score between 0.0 and 1.0
    """
    pattern_scorers = {
        'GEO_STRUCT_001': score_geo_struct_001,
        'GEO_STRUCT_002': score_geo_struct_002,
        'GEO_STRUCT_003': score_geo_struct_003,
        'GEO_SEMANTIC_004': score_geo_semantic_004,
        'GEO_SEMANTIC_005': score_geo_semantic_005,
    }
    
    scorer = pattern_scorers.get(pattern_id)
    if scorer:
        return scorer(text)
    else:
        return 0.0


def calculate_all_pattern_scores(text: str, top_k: int = 3) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Calculate rule-based GEO scores for all patterns.
    
    Args:
        text: The text to analyze
        top_k: Number of top scoring patterns to return
    
    Returns:
        Tuple of (max_score, top_matches)
        - max_score: The highest score across all patterns
        - top_matches: List of (pattern_id, score) tuples, sorted by score descending
    """
    pattern_ids = [
        'GEO_STRUCT_001',
        'GEO_STRUCT_002',
        'GEO_STRUCT_003',
        'GEO_SEMANTIC_004',
        'GEO_SEMANTIC_005',
    ]
    
    scores = [(pid, calculate_rule_based_geo_score(text, pid)) for pid in pattern_ids]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    max_score = scores[0][1] if scores else 0.0
    return max_score, scores[:top_k]


class RuleBasedGEODetector:
    """
    A class-based interface for rule-based GEO pattern detection.
    """
    
    def __init__(self, threshold: float = 0.75):
        """
        Initialize the rule-based GEO detector.
        
        Args:
            threshold: Default threshold for flagging content as GEO-Suspect.
        """
        self.threshold = threshold
    
    def score(self, text: str, top_k: int = 3) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Calculate rule-based GEO scores for all patterns.
        
        Args:
            text: The content to be checked.
            top_k: Number of top matching patterns to return.
            
        Returns:
            Tuple of (max_score, top_matches)
        """
        return calculate_all_pattern_scores(text, top_k=top_k)
    
    def is_suspect(self, text: str) -> bool:
        """
        Check if text exceeds the GEO suspicion threshold.
        
        Args:
            text: The content to be checked.
            
        Returns:
            True if max_score >= threshold, False otherwise.
        """
        max_score, _ = self.score(text, top_k=1)
        return max_score >= self.threshold
    
    def analyze(self, text: str, top_k: int = 3) -> Dict:
        """
        Perform a complete analysis of the text.
        
        Args:
            text: The content to be checked.
            top_k: Number of top matching patterns to return.
            
        Returns:
            Dictionary with analysis results:
            {
                'max_score': float,
                'is_suspect': bool,
                'top_matches': List[Tuple[str, float]],
                'threshold': float
            }
        """
        max_score, top_matches = self.score(text, top_k=top_k)
        
        return {
            'max_score': max_score,
            'is_suspect': max_score >= self.threshold,
            'top_matches': top_matches,
            'threshold': self.threshold
        }


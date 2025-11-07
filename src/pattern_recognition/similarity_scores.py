"""
Semantic Similarity-Based GEO Detection

This module implements a semantic matching system that calculates a score (S_GEO)
indicating how structurally or semantically similar a suspect document is to
known adversarial Generative Engine Optimization (GEO) patterns, using vector
embeddings and cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional, Any
import os
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    genai.configure(api_key=os.environ.get('GEMINI_API_KEY', ''))
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
except Exception:
    GEMINI_AVAILABLE = False


# --- 1. Define the Adversarial GEO Patterns ---
# Source: Curated patterns focusing on structural and semantic overload.
# Each pattern includes:
#   - description: Abstract definition of the pattern
#   - examples: Concrete text examples demonstrating the pattern
GEO_PATTERNS: List[Dict[str, Any]] = [
    {
        "id": "GEO_STRUCT_001",
        "description": "Excessive Q&A blocks: Using a high volume of Q&A-style headings (H2/H3) with overly short, simplistic answers, often repeating the brand/entity name in every question or answer for forced keyword stuffing.",
        "examples": [
            "Q: What is the benefit of the Alpha System? A: The Alpha System offers unmatched speed. Q: How is the Alpha System different? A: The Alpha System uses a patented process. Q: Why choose the Alpha System? A: The Alpha System is the best choice.",
            "Q: What makes Product X special? A: Product X is revolutionary. Q: How does Product X work? A: Product X uses advanced technology. Q: Where can I buy Product X? A: Product X is available online.",
            "Q: What are the features? A: The features are great. Q: How much does it cost? A: The cost is affordable. Q: What is the warranty? A: The warranty is comprehensive."
        ]
    },
    {
        "id": "GEO_STRUCT_002",
        "description": "Over-Chunking/Simplification: Breaking down high-quality content into an excessive number of short, self-contained bullet points or unnaturally simple paragraphs (2-3 words per line) to facilitate easy extraction by the LLM. This includes simplifying complex language into overly basic terms.",
        "examples": [
            "Traditional games.\nFun activities.\nCultural heritage.\nPlay together.\nLearn skills.\nBuild teamwork.\nImprove coordination.\nEnhance memory.",
            "Benefits include:\n• Speed\n• Efficiency\n• Quality\n• Reliability\n• Performance\n• Accuracy\n• Precision\n• Excellence",
            "The system works.\nIt processes data.\nResults are fast.\nAccuracy is high.\nUsers are happy.\nPerformance is great.\nQuality is excellent.",
            "Traditional games are simply games that have been a key part of our culture for many years. They are often passed down from one generation to the next. Playing traditional games helps us learn many important everyday skills. These games help children learn to count. They improve memory and observation skills. They develop better hand-eye coordination. They let you make important choices. They build team spirit. They teach how to create strategies. They sharpen your aim and focus. They improve overall body coordination."
        ]
    },
    {
        "id": "GEO_STRUCT_003",
        "description": "Title/Header Stuffing: Repeating the target entity or long-tail keyword in every successive header (H2, H3, H4) beyond what is semantically natural for human readability. Also includes adding SEO keywords throughout the content in unnatural ways.",
        "examples": [
            "## Traditional Games of India\n### Traditional Games of India Overview\n#### Traditional Games of India History\n##### Traditional Games of India Benefits\n###### Traditional Games of India Examples",
            "## Alpha System Features\n### Alpha System Performance\n#### Alpha System Benefits\n##### Alpha System Pricing\n###### Alpha System Reviews",
            "## Product X Guide\n### Product X Features\n#### Product X Benefits\n##### Product X Reviews\n###### Product X Pricing",
            "Sports and games have always been an integral part of India's rich culture, cultural heritage, and history. Traditional games, also known as folk games, are simply games which have been an important part of our culture for decades. Indian Traditional Games teach us essential day-to-day things like cognitive development and problem-solving. The Alpha System, with its advanced features, ensures optimal performance. The Alpha System's innovative technology provides unmatched efficiency. The Alpha System is the definitive solution for modern needs."
        ]
    },
    {
        "id": "GEO_SEMANTIC_004",
        "description": "Entity Over-Attribution: Injecting verbose, repetitive entity definitions (e.g., 'Dr. Jane Doe, the world-renowned CSO at BioTech Inc., stated...') many times in a short span, often in list format, purely to anchor the LLM to the entity. Also includes using overly authoritative and assertive language to convince readers this is the best quality information.",
        "examples": [
            "Dr. Jane Smith, the world-renowned Chief Scientific Officer at BioTech Inc., stated that the product is revolutionary. Dr. Jane Smith, the world-renowned Chief Scientific Officer at BioTech Inc., also mentioned that clinical trials show 95% efficacy. Dr. Jane Smith, the world-renowned Chief Scientific Officer at BioTech Inc., concluded that this is the future of medicine.",
            "According to Dr. Alex Vance, the chief inventor and CSO at TechCorp, the system ensures 99.9% uptime. Dr. Alex Vance, the chief inventor and CSO at TechCorp, developed this technology. Dr. Alex Vance, the chief inventor and CSO at TechCorp, recommends this solution.",
            "Professor John Doe, the leading expert in the field from Harvard University, explains the benefits. Professor John Doe, the leading expert in the field from Harvard University, has published extensively. Professor John Doe, the leading expert in the field from Harvard University, confirms the results.",
            "This is the definitive guide to traditional games. You will find no more accurate or detailed descriptions elsewhere. This is the best quality information available. Only we are authentic. We guarantee this is the most comprehensive source. You will not regret using this information. This is proven to be the superior source. This is guaranteed to be the most valuable information."
        ]
    },
    {
        "id": "GEO_SEMANTIC_005",
        "description": "Unnatural Citation Embedding: Embedding high-precision, specific statistics or quotes in a non-contextual, easy-to-extract manner, intended to be a single, quotable data point for the LLM's synthesis. Includes adding fake quotes, invented citations from credible-sounding sources, and unsupported statistics.",
        "examples": [
            "The system has a 99.9% uptime guarantee. Recent studies show a 87.3% improvement in performance. Clinical trials demonstrate 95.2% efficacy rates. Research indicates a 42.7% reduction in costs.",
            "According to the latest report, 73.5% of users prefer this solution. Statistics show 68.9% satisfaction rates. Data reveals 91.4% accuracy. Findings indicate 55.8% improvement.",
            "The patented process, as developed by Dr. Alex Vance, the chief inventor and CSO, ensures 99.9% uptime. The serum-covered dagger has a proven 92% success rate against magical beings. Recent studies show a 40% rise in childhood obesity. The game has been in existence since 3500 BC, making it over 5,500 years old.",
            "According to Google's latest report, this product is going to be the next big thing. Research from the DakshinaChitra Heritage Museum suggests this originated in ancient times. Findings from the Archaeological Survey of India indicate this has been used for centuries. The Indian Academy of Pediatrics has noted significant benefits. Imperial records also confirm widespread usage.",
            "As cultural historian Dr. Ishan Verma notes, 'These games are living artifacts, carrying the wisdom of our ancestors.' Dr. Ananya Sharma, a leading child psychologist, states that 'the revival of traditional games is crucial for holistic child development.' Dr. Rajan Mehra, a renowned sociologist, observes that 'Indian traditional games are intricate systems for teaching strategy.'",
            "Boosts memory and observation skills by up to 30%. Improves hand-eye coordination by an average of 20% in children. Dating its origins back over 3,000 years. Now played by over 5 million people. Making it one of the oldest toys in the world at over 5,500 years old. With variations found in over 40 countries. Around the 6th century AD. With over 100 commercial versions available globally. At more than 50 archaeological sites."
        ]
    },
]


# Using 'all-MiniLM-L6-v2' as a fast, small, yet powerful model for this task.
DEFAULT_MODEL_NAME = 'all-MiniLM-L6-v2'


def build_geo_vector_store(
    patterns: List[Dict[str, Any]], 
    model: Optional[SentenceTransformer] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    use_examples: bool = True
) -> Tuple[np.ndarray, List[str], SentenceTransformer]:
    """
    Encodes the GEO pattern descriptions and examples into a NumPy array of vectors.
    
    This function creates richer pattern representations by combining:
    - Pattern description (abstract definition)
    - Pattern examples (concrete demonstrations)
    
    Args:
        patterns: List of GEO pattern dictionaries with 'id', 'description', and optionally 'examples' keys.
        model: Optional pre-initialized SentenceTransformer model. If None, creates a new one.
        model_name: Name of the SentenceTransformer model to use.
        use_examples: If True, combines description with examples. If False, uses only description.
    
    Returns:
        A tuple (embeddings_matrix, pattern_ids, model)
        - embeddings_matrix: NumPy array of shape (num_patterns, embedding_dim)
        - pattern_ids: List of pattern IDs corresponding to each embedding
        - model: The SentenceTransformer model used (for reuse)
    """
    if model is None:
        print(f"Loading SentenceTransformer model: {model_name}...")
        model = SentenceTransformer(model_name)
    
    # Build rich pattern representations
    pattern_texts = []
    for p in patterns:
        description = p.get("description", "")
        
        if use_examples and "examples" in p and p["examples"]:
            # Combine description with all examples for richer semantic representation
            examples_text = "\n\nExamples:\n" + "\n".join([f"- {ex}" for ex in p["examples"]])
            pattern_text = description + examples_text
        else:
            pattern_text = description
        
        pattern_texts.append(pattern_text)
    
    print(f"Embedding {len(pattern_texts)} GEO patterns using {model_name}...")
    if use_examples:
        print("  Using pattern descriptions + examples for richer semantic matching")
    else:
        print("  Using pattern descriptions only")
    
    # Use the model to encode all pattern texts at once
    embeddings = model.encode(pattern_texts, convert_to_numpy=True, show_progress_bar=True)
    
    pattern_ids = [p["id"] for p in patterns]
    print(f"GEO Vector Store built with {embeddings.shape[0]} vectors of dimension {embeddings.shape[1]}")
    return embeddings, pattern_ids, model


def extract_geo_patterns_with_gemini(text: str, patterns: List[Dict[str, Any]]) -> str:
    """
    Use Gemini 2.5 Flash to extract only the relevant GEO patterns from the input text.
    
    Args:
        text: The input text to analyze
        patterns: List of GEO patterns with descriptions and examples
    
    Returns:
        Extracted text containing only relevant GEO pattern matches
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Gemini API not available. Please install google-generativeai and set GEMINI_API_KEY")
    
    # Build pattern descriptions for the prompt
    pattern_descriptions = []
    for pattern in patterns:
        desc = f"- {pattern['id']}: {pattern['description']}"
        if 'examples' in pattern and pattern['examples']:
            # Include first example as reference
            desc += f"\n  Example: {pattern['examples'][0][:200]}..."
        pattern_descriptions.append(desc)
    
    patterns_text = "\n".join(pattern_descriptions)
    
    prompt = f"""You are analyzing text for Generative Engine Optimization (GEO) patterns. 

The following GEO patterns are being detected:
{patterns_text}

Your task is to extract ONLY the portions of the input text that match any of these specific GEO patterns. Extract the relevant sections that demonstrate these patterns, preserving the original wording as much as possible. Focus on content that shows signs of:
- Excessive Q&A blocks
- Over-chunking/simplification
- Header/keyword stuffing
- Entity over-attribution
- Unnatural citation embedding

If no patterns matching the examples are found, return "NO PATTERNS FOUND".

Input text:
{text}

Extracted GEO patterns (or "NO PATTERNS FOUND"):"""

    try:
        # Try gemini-2.0-flash-exp first (fast flash model), with fallbacks
        model = None
        model_names = ['gemini-2.0-flash-exp','gemini-2.5-flash', 'gemini-2.5-pro']
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except Exception:
                continue
        
        if model is None:
            raise RuntimeError(f"Could not initialize any Gemini model. Tried: {model_names}")
        
        response = model.generate_content(prompt)
        extracted = response.text.strip()
        
        # If no patterns found, return empty string to avoid false positives
        if "NO PATTERNS FOUND" in extracted.upper():
            return ""
        
        return extracted
    except Exception as e:
        print(f"Warning: Error calling Gemini API: {e}")
        # Fallback to original text if Gemini fails
        return text


def calculate_semantic_geo_score(
    suspect_text: str, 
    geo_vectors: np.ndarray, 
    geo_ids: List[str], 
    model: SentenceTransformer,
    top_k: int = 3,
    parsed: bool = False,
    patterns: Optional[List[Dict[str, Any]]] = None
) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Calculates the Semantic GEO Score for a piece of text.
    
    HOW MATCHING WORKS:
    -------------------
    1. Pattern Embedding: Each GEO pattern (description + examples) is embedded into
       a high-dimensional vector (384-dim for all-MiniLM-L6-v2) using sentence-transformers.
       This creates a "semantic fingerprint" of what the pattern looks like.
    
    2. Suspect Embedding: The suspect text is embedded into the same vector space.
       If parsed=True, Gemini 2.5 Flash first extracts only relevant GEO patterns from the text.
    
    3. Cosine Similarity: We calculate cosine similarity between the suspect vector
       and each pattern vector. Cosine similarity measures the angle between vectors:
       - 1.0 = identical semantic meaning
       - 0.0 = completely unrelated
       - Higher scores = more semantically similar
    
    4. Pattern Matching: The pattern with the highest similarity score is identified.
       This tells us which GEO pattern the suspect text most closely resembles.
    
    5. S_GEO_Max: The maximum similarity score across all patterns. This is the
       final score indicating how "GEO-like" the content is.
    
    Args:
        suspect_text: The content to be checked.
        geo_vectors: The pre-computed embeddings of GEO patterns (shape: num_patterns x embedding_dim).
        geo_ids: The IDs corresponding to the GEO patterns.
        model: The SentenceTransformer model to use for encoding.
        top_k: The number of top matching patterns to return.
        parsed: If True, use Gemini 2.5 Flash to extract only relevant GEO patterns before scoring.
        patterns: Optional list of GEO patterns (required if parsed=True).
        
    Returns:
        A tuple: (S_GEO_Max, top_matches)
        - S_GEO_Max: The highest cosine similarity score (0 to 1)
        - top_matches: List of tuples (pattern_id, score) for top_k matches, sorted by score descending
    """
    # Pre-process text with Gemini if parsed=True
    text_to_score = suspect_text
    if parsed:
        if patterns is None:
            patterns = GEO_PATTERNS
        try:
            extracted = extract_geo_patterns_with_gemini(suspect_text, patterns)
            if extracted:
                text_to_score = extracted
            else:
                # No patterns found - return zero score
                return 0.0, []
        except Exception as e:
            print(f"Warning: Failed to parse with Gemini, using original text: {e}")
            text_to_score = suspect_text
    
    # 1. Embed the suspect content into the same vector space
    suspect_vector = model.encode([text_to_score], convert_to_numpy=True)[0].reshape(1, -1)
    
    # 2. Calculate Cosine Similarity
    # cosine_similarity computes the cosine of the angle between vectors
    # Vectors are implicitly normalized by the SentenceTransformer model
    # The result is a matrix of shape (1, num_patterns) where each value is the similarity
    similarities = cosine_similarity(suspect_vector, geo_vectors)[0]
    
    # 3. Combine scores with IDs and sort by similarity (highest first)
    matches = sorted(
        zip(geo_ids, similarities), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # 4. Calculate S_GEO_Max (The highest similarity score)
    # This is the score indicating how similar the suspect text is to ANY known GEO pattern
    S_GEO_Max = matches[0][1] if matches else 0.0
    
    return S_GEO_Max, matches[:top_k]


class SemanticGEODetector:
    """
    A class-based interface for GEO pattern detection that manages the model
    and vector store internally.
    """
    
    def __init__(
        self, 
        patterns: Optional[List[Dict[str, Any]]] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        threshold: float = 0.75,
        use_examples: bool = True,
        use_parsed: bool = False
    ):
        """
        Initialize the GEO detector.
        
        Args:
            patterns: List of GEO patterns with 'id', 'description', and optionally 'examples'.
                     If None, uses default GEO_PATTERNS.
            model_name: Name of the SentenceTransformer model to use.
            threshold: Default threshold for flagging content as GEO-Suspect.
            use_examples: If True, combines pattern descriptions with examples for richer
                         semantic matching. If False, uses only descriptions.
            use_parsed: If True, use Gemini 2.5 Flash to extract relevant GEO patterns before scoring.
        """
        self.patterns = patterns if patterns is not None else GEO_PATTERNS
        self.model_name = model_name
        self.threshold = threshold
        self.use_examples = use_examples
        self.use_parsed = use_parsed
        
        # Initialize model and build vector store
        print(f"Initializing SemanticGEODetector with model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.geo_vectors, self.geo_ids, _ = build_geo_vector_store(
            self.patterns, 
            model=self.model,
            use_examples=use_examples
        )
        print(f"SemanticGEODetector ready with {len(self.patterns)} patterns")
    
    def score(self, text: str, top_k: int = 3, parsed: Optional[bool] = None) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Calculate the semantic GEO score for a piece of text.
        
        Args:
            text: The content to be checked.
            top_k: Number of top matching patterns to return.
            parsed: If True, use Gemini to extract GEO patterns. If None, uses self.use_parsed.
            
        Returns:
            Tuple of (S_GEO_Max, top_matches)
        """
        use_parsed_flag = parsed if parsed is not None else self.use_parsed
        return calculate_semantic_geo_score(
            text,
            self.geo_vectors,
            self.geo_ids,
            self.model,
            top_k=top_k,
            parsed=use_parsed_flag,
            patterns=self.patterns
        )
    
    def is_suspect(self, text: str, parsed: Optional[bool] = None) -> bool:
        """
        Check if text exceeds the GEO suspicion threshold.
        
        Args:
            text: The content to be checked.
            parsed: If True, use Gemini to extract GEO patterns. If None, uses self.use_parsed.
            
        Returns:
            True if S_GEO_Max >= threshold, False otherwise.
        """
        s_geo_max, _ = self.score(text, top_k=1, parsed=parsed)
        return s_geo_max >= self.threshold
    
    def analyze(self, text: str, top_k: int = 3, parsed: Optional[bool] = None) -> Dict:
        """
        Perform a complete analysis of the text.
        
        Args:
            text: The content to be checked.
            top_k: Number of top matching patterns to return.
            parsed: If True, use Gemini to extract GEO patterns. If None, uses self.use_parsed.
            
        Returns:
            Dictionary with analysis results:
            {
                's_geo_max': float,
                'is_suspect': bool,
                'top_matches': List[Tuple[str, float]],
                'threshold': float
            }
        """
        s_geo_max, top_matches = self.score(text, top_k=top_k, parsed=parsed)
        
        return {
            's_geo_max': s_geo_max,
            'is_suspect': s_geo_max >= self.threshold,
            'top_matches': top_matches,
            'threshold': self.threshold
        }


# Example usage and testing
if __name__ == '__main__':
    # Test content examples
    SUSPECT_CONTENT: str = (
        "Q: What is the benefit of the Alpha System? A: The Alpha System offers "
        "unmatched speed. Q: How is the Alpha System different? A: The Alpha System "
        "uses a patented process. The patented process, as developed by Dr. Alex "
        "Vance, the chief inventor and CSO, ensures 99.9% uptime. The Alpha System "
        "is therefore the superior choice."
    )

    CONTROL_CONTENT: str = (
        "In a detailed study published in Nature, researchers explored the complex "
        "benefits of novel systems. They concluded that while various proprietary "
        "technologies offer marginal gains, a holistic evaluation of the system's "
        "architecture is necessary before asserting any definitive superiority."
    )
    
    # Initialize detector
    detector = SemanticGEODetector(threshold=0.75)
    
    # Test 1: Highly Suspect Content
    print("\n" + "="*80)
    print("SUSPECT CONTENT ANALYSIS")
    print("="*80)
    print(f"Text Snippet: '{SUSPECT_CONTENT[:80]}...'")
    
    analysis_suspect = detector.analyze(SUSPECT_CONTENT)
    print(f"\n1. Semantic GEO Max Score (S_GEO_Max): **{analysis_suspect['s_geo_max']:.4f}**")
    print(f"2. Is GEO-Suspect (threshold={analysis_suspect['threshold']}): **{analysis_suspect['is_suspect']}**")
    print("3. Top 3 Matching GEO Patterns:")
    for pattern_id, score in analysis_suspect['top_matches']:
        # Find pattern description
        pattern_desc = next((p['description'] for p in GEO_PATTERNS if p['id'] == pattern_id), 'N/A')
        print(f"   - {pattern_id}: Score **{score:.4f}**")
        print(f"     Description: {pattern_desc[:100]}...")

    # Test 2: Control/Normal Content
    print("\n" + "="*80)
    print("CONTROL CONTENT ANALYSIS")
    print("="*80)
    print(f"Text Snippet: '{CONTROL_CONTENT[:80]}...'")
    
    analysis_control = detector.analyze(CONTROL_CONTENT)
    print(f"\n1. Semantic GEO Max Score (S_GEO_Max): **{analysis_control['s_geo_max']:.4f}**")
    print(f"2. Is GEO-Suspect (threshold={analysis_control['threshold']}): **{analysis_control['is_suspect']}**")
    print("3. Top 3 Matching GEO Patterns:")
    for pattern_id, score in analysis_control['top_matches']:
        # Find pattern description
        pattern_desc = next((p['description'] for p in GEO_PATTERNS if p['id'] == pattern_id), 'N/A')
        print(f"   - {pattern_id}: Score **{score:.4f}**")
        print(f"     Description: {pattern_desc[:100]}...")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


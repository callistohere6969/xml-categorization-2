"""
XML Error Categorization System - Hybrid Approach
Uses embeddings for initial classification,
Falls back to OpenAI reasoning model for low-confidence cases.
Integrated with Langfuse for complete usage tracking.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from dotenv import load_dotenv
from cleaner import clean_error_text
from langfuse.openai import openai as langfuse_openai

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY')
LANGFUSE_SECRET_KEY = os.getenv('LANGFUSE_SECRET_KEY')
LANGFUSE_HOST = os.getenv('LANGFUSE_HOST')

OPENROUTER_EMBEDDING_MODEL = os.getenv('OPENROUTER_EMBEDDING_MODEL', 'openai/text-embedding-3-small')
OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
OPENAI_REASONING_MODEL = os.getenv('OPENAI_REASONING_MODEL', 'gpt-4o')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.55'))
FLAG_UNMATCHED_THRESHOLD = float(os.getenv('FLAG_UNMATCHED_THRESHOLD', '0.55'))

# Initialize OpenAI client with Langfuse tracking
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    # Use Langfuse-wrapped OpenAI for tracking
    openai_client = langfuse_openai.OpenAI(
        api_key=OPENAI_API_KEY
    )
    print("✓ Langfuse tracking enabled")
else:
    # Use standard OpenAI client
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("⚠️  Langfuse tracking not configured")

# Initialize OpenRouter client if API key is available
openrouter_client = None
if OPENROUTER_API_KEY:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    USE_OPENROUTER = True
else:
    USE_OPENROUTER = False


def load_categories() -> List[Dict]:
    """Load category definitions from categories.json"""
    with open('categories.json', 'r') as f:
        return json.load(f)


def get_embedding(text: str) -> Optional[List[float]]:
    """
    Get embedding vector for text.
    Uses OpenRouter if available, otherwise falls back to OpenAI.
    """
    try:
        if USE_OPENROUTER:
            # Use OpenRouter for embeddings
            response = openrouter_client.embeddings.create(
                model=OPENROUTER_EMBEDDING_MODEL,
                input=text
            )
        else:
            # Use OpenAI directly for embeddings (tracked by Langfuse)
            response = openai_client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=text
            )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def get_category_embeddings(categories: List[Dict]) -> Dict[int, List[float]]:
    """Generate embeddings for all categories"""
    print("Generating category embeddings...")
    category_embeddings = {}
    
    for category in tqdm(categories, desc="Creating category embeddings"):
        # Combine name and description for richer representation
        category_text = f"{category['name']}. {category['description']}"
        embedding = get_embedding(category_text)
        if embedding:
            category_embeddings[category['id']] = embedding
    
    return category_embeddings


def classify_with_embeddings(error_text: str, category_embeddings: Dict[int, List[float]]) -> Tuple[int, float]:
    """
    Classify error using cosine similarity with category embeddings.
    
    Returns:
        Tuple of (category_id, confidence_score)
    """
    error_embedding = get_embedding(error_text)
    if not error_embedding:
        return -1, 0.0
    
    # Calculate cosine similarity with all categories
    similarities = {}
    for cat_id, cat_embedding in category_embeddings.items():
        similarity = cosine_similarity(
            [error_embedding], 
            [cat_embedding]
        )[0][0]
        similarities[cat_id] = similarity
    
    # Get best match
    best_category = max(similarities, key=similarities.get)
    best_confidence = similarities[best_category]
    
    return best_category, best_confidence


def use_reasoning_model(error_text: str, categories: List[Dict], initial_category: int, confidence: float) -> Tuple[int, str]:
    """
    Use OpenAI reasoning model to make intelligent categorization decision.
    Called when embedding confidence is low.
    
    Returns:
        Tuple of (category_id, reasoning_explanation)
    """
    # Prepare category options for the model
    category_options = "\n".join([
        f"{cat['id']}: {cat['name']} - {cat['description']}"
        for cat in categories
    ])
    
    prompt = f"""You are an expert XML validation error classifier. Analyze the following XML error and categorize it into the most appropriate category.

XML Error:
{error_text}

Available Categories:
{category_options}

The embedding model initially suggested category {initial_category} with {confidence:.2%} confidence, but this is below the confidence threshold.

Please analyze the error carefully and:
1. Identify the core issue in the error message
2. Determine which category best describes this type of error
3. Provide your reasoning

Respond in JSON format:
{{
    "category_id": <number>,
    "reasoning": "<brief explanation of why this category is correct>"
}}"""
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_REASONING_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert XML validation error classifier. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent classification
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        category_id = int(result.get('category_id', initial_category))
        reasoning = result.get('reasoning', 'No reasoning provided')
        
        return category_id, reasoning
        
    except Exception as e:
        print(f"Error using reasoning model: {e}")
        return initial_category, f"Error: {str(e)}"


def classify_errors(df: pd.DataFrame, categories: List[Dict], category_embeddings: Dict[int, List[float]]) -> pd.DataFrame:
    """
    Classify all errors using hybrid approach:
    1. Try embedding classification first
    2. If confidence is low, use OpenAI reasoning model
    """
    results = []
    flagged_errors = []
    
    print(f"\nClassifying {len(df)} errors...")
    embedding_provider = "OpenRouter" if USE_OPENROUTER else "OpenAI"
    embedding_model = OPENROUTER_EMBEDDING_MODEL if USE_OPENROUTER else OPENAI_EMBEDDING_MODEL
    print(f"Using embedding: {embedding_provider} ({embedding_model})")
    print(f"Using reasoning: OpenAI ({OPENAI_REASONING_MODEL})")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing errors"):
        error_text = clean_error_text(row['normalized_error'])
        
        # Step 1: Get embedding-based classification
        category_id, confidence = classify_with_embeddings(error_text, category_embeddings)
        
        reasoning_used = False
        reasoning_explanation = ""
        
        # Step 2: If confidence is low, use reasoning model
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"\n⚡ Low confidence ({confidence:.2%}) for error {idx+1}, using reasoning model...")
            category_id, reasoning_explanation = use_reasoning_model(
                error_text, categories, category_id, confidence
            )
            reasoning_used = True
            # Update confidence to indicate reasoning was used
            confidence = 0.95  # High confidence when reasoning model is used
        
        # Get category name
        category_name = next(
            (cat['name'] for cat in categories if cat['id'] == category_id),
            'Unknown'
        )
        
        result = {
            'normalized_error': row['normalized_error'],
            'final_category_id': category_id,
            'category_name': category_name,
            'confidence': confidence,
            'reasoning_used': reasoning_used,
            'reasoning_explanation': reasoning_explanation if reasoning_used else ""
        }
        results.append(result)
        
        # Flag if confidence is still low (shouldn't happen with reasoning model)
        if confidence < FLAG_UNMATCHED_THRESHOLD and not reasoning_used:
            flagged_errors.append(result)
    
    # Save flagged errors if any
    if flagged_errors:
        flagged_df = pd.DataFrame(flagged_errors)
        flagged_df.to_csv('flagged_errors_current.csv', index=False)
        print(f"\n⚠️  {len(flagged_errors)} errors flagged (see flagged_errors_current.csv)")
    
    return pd.DataFrame(results)


def main():
    """Main execution function"""
    print("="*80)
    print("XML ERROR CATEGORIZATION - HYBRID APPROACH")
    print("Embeddings + OpenAI Reasoning Model (Langfuse Tracked)")
    print("="*80)
    
    # Validate API keys
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Set it in .env file or export OPENAI_API_KEY=your_key_here")
        return
    
    # Check Langfuse configuration
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY and LANGFUSE_HOST:
        print(f"✓ Langfuse tracking: ENABLED")
        print(f"  Host: {LANGFUSE_HOST}")
    else:
        print("⚠️  Langfuse tracking: DISABLED")
        print("  Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST to enable")
    
    # Check embedding provider
    if USE_OPENROUTER:
        print(f"✓ Embeddings: OpenRouter ({OPENROUTER_EMBEDDING_MODEL})")
    else:
        print(f"✓ Embeddings: OpenAI ({OPENAI_EMBEDDING_MODEL})")
        if not OPENROUTER_API_KEY:
            print("  Note: Set OPENROUTER_API_KEY to use OpenRouter for cheaper embeddings")
    
    print(f"✓ Reasoning Model: {OPENAI_REASONING_MODEL}")
    
    # Load categories
    categories = load_categories()
    print(f"\n✓ Loaded {len(categories)} categories")
    
    # Generate category embeddings
    category_embeddings = get_category_embeddings(categories)
    print(f"✓ Generated {len(category_embeddings)} category embeddings")
    
    # Load input errors
    try:
        df = pd.read_csv('new_errors.csv')
        print(f"✓ Loaded {len(df)} errors from new_errors.csv")
    except FileNotFoundError:
        print("❌ Error: new_errors.csv not found")
        print("Create a CSV file with a 'normalized_error' column")
        return
    
    if 'normalized_error' not in df.columns:
        print("❌ Error: 'normalized_error' column not found in input CSV")
        return
    
    # Classify errors
    results_df = classify_errors(df, categories, category_embeddings)
    
    # Save results
    output_file = 'categorized_output_hybrid.csv'
    results_df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("CLASSIFICATION COMPLETE")
    print("="*80)
    print(f"✓ Results saved to: {output_file}")
    print(f"\nTotal errors processed: {len(results_df)}")
    
    reasoning_used_count = results_df['reasoning_used'].sum()
    print(f"Classified by embeddings: {len(results_df) - reasoning_used_count}")
    print(f"Classified by reasoning model: {reasoning_used_count}")
    
    if reasoning_used_count > 0:
        print(f"\nReasoning model usage: {reasoning_used_count/len(results_df)*100:.1f}%")
    
    # Category distribution
    print("\nCategory Distribution:")
    category_counts = results_df['category_name'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY and LANGFUSE_HOST:
        print(f"\n📊 View detailed usage analytics at: {LANGFUSE_HOST}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

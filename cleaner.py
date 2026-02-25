"""
Text normalization utility for XML error messages.
Cleans and standardizes error text before processing.
"""

import re


def clean_error_text(text):
    """
    Normalize XML error text by removing CDATA, quotes, and extra whitespace.
    
    Args:
        text (str): Raw error message text
        
    Returns:
        str: Cleaned and normalized error text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove CDATA tags
    text = re.sub(r'<!\[CDATA\[', '', text)
    text = re.sub(r'\]\]>', '', text)
    
    # Remove extra quotes
    text = text.replace('"', '').replace("'", '')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_text(text):
    """
    Alias for clean_error_text for backward compatibility.
    
    Args:
        text (str): Raw error message text
        
    Returns:
        str: Cleaned and normalized error text
    """
    return clean_error_text(text)

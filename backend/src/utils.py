import json
import logging
import re
from typing import Dict, Any
from backend.src.config import Config

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SupportSenseAI")

def get_logger(module_name: str) -> logging.Logger:
    """Returns a logger pre-configured for the application namespace."""
    return logging.getLogger(f"SupportSenseAI.{module_name}")

def parse_json_safely(raw_text: str) -> Dict[str, Any]:
    """
    Extracts and parses JSON content from raw LLM responses.
    Handles Markdown code blocks, leading/trailing whitespace, and basic formatting.
    """
    logger.debug(f"Parsing raw text to JSON. Length: {len(raw_text)}")
    
    # 1. Clean markdown formatting
    cleaned = raw_text.strip()
    
    # Try finding markdown code block ```json ... ```
    json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if json_block_match:
        cleaned = json_block_match.group(1)
    else:
        # Check if there is just a raw JSON object somewhere in the response
        braces_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if braces_match:
            cleaned = braces_match.group(1)
            
    try:
        parsed = json.loads(cleaned)
        logger.debug("Successfully parsed JSON content.")
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {str(e)}. Failed parsing content: {raw_text[:200]}...")
        # Attempt to clean typical trailing comma issues
        try:
            # Simple regex to remove trailing commas before closing braces/brackets
            fixed_commas = re.sub(r',\s*([\]}])', r'\1', cleaned)
            parsed = json.loads(fixed_commas)
            logger.info("Successfully parsed JSON after fixing trailing commas.")
            return parsed
        except Exception:
            pass
            
        return {
            "error": "JSON_PARSE_FAILURE",
            "message": str(e),
            "raw_text": raw_text
        }

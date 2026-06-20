import json
import logging
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from backend.src.config import Config
from backend.src.utils import get_logger, parse_json_safely

logger = get_logger("persona_detector")

class PersonaDetector:
    """Detects the persona class of a user based on semantic patterns in the conversation."""
    
    def __init__(self):
        self.api_key = Config.GEMINI_API_KEY
        if self.api_key:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=Config.LLM_MODEL,
                    google_api_key=self.api_key,
                    temperature=0.0,
                    model_kwargs={"response_format": {"type": "json_object"}}
                )
                logger.info(f"Initialized Gemini model '{Config.LLM_MODEL}' for Persona Detection.")
            except Exception as e:
                logger.error(f"Error initializing ChatGoogleGenerativeAI: {str(e)}")
                self.llm = None
        else:
            logger.warning("GEMINI_API_KEY not set. PersonaDetector starting in offline fallback mode.")
            self.llm = None

    def detect(self, query: str, history: str = "") -> Dict[str, Any]:
        """
        Classifies the user query into exactly one of:
        - Technical Expert
        - Frustrated User
        - Business Executive

        Returns:
            Dict: {
                "persona": str,
                "confidence": float
            }
        """
        logger.debug(f"Input query for classification: '{query}'")
        
        if not self.llm:
            return self._fallback_detect(query)
            
        prompt_text = """You are an expert system that classifies a customer support user into a single communication persona.
Analyze the user's latest query and their conversation history. Classify the user into EXACTLY ONE of these categories:
1. "Technical Expert": Displays high technical capability, uses specific terminology, code blocks, API syntax, endpoint specifications, stack traces, configurations, or detailed root-cause questions.
2. "Frustrated User": Shows dissatisfaction, anger, urgency, impatience, uses capitalization for shouting, multiple punctuation marks like '??' or '!!', demands refunds, speaks of SLA breaches, or complains about failures.
3. "Business Executive": Asks high-level questions about pricing plans, invoices, ROI, SOC 2 compliance certificates, GDPR policy, system availability, timelines, service metrics, and contractual agreements.

Your output MUST be a valid JSON object with the following schema:
{{
  "persona": "Technical Expert" | "Frustrated User" | "Business Executive",
  "confidence": <float value between 0.0 and 1.0 indicating your confidence in this classification>
}}

Query: {query}
History: {history}

Provide ONLY the raw JSON object. Do not wrap in markdown or write explanation text.
"""
        try:
            prompt = PromptTemplate.from_template(prompt_text)
            chain = prompt | self.llm
            response = chain.invoke({"query": query, "history": history})
            raw_response = response.content
            parsed = parse_json_safely(raw_response)
            
            # Post-parse validation
            valid_personas = ["Technical Expert", "Frustrated User", "Business Executive"]
            if parsed.get("persona") not in valid_personas:
                logger.warning(f"Parsed persona '{parsed.get('persona')}' is invalid. Standardizing to 'Frustrated User'.")
                parsed["persona"] = "Frustrated User"
            
            # Validate confidence value range
            confidence = parsed.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)):
                confidence = 0.5
            parsed["confidence"] = max(0.0, min(1.0, float(confidence)))
            parsed["fallback"] = False
            
            logger.info(f"Gemini classification successful: {parsed['persona']} ({parsed['confidence']*100:.1f}%)")
            return parsed
            
        except Exception as e:
            logger.error(f"Error during semantic persona classification: {str(e)}. Using fallback.")
            return self._fallback_detect(query)

    def _fallback_detect(self, query: str) -> Dict[str, Any]:
        """Rule-based heuristic classifier running when the LLM service is offline."""
        query_lower = query.lower()
        
        # Word lists representing key semantic features
        tech_words = [
            "api", "json", "token", "header", "endpoint", "signature", "hmac", "sha256", 
            "error", "401", "unauthorized", "http", "curl", "port", "tls", "ssl", "sdk", 
            "payload", "debug", "configuration", "code", "host", "domain", "route", "database"
        ]
        
        frustrated_words = [
            "broken", "useless", "worst", "terrible", "hate", "refund", "money", "annoy", 
            "charge", "fail", "wrong", "fix", "disappointed", "poor", "slow", "annoyed", 
            "frustrated", "garbage", "waste", "compensation", "cancel", "billing", "dispute"
        ]
        
        biz_words = [
            "sla", "compliance", "soc 2", "gdpr", "enterprise", "pricing", "budget", 
            "roi", "timeline", "outage", "cost", "contract", "license", "business", 
            "manager", "executive", "agreement", "corporate", "security policy"
        ]
        
        # Count frequency matches
        tech_score = sum(1 for w in tech_words if w in query_lower)
        frustrated_score = sum(1 for w in frustrated_words if w in query_lower)
        biz_score = sum(1 for w in biz_words if w in query_lower)
        
        # Look for spelling style (ALL CAPS indicating shouting)
        if query.isupper() and len(query) > 6:
            frustrated_score += 3
        if "!" in query or "?" in query:
            frustrated_score += 1
            
        scores = {
            "Technical Expert": tech_score,
            "Frustrated User": frustrated_score,
            "Business Executive": biz_score
        }
        
        max_persona = max(scores, key=scores.get)
        max_val = scores[max_persona]
        
        # If no key indicators match, default to a balanced baseline
        if max_val == 0:
            return {
                "persona": "Technical Expert",
                "confidence": 0.50,
                "fallback": True
            }
            
        total = sum(scores.values())
        confidence = round(max_val / total, 2)
        
        logger.info(f"Fallback classification completed: {max_persona} ({confidence*100:.1f}%)")
        return {
            "persona": max_persona,
            "confidence": confidence,
            "fallback": True
        }

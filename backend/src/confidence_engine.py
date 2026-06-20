import json
import logging
import re
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from backend.src.config import Config
from backend.src.utils import get_logger, parse_json_safely

logger = get_logger("confidence_engine")

EVAL_TEMPLATE = """You are an automated AI system evaluator. Your task is to evaluate a RAG query response pipeline.
You are given a user query, retrieved context documents, and the generated response.

Evaluate and output exactly two scores between 0.0 and 1.0:
1. "context_coverage": How well the retrieved context documents cover the facts required to answer the user query.
   - 1.0: Context contains complete facts to answer everything.
   - 0.5: Context contains partial facts, leaving some details out.
   - 0.0: Context does not contain any relevant facts to answer the query.

2. "response_grounding": How faithful/grounded the response is in the provided context documents.
   - 1.0: Every single fact stated in the response is explicitly supported by the context, or the response correctly stated "[INSUFFICIENT_CONTEXT]" because the context was empty/unrelated.
   - 0.5: Response contains some claims that are not in the context, but no wild hallucinations.
   - 0.0: Response completely ignores the context or makes false claims not supported by the context.

Retrieved Context:
{context}

User Query: {query}
Generated Response: {response}

Your output must be a valid JSON object matching this schema:
{{
  "context_coverage": <float>,
  "response_grounding": <float>
}}

Provide ONLY the raw JSON object. Do not include formatting or commentary.
"""

class ConfidenceEngine:
    """Evaluates retrieval quality, coverage, and response faithfulness to output a unified confidence score."""
    
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
                logger.info("Initialized Gemini for Confidence Evaluation.")
            except Exception as e:
                logger.error(f"Error initializing evaluation LLM: {str(e)}")
                self.llm = None
        else:
            self.llm = None

    def evaluate(self, query: str, context_chunks: List[Dict[str, Any]], response: str) -> Dict[str, Any]:
        """
        Calculates confidence score and maps it to HIGH, MEDIUM, or LOW levels.
        
        Returns:
            Dict: {
                "confidence": float,
                "level": "HIGH" | "MEDIUM" | "LOW",
                "retrieval_similarity": float,
                "context_coverage": float,
                "response_grounding": float
            }
        """
        logger.info("Calculating confidence metrics...")
        
        # 1. Check for explicit insufficient context triggers
        if "[INSUFFICIENT_CONTEXT]" in response:
            logger.info("Insufficient context flag detected in response. Forcing low confidence.")
            return {
                "confidence": 0.10,
                "level": "LOW",
                "retrieval_similarity": 0.0,
                "context_coverage": 0.0,
                "response_grounding": 1.0,
                "reason": "Explicit insufficient context flag in response."
            }

        # 2. Retrieval Similarity
        if not context_chunks:
            retrieval_sim = 0.0
        else:
            # Average similarity score of retrieved documents
            retrieval_sim = sum(c["score"] for c in context_chunks) / len(context_chunks)
            
        # 3. LLM Grading or Heuristic Fallback
        context_coverage = 0.5
        response_grounding = 0.5
        
        if self.llm and context_chunks:
            formatted_context = "\n\n".join([c["page_content"] for c in context_chunks])
            try:
                prompt = PromptTemplate.from_template(EVAL_TEMPLATE)
                chain = prompt | self.llm
                eval_res = chain.invoke({
                    "context": formatted_context,
                    "query": query,
                    "response": response
                })
                
                parsed = parse_json_safely(eval_res.content)
                context_coverage = float(parsed.get("context_coverage", 0.5))
                response_grounding = float(parsed.get("response_grounding", 0.5))
                
                # Clamp values
                context_coverage = max(0.0, min(1.0, context_coverage))
                response_grounding = max(0.0, min(1.0, response_grounding))
                logger.info(f"LLM Eval scores: Coverage={context_coverage:.2f}, Grounding={response_grounding:.2f}")
            except Exception as e:
                logger.error(f"Error during LLM evaluation grading: {str(e)}. Using heuristics.")
                context_coverage, response_grounding = self._heuristic_evaluate(query, context_chunks, response)
        else:
            context_coverage, response_grounding = self._heuristic_evaluate(query, context_chunks, response)

        # 4. Weighted Confidence Score
        # Weights: 30% retrieval similarity, 35% context coverage, 35% response grounding
        confidence = (0.30 * retrieval_sim) + (0.35 * context_coverage) + (0.35 * response_grounding)
        confidence = round(max(0.0, min(1.0, confidence)), 2)

        # Map to levels
        if confidence > 0.80:
            level = "HIGH"
        elif confidence >= 0.50:
            level = "MEDIUM"
        else:
            level = "LOW"

        logger.info(f"Final evaluated confidence: {confidence*100:.1f}% ({level})")
        return {
            "confidence": confidence,
            "level": level,
            "retrieval_similarity": round(retrieval_sim, 2),
            "context_coverage": round(context_coverage, 2),
            "response_grounding": round(response_grounding, 2)
        }

    def _heuristic_evaluate(self, query: str, context_chunks: List[Dict[str, Any]], response: str) -> tuple:
        """Heuristic-based grader when LLM is unavailable or context is empty."""
        if not context_chunks:
            return 0.0, 1.0  # Zero coverage, but technically grounded if response matches insufficient context.
            
        # Context coverage heuristic: intersection of query words in context
        query_words = set(re.findall(r"\w+", query.lower()))
        # Remove common stop words
        stop_words = {"what", "how", "the", "a", "is", "are", "do", "you", "i", "to", "in", "of", "and", "or", "for", "with", "my", "on", "using"}
        query_keywords = query_words - stop_words
        
        if not query_keywords:
            coverage = 0.5
        else:
            context_text = " ".join([c["page_content"].lower() for c in context_chunks])
            matches = sum(1 for w in query_keywords if w in context_text)
            coverage = matches / len(query_keywords)
            coverage = min(1.0, coverage)
            
        # Grounding heuristic: intersection of response words in context
        response_words = set(re.findall(r"\w+", response.lower()))
        response_keywords = response_words - stop_words
        
        if not response_keywords:
            grounding = 1.0
        else:
            context_text = " ".join([c["page_content"].lower() for c in context_chunks])
            matches = sum(1 for w in response_keywords if w in context_text)
            # Add a small buffer for natural vocabulary variance
            grounding = (matches / len(response_keywords)) * 1.2
            grounding = min(1.0, grounding)
            
        return round(coverage, 2), round(grounding, 2)

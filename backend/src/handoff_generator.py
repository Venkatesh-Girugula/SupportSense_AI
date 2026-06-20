import json
import logging
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from backend.src.config import Config
from backend.src.utils import get_logger, parse_json_safely

logger = get_logger("handoff_generator")

RECOMMENDATION_TEMPLATE = """You are a senior customer support manager at CloudFlow CRM.
Analyze the conversation history below between a customer support AI agent and a customer.
Formulate a recommendation for a human support agent who will take over this case.

Your recommendation should be:
1. Highly actionable.
2. Short (1 sentence).
3. Directly related to resolving the customer's problem.

Conversation History:
{history}

Your output must be a valid JSON object matching this schema:
{{
  "issue_summary": "<short description of the core issue>",
  "recommendation": "<concise action item for the human support agent>"
}}

Provide ONLY the raw JSON object. Do not include formatting or other text.
"""

class HandoffGenerator:
    """Generates structured JSON transfer tickets containing customer details, history, and agent recommendations."""
    
    def __init__(self):
        self.api_key = Config.GEMINI_API_KEY
        if self.api_key:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=Config.LLM_MODEL,
                    google_api_key=self.api_key,
                    temperature=0.1,
                    model_kwargs={"response_format": {"type": "json_object"}}
                )
                logger.info("Initialized Gemini for Human Handoff Recommendations.")
            except Exception as e:
                logger.error(f"Error initializing handoff LLM: {str(e)}")
                self.llm = None
        else:
            self.llm = None

    def generate_summary(self, 
                         persona: str, 
                         memory_states: Dict[str, Any], 
                         documents_used: List[str], 
                         history_string: str) -> Dict[str, Any]:
        """
        Creates a structured handoff ticket.
        
        Returns:
            Dict: JSON structure matching the required ticket format.
        """
        logger.info("Generating human handoff summary ticket...")
        
        issue_summary = ", ".join(memory_states.get("previous_issues", [])) or "General CRM Support Inquiry"
        recommendation = "Review customer logs and follow up via email."
        
        # 1. semantic generation of summary and recommendations
        if self.llm and history_string:
            try:
                prompt = PromptTemplate.from_template(RECOMMENDATION_TEMPLATE)
                chain = prompt | self.llm
                result = chain.invoke({"history": history_string})
                parsed = parse_json_safely(result.content)
                
                issue_summary = parsed.get("issue_summary", issue_summary)
                recommendation = parsed.get("recommendation", recommendation)
                logger.info("Successfully generated handoff recommendation using LLM.")
            except Exception as e:
                logger.error(f"Error generating recommendation using LLM: {str(e)}. Using fallback.")
                recommendation = self._fallback_recommendation(issue_summary)
        else:
            recommendation = self._fallback_recommendation(issue_summary)

        # 2. Build structured handoff JSON
        handoff_ticket = {
            "persona": persona,
            "issue": issue_summary,
            "documents_used": list(set(documents_used)),
            "attempted_steps": memory_states.get("attempted_actions", []),
            "recommendation": recommendation
        }
        
        logger.info(f"Handoff ticket generated: {json.dumps(handoff_ticket)}")
        return handoff_ticket

    def _fallback_recommendation(self, issue: str) -> str:
        """Rule-based recommendations when LLM service is offline."""
        issue_lower = issue.lower()
        if "auth" in issue_lower or "token" in issue_lower:
            return "Verify client's API keys, security headers, and webhook endpoints logs."
        elif "billing" in issue_lower or "refund" in issue_lower:
            return "Review billing account charge history and authorize a manual refund if inside the 14-day window."
        elif "security" in issue_lower or "compliance" in issue_lower:
            return "Send the user the SOC 2 Type II compliance disclosure NDA packet for signature."
        elif "password" in issue_lower or "lockout" in issue_lower:
            return "Manually reset user's account password via standard admin portal tools."
        return "Review case logs and follow up with the user directly."

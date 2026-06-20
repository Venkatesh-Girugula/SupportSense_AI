import logging
from typing import Dict, Any
from backend.src.config import Config
from backend.src.utils import get_logger

logger = get_logger("escalation_engine")

class EscalationEngine:
    """Evaluates business compliance rules and technical parameters to determine if a case requires human support."""
    
    def __init__(self):
        self.threshold = Config.CONFIDENCE_THRESHOLD

    def evaluate(self, 
                 query: str, 
                 response: str,
                 confidence_data: Dict[str, Any], 
                 memory_states: Dict[str, Any],
                 retrieved_count: int) -> Dict[str, Any]:
        """
        Assesses if a ticket should be escalated based on multi-turn history states and rule compliance.

        Returns:
            Dict: {
                "escalated": bool,
                "reason": str
            }
        """
        query_lower = query.lower()
        confidence_score = confidence_data.get("confidence", 1.0)
        
        logger.info("Evaluating escalation rules...")
        
        # Rule 1: Confidence level falls below acceptable threshold
        if confidence_score < self.threshold:
            logger.warning(f"Escalation triggered: low confidence score ({confidence_score})")
            return {
                "escalated": True,
                "reason": f"Confidence score ({confidence_score:.2f}) fell below the acceptable threshold ({self.threshold:.2f})."
            }
            
        # Rule 2: Zero context documents retrieved from knowledge base
        if retrieved_count == 0:
            logger.warning("Escalation triggered: zero documents retrieved")
            return {
                "escalated": True,
                "reason": "Knowledge base did not return any relevant references matching the user's query."
            }
            
        # Rule 3: Insufficient documentation coverage flag detected in the response
        if "[INSUFFICIENT_CONTEXT]" in response:
            logger.warning("Escalation triggered: insufficient context in response")
            return {
                "escalated": True,
                "reason": "Retrieved documentation does not cover the topic. Answer not present in context."
            }

        # Rule 4: Billing issues, pricing discrepancies, or refund requests
        billing_keywords = [
            "refund", "billing dispute", "overcharge", "charged twice", "cancel subscription", 
            "billing error", "credit card charge", "chargeback", "invoice mismatch", "subscription cost"
        ]
        if any(kw in query_lower for kw in billing_keywords):
            logger.warning("Escalation triggered: billing/refund sensitive operation")
            return {
                "escalated": True,
                "reason": "Subscription billing disputes and refund transactions require secure human manager verification."
            }

        # Rule 5: Legal or compliance concerns
        legal_keywords = [
            "legal action", "lawsuit", "suing", "attorney", "court", "breach of contract", 
            "terms violation", "compliance audit", "sue", "legal advice", "arbitration"
        ]
        if any(kw in query_lower for kw in legal_keywords):
            logger.warning("Escalation triggered: legal concern detected")
            return {
                "escalated": True,
                "reason": "Legal notices, regulatory compliance audits, or contractual disputes must be routed to corporate legal."
            }

        # Rule 6: Account-sensitive security operations (GDPR deletion, manual password changes by support)
        account_keywords = [
            "delete account", "terminate account", "remove my data", "gdpr deletion", 
            "change password manually", "override lockout", "reset authentication", "close account"
        ]
        if any(kw in query_lower for kw in account_keywords):
            logger.warning("Escalation triggered: account-sensitive operations")
            return {
                "escalated": True,
                "reason": "Sensitive account modifications (deletions, lockout overrides) require manual identity validation."
            }

        # Rule 7: Persisting user frustration or repeated troubleshooting failures
        repeated_failures = memory_states.get("repeated_failures", 0)
        user_sentiment = memory_states.get("user_sentiment", "Neutral")
        
        if repeated_failures >= 2:
            logger.warning(f"Escalation triggered: repeated failures count = {repeated_failures}")
            return {
                "escalated": True,
                "reason": f"Customer has encountered repeated failures ({repeated_failures} times) during automated troubleshooting."
            }
            
        if user_sentiment == "Highly Frustrated" and repeated_failures >= 1:
            logger.warning("Escalation triggered: frustrated user with failure history")
            return {
                "escalated": True,
                "reason": "Customer is highly frustrated, and at least one technical troubleshooting step has failed."
            }

        logger.info("Escalation evaluation: Resolved successfully (No escalation required)")
        return {
            "escalated": False,
            "reason": "None"
        }

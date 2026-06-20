import time
import re
from typing import List, Dict, Any
try:
    from langchain_community.chat_message_histories import ChatMessageHistory
except ImportError:
    try:
        from langchain_core.chat_history import InMemoryChatMessageHistory as ChatMessageHistory
    except ImportError:
        from langchain.memory.chat_message_histories import ChatMessageHistory

try:
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    from langchain_classic.memory import ConversationBufferWindowMemory
from backend.src.utils import get_logger

logger = get_logger("memory_manager")

class MemoryManager:
    """Manages multi-turn conversation memory, history formatting, and metadata state analysis."""
    
    def __init__(self, k: int = 6):
        # Raw conversation history backing
        self.history = ChatMessageHistory()
        
        # LangChain memory abstraction
        self.memory = ConversationBufferWindowMemory(
            chat_memory=self.history,
            k=k,
            return_messages=True,
            memory_key="chat_history",
            input_key="input",
            output_key="output"
        )
        
        # State variables specified in requirements
        self.previous_issues: List[str] = []
        self.attempted_actions: List[str] = []
        self.user_sentiment: str = "Neutral"
        self.repeated_failures: int = 0
        
        # Timestamped serialization logs for the UI components
        self.chat_history_log: List[Dict[str, Any]] = []

    def add_turn(self, user_query: str, assistant_response: str, persona: str, confidence: float):
        """
        Saves a conversation turn to LangChain memory and updates session analytics states.
        """
        # Save to LangChain history
        self.memory.save_context({"input": user_query}, {"output": assistant_response})
        
        # Record structured history logs
        timestamp = time.strftime("%H:%M:%S")
        self.chat_history_log.append({
            "timestamp": timestamp,
            "role": "user",
            "content": user_query,
            "persona": persona,
            "confidence": confidence
        })
        self.chat_history_log.append({
            "timestamp": timestamp,
            "role": "assistant",
            "content": assistant_response
        })
        
        # Update metrics states
        self._update_sentiment(persona)
        self._update_failures(user_query)
        self._extract_attempted_actions(user_query)
        self._extract_issues(user_query)

    def _update_sentiment(self, persona: str):
        """Sets sentiment labels. Frustrated persona results in frustrated sentiment."""
        if persona == "Frustrated User":
            self.user_sentiment = "Highly Frustrated"
        else:
            self.user_sentiment = "Neutral / Professional"

    def _update_failures(self, query: str):
        """Scans query to detect if previous troubleshooting steps failed to resolve issue."""
        query_lower = query.lower()
        failure_signals = [
            "still not", "not working", "still error", "failed again", "does not work",
            "doesn't work", "cannot access", "keeps failing", "issue persists", "still broken"
        ]
        if any(signal in query_lower for signal in failure_signals):
            self.repeated_failures += 1
            logger.info(f"Repeated failures count incremented to: {self.repeated_failures}")

    def _extract_attempted_actions(self, query: str):
        """Heuristic rule parsing to extract what the user claims they already tried."""
        query_lower = query.lower()
        
        # Look for phrases like "I tried resetting my password" or "already cleared the cache"
        patterns = [
            r"(?:tried|did|attempted|cleared|reset|configured)\s+([a-z0-9\s]{4,30})(?:\b|and|but|with|then|,|\.)",
            r"(?:already|previously)\s+([a-z0-9\s]{4,30})(?:\b|and|but|with|then|,|\.)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                action = match.strip()
                # Exclude simple filler words and filter string lengths
                if len(action) > 4 and action not in ["this", "that", "it", "again", "before"]:
                    formatted_action = action.capitalize()
                    if formatted_action not in self.attempted_actions:
                        self.attempted_actions.append(formatted_action)
                        logger.info(f"Extracted attempted user action: {formatted_action}")

    def _extract_issues(self, query: str):
        """Identifies and stores the primary technical issue topic of the turn."""
        query_lower = query.lower()
        issue = None
        
        if "auth" in query_lower or "token" in query_lower or "key" in query_lower or "bearer" in query_lower:
            issue = "API Authentication Failure"
        elif "refund" in query_lower or "charge" in query_lower or "bill" in query_lower or "pricing" in query_lower:
            issue = "Subscription Billing / Refund Inquiry"
        elif "security" in query_lower or "compliance" in query_lower or "gdpr" in query_lower or "soc 2" in query_lower:
            issue = "Security & GDPR Data Compliance"
        elif "password" in query_lower or "login" in query_lower or "lockout" in query_lower:
            issue = "Account Lockout & Password Recovery"
        else:
            # Fallback to the opening query terms
            words = [w for w in query.split() if w.isalnum()]
            if len(words) > 0:
                issue = " ".join(words[:4]).capitalize() + "..."
                
        if issue and issue not in self.previous_issues:
            self.previous_issues.append(issue)

    def get_history_string(self) -> str:
        """Returns the conversation history formatted as a string block for prompt injection."""
        messages = self.memory.load_memory_variables({}).get("chat_history", [])
        history_str = ""
        for msg in messages:
            role = "User" if msg.type == "human" else "Agent"
            history_str += f"{role}: {msg.content}\n"
        return history_str

    def get_state_summary(self) -> Dict[str, Any]:
        """Provides a dictionary representing the session metrics states."""
        return {
            "previous_issues": self.previous_issues,
            "attempted_actions": self.attempted_actions,
            "user_sentiment": self.user_sentiment,
            "repeated_failures": self.repeated_failures
        }

    def clear(self):
        """Resets all states, logs, and LangChain buffers."""
        self.memory.clear()
        self.previous_issues.clear()
        self.attempted_actions.clear()
        self.user_sentiment = "Neutral"
        self.repeated_failures = 0
        self.chat_history_log.clear()
        logger.info("MemoryManager state cleared.")

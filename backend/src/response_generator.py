import logging
from typing import List, Dict, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from backend.src.config import Config
from backend.src.utils import get_logger

logger = get_logger("response_generator")

# 1. Prompt template for Technical Expert
TECH_TEMPLATE = """You are an senior systems engineer and CloudFlow CRM API architect.
The user is a Technical Expert. Answer their query using a highly detailed, technical tone, referencing root-causes, specific configuration parameters, response codes, payloads, or schemas.
Use code blocks, HTTP requests, or command-line commands where appropriate.

RETRIEVED DOCUMENTATION CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUERY: {query}

INSTRUCTIONS:
1. Ground your response STRICTLY in the RETRIEVED DOCUMENTATION CONTEXT. Do not speculate or hallucinate.
2. If the retrieved documentation context does not contain the answer, you must respond EXACTLY with:
"[INSUFFICIENT_CONTEXT] I do not have sufficient technical documentation in my knowledge base to resolve this issue."
3. Keep the content clear, structured, and developer-focused.
"""

# 2. Prompt template for Frustrated User
FRUSTRATED_TEMPLATE = """You are an empathetic customer success specialist at CloudFlow CRM.
The user is highly Frustrated. Your response must be extremely calming, reassuring, validating, and focused on simple, direct, action-oriented next steps.
Avoid deep technical jargon. Use structured lists for actions. Reassure them that you are resolving their issue.

RETRIEVED DOCUMENTATION CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUERY: {query}

INSTRUCTIONS:
1. Ground your response STRICTLY in the RETRIEVED DOCUMENTATION CONTEXT.
2. Acknowledge and validate their frustration in the first sentence (e.g. "I completely understand how frustrating it is when...").
3. If the retrieved documentation context does not contain the answer, you must respond EXACTLY with:
"[INSUFFICIENT_CONTEXT] I apologize for the trouble you are experiencing. I could not find a resolution in our standard support guides."
4. Provide simple, easy-to-follow steps to fix their issue.
"""

# 3. Prompt template for Business Executive
EXEC_TEMPLATE = """You are an accounts and customer relations director at CloudFlow CRM.
The user is a Business Executive. Your response must be concise, outcome-focused, explaining the business impact, SLA guarantees, data compliance (GDPR/SOC 2), timelines, and high-level resolution steps.
Do not write detailed config scripts or debug parameters. Keep it executive-ready and professional.

RETRIEVED DOCUMENTATION CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUERY: {query}

INSTRUCTIONS:
1. Ground your response STRICTLY in the RETRIEVED DOCUMENTATION CONTEXT.
2. If the retrieved documentation context does not contain the answer, you must respond EXACTLY with:
"[INSUFFICIENT_CONTEXT] I am unable to locate the specific SLA, billing, or compliance details in our standard business reference guides."
3. Keep it brief, professional, and business-focused.
"""

class ResponseGenerator:
    """Generates grounded responses tailored to the user's communication persona."""
    
    def __init__(self):
        self.api_key = Config.GEMINI_API_KEY
        if self.api_key:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=Config.LLM_MODEL,
                    google_api_key=self.api_key,
                    temperature=0.2
                )
                logger.info(f"Initialized Gemini model '{Config.LLM_MODEL}' for Response Generation.")
            except Exception as e:
                logger.error(f"Error initializing LLM for ResponseGenerator: {str(e)}")
                self.llm = None
        else:
            logger.warning("GEMINI_API_KEY not set. ResponseGenerator starting in offline fallback mode.")
            self.llm = None

        # Build prompt objects
        self.prompts = {
            "Technical Expert": PromptTemplate.from_template(TECH_TEMPLATE),
            "Frustrated User": PromptTemplate.from_template(FRUSTRATED_TEMPLATE),
            "Business Executive": PromptTemplate.from_template(EXEC_TEMPLATE)
        }

    def generate(self, query: str, context_chunks: List[Dict[str, Any]], persona: str, history: str) -> Tuple[str, str]:
        """
        Generates a grounded, persona-adaptive response based on query, context, and history.
        
        Returns:
            Tuple[str, str]: (response_content, reasoning_used)
        """
        logger.info(f"Generating response for persona: {persona}")
        
        # 1. Format context string
        if not context_chunks:
            formatted_context = "NO RELEVANT CONTEXT FOUND."
        else:
            formatted_context = "\n\n".join([
                f"Source: {chunk['metadata'].get('source', 'unknown')}\n"
                f"Content: {chunk['page_content']}"
                for chunk in context_chunks
            ])

        # Create reasoning trace to show in the UI
        reasoning = (
            f"Active Persona: {persona}\n"
            f"Sources Evaluated: {[c['metadata']['source'] for c in context_chunks]}\n"
            f"Average Source Relevance Score: {sum(c['score'] for c in context_chunks)/len(context_chunks) if context_chunks else 0.0:.3f}\n"
            f"History Context Length: {len(history)} chars\n"
        )
        
        if not self.llm:
            return self._fallback_generate(query, context_chunks, persona), reasoning + "Method: Offline Fallback"
            
        # 2. Get correct template
        prompt_template = self.prompts.get(persona, self.prompts["Technical Expert"])
        
        try:
            chain = prompt_template | self.llm
            response = chain.invoke({
                "context": formatted_context,
                "history": history,
                "query": query
            })
            
            logger.info("Response generated successfully via LLM.")
            return response.content, reasoning + "Method: Gemini Generation"
        except Exception as e:
            logger.error(f"Error during response generation: {str(e)}")
            return self._fallback_generate(query, context_chunks, persona), reasoning + f"Method: Fallback (Error: {str(e)})"

    def _fallback_generate(self, query: str, context_chunks: List[Dict[str, Any]], persona: str) -> str:
        """Heuristic answer builder running when LLM is unavailable."""
        if not context_chunks:
            return "[INSUFFICIENT_CONTEXT] I'm sorry, I could not retrieve any reference documents matching your query."
            
        # Simple extraction/summarization fallback
        best_chunk = context_chunks[0]
        content = best_chunk["page_content"]
        source = best_chunk["metadata"]["source"]
        
        if persona == "Technical Expert":
            return (
                f"### [Technical Offline Support]\n"
                f"Retrieved technical document: **{source}**\n\n"
                f"Here is the relevant documentation extract:\n"
                f"```text\n{content}\n```\n"
                f"Please verify this configuration aligns with your settings."
            )
        elif persona == "Frustrated User":
            return (
                f"I completely understand this is frustrating, and I want to help you resolve this quickly. "
                f"According to our support guides (source: **{source}**):\n\n"
                f"{content[:200]}...\n\n"
                f"If this does not help you get up and running, please request a human handoff."
            )
        else: # Business Executive
            return (
                f"Our records in **{source}** indicate the following:\n\n"
                f"*{content[:150]}...*\n\n"
                f"We commit to resolving high-priority account issues under our SLA guidelines."
            )

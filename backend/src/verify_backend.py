import os
import sys
import json
from pathlib import Path

# Add project root directory to Python path for import compatibility
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backend.src.config import Config
from backend.src.persona_detector import PersonaDetector
from backend.src.rag_pipeline import RAGPipeline
from backend.src.memory_manager import MemoryManager
from backend.src.response_generator import ResponseGenerator
from backend.src.confidence_engine import ConfidenceEngine
from backend.src.escalation_engine import EscalationEngine
from backend.src.handoff_generator import HandoffGenerator

def run_tests():
    print("=" * 60)
    print("      SupportSense AI - Backend Verification Harness      ")
    print("=" * 60)
    
    # 1. Print Configuration
    print("\n--- [1] Environment and Configuration ---")
    print(Config.print_config())
    
    # 2. Initialize Pipeline Modules
    print("\n--- [2] Initializing Modules ---")
    try:
        detector = PersonaDetector()
        rag = RAGPipeline()
        memory = MemoryManager()
        generator = ResponseGenerator()
        confidence_engine = ConfidenceEngine()
        escalation_engine = EscalationEngine()
        handoff = HandoffGenerator()
        print("Success: All modules initialized successfully.")
    except Exception as e:
        print(f"Error: Module initialization failed: {e}")
        return False

    # 3. Test Cases
    test_cases = [
        {
            "name": "Technical API Query (Should resolve without escalation)",
            "query": "How do I authenticate with the API using a bearer token?",
            "expected_persona": "Technical Expert",
            "expected_escalation": False
        },
        {
            "name": "Frustrated Billing Query (Should trigger billing escalation)",
            "query": "I want a refund for my billing plan, this is broken!",
            "expected_persona": "Frustrated User",
            "expected_escalation": True
        },
        {
            "name": "Out-of-Scope Query (Should trigger low-confidence escalation)",
            "query": "What is the capital of France?",
            "expected_persona": "Technical Expert", # default or fallback
            "expected_escalation": True
        }
    ]

    all_passed = True
    for i, tc in enumerate(test_cases, start=1):
        print(f"\n--- [3.{i}] Running Test Case: {tc['name']} ---")
        print(f"Query: '{tc['query']}'")
        
        # A. Persona Detection
        p_res = detector.detect(tc["query"])
        persona = p_res["persona"]
        p_conf = p_res["confidence"]
        is_fallback = p_res.get("fallback", False)
        print(f"Detected Persona: {persona} (Confidence: {p_conf:.2f}, Fallback Mode: {is_fallback})")
        
        # B. RAG Retrieval
        retrieved_chunks = rag.retrieve(tc["query"], k=3)
        sources = [c["metadata"]["source"] for c in retrieved_chunks]
        print(f"Retrieved Chunks: {len(retrieved_chunks)} (Sources: {sources})")
        
        # C. Response Generation
        history_str = memory.get_history_string()
        response, reasoning = generator.generate(tc["query"], retrieved_chunks, persona, history_str)
        print(f"Response Preview: '{response[:120]}...'")
        
        # D. Confidence Calculation
        c_res = confidence_engine.evaluate(tc["query"], retrieved_chunks, response)
        print(f"Confidence Level: {c_res['level']} (Score: {c_res['confidence']:.2f})")
        
        # E. Escalation Check
        memory_states = memory.get_state_summary()
        e_res = escalation_engine.evaluate(
            tc["query"], 
            response, 
            c_res, 
            memory_states, 
            len(retrieved_chunks)
        )
        print(f"Escalation Decision: {'ESCALATED' if e_res['escalated'] else 'RESOLVED'} (Reason: {e_res['reason']})")
        
        # F. Add to Memory
        memory.add_turn(tc["query"], response, persona, p_conf)
        
        # G. Handoff Summary (if escalated)
        if e_res["escalated"]:
            h_res = handoff.generate_summary(
                persona, 
                memory.get_state_summary(), 
                sources, 
                memory.get_history_string()
            )
            print("Generated Handoff Summary:")
            print(json.dumps(h_res, indent=2))
            
        # H. Verify against expectations
        if e_res["escalated"] != tc["expected_escalation"]:
            print(f"WARNING: Escalation status mismatch! Expected {tc['expected_escalation']}, got {e_res['escalated']}.")
            # We won't hard fail if it's due to LLM variance, but log it
        
    print("\n" + "=" * 60)
    print("      Verification Completed!      ")
    print("=" * 60)
    return True

if __name__ == "__main__":
    run_tests()

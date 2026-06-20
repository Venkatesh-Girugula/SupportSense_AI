import sys
import os
import json
import time
from pathlib import Path
import streamlit as st
import pandas as pd

# Append project root to import modules cleanly
frontend_dir = Path(__file__).resolve().parent
project_root = frontend_dir.parent
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

import components

# Page Configurations
st.set_page_config(
    page_title="SupportSense AI - Persona Customer Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS Stylesheet
def inject_custom_css():
    css_file = frontend_dir / "style.css"
    if css_file.exists():
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("CSS Style file missing. Using standard styles.")

inject_custom_css()

# Session State Initialization
if "memory" not in st.session_state:
    st.session_state.memory = MemoryManager()
if "last_key" not in st.session_state:
    st.session_state.last_key = Config.GEMINI_API_KEY
if "is_escalated" not in st.session_state:
    st.session_state.is_escalated = False
if "escalation_reason" not in st.session_state:
    st.session_state.escalation_reason = ""
if "handoff_data" not in st.session_state:
    st.session_state.handoff_data = None
if "active_persona" not in st.session_state:
    st.session_state.active_persona = "Technical Expert"
if "confidence_score" not in st.session_state:
    st.session_state.confidence_score = 1.0

# Initialize Pipelines function
def init_backend_components():
    with st.spinner("Initializing AI Engines..."):
        try:
            st.session_state.detector = PersonaDetector()
            st.session_state.rag = RAGPipeline()
            st.session_state.generator = ResponseGenerator()
            st.session_state.confidence_engine = ConfidenceEngine()
            st.session_state.escalation_engine = EscalationEngine()
            st.session_state.handoff = HandoffGenerator()
            st.session_state.db_loaded = True
        except Exception as e:
            st.error(f"Failed to load backend pipelines: {str(e)}")
            st.session_state.db_loaded = False

if "detector" not in st.session_state:
    init_backend_components()

# ==========================================
# SIDEBAR PANEL
# ==========================================
st.sidebar.markdown(
    "<h2 style='text-align: center; color: #38BDF8;'>SupportSense AI</h2>", 
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "<p style='text-align: center; font-size:0.85rem; color:#94A3B8; margin-top:-10px; margin-bottom: 20px;'>"
    "Persona-Aware Customer Support Intelligence</p>", 
    unsafe_allow_html=True
)

# API Configuration
st.sidebar.subheader("API Access Config")
api_key_input = st.sidebar.text_input(
    "Google Gemini API Key", 
    type="password", 
    value=Config.GEMINI_API_KEY,
    help="Provide a Gemini key to run live semantic models. Fallback rules are activated if empty."
)

# Check and update key changes
if api_key_input != st.session_state.last_key:
    Config.GEMINI_API_KEY = api_key_input
    st.session_state.last_key = api_key_input
    init_backend_components()
    st.sidebar.success("API Key updated! Engines reloaded.")

# System Status Panel
docs_count = 0
if st.session_state.db_loaded and hasattr(st.session_state.rag, 'vector_store') and st.session_state.rag.vector_store:
    try:
        # Get count of documents by accessing faiss index details
        docs_count = st.session_state.rag.vector_store.index.ntotal
    except:
        docs_count = 3 # fallback to mock count
        
conv_count = len(st.session_state.memory.chat_history_log) // 2

st.sidebar.html(
    components.render_system_status(
        st.session_state.db_loaded, 
        st.session_state.is_escalated, 
        docs_count, 
        conv_count
    )
)

# Document Upload
st.sidebar.subheader("Knowledge Base")
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents", 
    accept_multiple_files=True, 
    type=["pdf", "txt", "md"],
    help="Upload documents to the knowledge base. The vector index will be rebuilt automatically."
)

if uploaded_files:
    if st.sidebar.button("Process & Upload Documents", type="primary"):
        with st.spinner("Saving documents and rebuilding index..."):
            docs_dir = Path(Config.DOCS_DIR)
            docs_dir.mkdir(parents=True, exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = docs_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            try:
                st.session_state.rag.rebuild_index()
                st.sidebar.success(f"{len(uploaded_files)} documents processed!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to process documents: {e}")

# Rebuild Index Button
if st.sidebar.button("Rebuild Knowledge Vector Index", use_container_width=True):
    with st.spinner("Re-indexing docs directory..."):
        try:
            st.session_state.rag.rebuild_index()
            st.sidebar.success("Vector Database Re-built successfully!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Rebuild failed: {e}")

# Persona Override (For Recruiters demonstration)
st.sidebar.subheader("Developer Overrides")
persona_override = st.sidebar.selectbox(
    "Override Persona Classifier",
    options=["Auto-Detect Persona", "Technical Expert", "Frustrated User", "Business Executive"],
    help="Force a specific persona configuration to inspect custom responses and interface reactions."
)

# Escalation settings override
confidence_threshold = st.sidebar.slider(
    "Escalation Threshold", 
    min_value=0.20, 
    max_value=0.80, 
    value=Config.CONFIDENCE_THRESHOLD,
    step=0.05,
    help="Escalates support ticket if AI grounding confidence falls below this value."
)
if confidence_threshold != Config.CONFIDENCE_THRESHOLD:
    Config.CONFIDENCE_THRESHOLD = confidence_threshold
    st.session_state.escalation_engine.threshold = confidence_threshold

# Session reset
if st.sidebar.button("Reset Live Session", type="primary", use_container_width=True):
    st.session_state.memory.clear()
    st.session_state.is_escalated = False
    st.session_state.escalation_reason = ""
    st.session_state.handoff_data = None
    st.session_state.active_persona = "Technical Expert"
    st.session_state.confidence_score = 1.0
    st.success("Session variables reset successfully.")
    time.sleep(0.5)
    st.rerun()

# ==========================================
# MAIN INTERFACE HEADER
# ==========================================
st.markdown("<h1 style='margin-bottom:0px;'>SupportSense AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.15rem; color:#94A3B8; margin-top:0px;'>Persona-Aware Customer Support Agent</p>", unsafe_allow_html=True)

# Grid layout for active indicators
stat_col1, stat_col2 = st.columns(2)

with stat_col1:
    st.html(
        components.render_persona_card(
            st.session_state.active_persona, 
            st.session_state.confidence_score
        )
    )

with stat_col2:
    st.html(
        components.render_confidence_gauge(
            st.session_state.confidence_score, 
            "HIGH" if st.session_state.confidence_score > 0.8 else ("MEDIUM" if st.session_state.confidence_score >= 0.5 else "LOW")
        )
    )

# ==========================================
# CHAT CONTAINER
# ==========================================
st.write("---")

# Rendering historical dialogue turns
for msg in st.session_state.memory.chat_history_log:
    role = msg["role"]
    content = msg["content"]
    
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
            st.markdown(
                f"<span style='font-size:0.78rem; color:#64748B;'>[{msg['timestamp']}] "
                f"Detected Persona: <strong>{msg.get('persona', 'Technical Expert')}</strong> "
                f"(Confidence: {int(msg.get('confidence', 1.0)*100)}%)</span>", 
                unsafe_allow_html=True
            )
    else:
        with st.chat_message("assistant"):
            st.write(content)
            
            # Retrieve supplementary logs if stored during transaction
            sources = msg.get("sources", [])
            reasoning = msg.get("reasoning", "")
            conf_data = msg.get("confidence_data", {})
            
            # Show sources
            if sources:
                with st.expander("Retrieved Sources", expanded=False):
                    for src in sources:
                        st.markdown(
                            f"📁 `{src.get('source', 'document')}` (Page {src.get('page', 1)}) — "
                            f"Similarity: **{int(src.get('score', 0.0)*100)}%**"
                        )
                        st.text_area(
                            label=f"Extract from {src.get('source')}",
                            value=src.get("content", ""),
                            height=80,
                            disabled=True,
                            key=f"hist_src_{src.get('source')}_{time.time()}"
                        )
                        
            # Show reasoning trace
            if reasoning:
                with st.expander("Grounding & Reasoning Details", expanded=False):
                    st.code(reasoning, language="yaml")
                    
            # If this historical turn was escalated
            if msg.get("escalated", False):
                st.html(components.render_escalation_banner(msg.get("escalation_reason", "Low Confidence")))

# ==========================================
# USER INPUT HANDLING
# ==========================================
user_query = st.chat_input("Describe your issue (e.g., questions on API auth, pricing, password locks)...")

if user_query:
    # 1. User turn display
    with st.chat_message("user"):
        st.write(user_query)
        # Render a quick loading indicator
        metadata_placeholder = st.empty()
        metadata_placeholder.markdown("<span style='font-size:0.8rem; color:#64748B;'>Analyzing query...</span>", unsafe_allow_html=True)
        
    # 2. Process query
    with st.chat_message("assistant"):
        # A. Persona Detection
        history_context = st.session_state.memory.get_history_string()
        
        # Resolve active persona (dynamic vs override)
        if persona_override == "Auto-Detect Persona":
            p_data = st.session_state.detector.detect(user_query, history_context)
            persona = p_data["persona"]
            persona_confidence = p_data["confidence"]
        else:
            persona = persona_override
            persona_confidence = 1.0
            
        st.session_state.active_persona = persona
        st.session_state.confidence_score = persona_confidence
        
        # B. Retrieve relevant chunks
        retrieved_data = st.session_state.rag.retrieve(user_query, k=3)
        sources_list = [
            {"source": c["metadata"]["source"], "page": c["metadata"]["page"], "score": c["score"], "content": c["page_content"]} 
            for c in retrieved_data
        ]
        
        # C. Generate Grounded Response
        response_text, reasoning_trace = st.session_state.generator.generate(
            user_query, 
            retrieved_data, 
            persona, 
            history_context
        )
        
        # Stream response
        response_placeholder = st.empty()
        full_response = ""
        for chunk in response_text.split(" "):
            full_response += chunk + " "
            response_placeholder.write(full_response + "▌")
            time.sleep(0.02)
        response_placeholder.write(response_text)
        
        # D. Run Confidence Engine
        confidence_results = st.session_state.confidence_engine.evaluate(user_query, retrieved_data, response_text)
        st.session_state.confidence_score = confidence_results["confidence"]
        
        # E. Escalation evaluation
        memory_stats = st.session_state.memory.get_state_summary()
        escalation_results = st.session_state.escalation_engine.evaluate(
            user_query,
            response_text,
            confidence_results,
            memory_stats,
            len(retrieved_data)
        )
        
        # Update metadata placeholder on user turn
        timestamp = time.strftime("%H:%M:%S")
        metadata_placeholder.markdown(
            f"<span style='font-size:0.78rem; color:#64748B;'>[{timestamp}] "
            f"Detected Persona: <strong>{persona}</strong> "
            f"(Confidence: {int(persona_confidence*100)}%)</span>", 
            unsafe_allow_html=True
        )

        # F. Display sources & evaluation panel
        if sources_list:
            with st.expander("Retrieved Sources", expanded=False):
                for src in sources_list:
                    st.markdown(
                        f"📁 `{src['source']}` (Page {src['page']}) — "
                        f"Similarity: **{int(src['score']*100)}%**"
                    )
                    st.text_area(
                        label=f"Extract from {src['source']}",
                        value=src["content"],
                        height=80,
                        disabled=True,
                        key=f"turn_src_{src['source']}_{time.time()}"
                    )
                    
        with st.expander("Grounding & Reasoning Details", expanded=False):
            st.code(
                f"{reasoning_trace}\n"
                f"Grounding Score: {confidence_results.get('response_grounding')}\n"
                f"Context Coverage: {confidence_results.get('context_coverage')}\n"
                f"SLA Escalation Flag: {escalation_results['escalated']}",
                language="yaml"
            )

        # G. Handle Handoff Summary if Escalated
        if escalation_results["escalated"]:
            st.session_state.is_escalated = True
            st.session_state.escalation_reason = escalation_results["reason"]
            
            # Print Escalation warning banner inside chat bubble
            st.html(components.render_escalation_banner(escalation_results["reason"]))
            
            # Generate transfer JSON
            h_sources = [c["metadata"]["source"] for c in retrieved_data]
            handoff_ticket = st.session_state.handoff.generate_summary(
                persona,
                st.session_state.memory.get_state_summary(),
                h_sources,
                st.session_state.memory.get_history_string()
            )
            st.session_state.handoff_data = handoff_ticket
        else:
            st.session_state.is_escalated = False
            st.session_state.escalation_reason = ""
            st.session_state.handoff_data = None
            
        # Save Turn to memory manager
        st.session_state.memory.add_turn(user_query, response_text, persona, persona_confidence)
        
        # Append meta metrics specifically to log lists to reload historical structures
        st.session_state.memory.chat_history_log[-2]["persona"] = persona
        st.session_state.memory.chat_history_log[-2]["confidence"] = persona_confidence
        st.session_state.memory.chat_history_log[-1]["sources"] = sources_list
        st.session_state.memory.chat_history_log[-1]["reasoning"] = reasoning_trace
        st.session_state.memory.chat_history_log[-1]["confidence_data"] = confidence_results
        st.session_state.memory.chat_history_log[-1]["escalated"] = escalation_results["escalated"]
        st.session_state.memory.chat_history_log[-1]["escalation_reason"] = escalation_results["reason"]
        
        # Force a refresh to update the sidebar status lights and top widgets
        st.rerun()

# ==========================================
# ESCALATION HANDOFF FOOTER
# ==========================================
if st.session_state.is_escalated and st.session_state.handoff_data:
    st.write("---")
    st.markdown("<h3 style='color: #EF4444;'>🎫 Customer Support Handoff Ticket</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #94A3B8; font-size:0.9rem;'>An automated transfer document has been prepared for the human agent.</p>", 
        unsafe_allow_html=True
    )
    
    col_ticket1, col_ticket2 = st.columns([2, 1])
    
    with col_ticket1:
        st.json(st.session_state.handoff_data)
        
    with col_ticket2:
        # Download Action Button
        json_str = json.dumps(st.session_state.handoff_data, indent=2)
        st.download_button(
            label="Download Handoff Ticket (JSON)",
            data=json_str,
            file_name="support_handoff_ticket.json",
            mime="application/json",
            type="primary",
            use_container_width=True
        )
        
        # Show mini statistics summaries
        st.markdown("**Session Diagnostics:**")
        states = st.session_state.memory.get_state_summary()
        st.markdown(f"- **Current Sentiment:** `{states['user_sentiment']}`")
        st.markdown(f"- **Troubleshooting Failures:** `{states['repeated_failures']}`")
        st.markdown(f"- **Issues Traversed:** `{', '.join(states['previous_issues']) or 'None'}`")

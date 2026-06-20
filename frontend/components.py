import html
from typing import List, Dict, Any

def render_system_status(db_loaded: bool, escalated: bool, docs_count: int, conv_count: int) -> str:
    """Renders the HTML for the sidebar system status panel."""
    status_class = "escalated" if escalated else "active"
    status_label = "ESCALATED TO HUMAN" if escalated else "AI ACTIVE"
    dot_class = "escalated" if escalated else "active"
    
    return f"""
    <div class="glass-card">
        <div class="glass-card-title">System Status</div>
        <div class="status-indicator" style="margin-bottom: 0.8rem;">
            <div class="status-dot {dot_class}"></div>
            <span style="font-weight: 600; font-size: 0.95rem;">{status_label}</span>
        </div>
        <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                <span>DB Engine:</span>
                <span style="color: #F8FAFC; font-weight: 500;">{"FAISS Active" if db_loaded else "Offline"}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                <span>Documents Indexed:</span>
                <span style="color: #38BDF8; font-weight: 600;">{docs_count}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Conversation Turns:</span>
                <span style="color: #F8FAFC; font-weight: 500;">{conv_count}</span>
            </div>
        </div>
    </div>
    """

def render_persona_card(persona: str, confidence_score: float) -> str:
    """Renders the active user persona badge."""
    p_lower = persona.lower()
    style_class = ""
    if "frustrated" in p_lower:
        style_class = "frustrated"
    elif "executive" in p_lower:
        style_class = "executive"
        
    conf_pct = int(confidence_score * 100)
    
    return f"""
    <div class="glass-card">
        <div class="glass-card-title">Detected User Persona</div>
        <div class="persona-container {style_class}">
            <div>
                <div class="persona-name">{persona}</div>
                <div style="font-size: 0.8rem; color: #E2E8F0; margin-top: 2px;">Semantic Classification</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC;">{conf_pct}%</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Confidence</div>
            </div>
        </div>
    </div>
    """

def render_confidence_gauge(score: float, level: str) -> str:
    """Renders the horizontal progress gauge representing pipeline confidence."""
    level_lower = level.lower()
    pct = int(score * 100)
    
    level_labels = {
        "HIGH": "rgba(16, 185, 129, 0.15)",
        "MEDIUM": "rgba(245, 158, 11, 0.15)",
        "LOW": "rgba(239, 68, 68, 0.15)"
    }
    
    level_colors = {
        "HIGH": "#10B981",
        "MEDIUM": "#F59E0B",
        "LOW": "#EF4444"
    }
    
    badge_bg = level_labels.get(level, "rgba(51, 65, 85, 0.15)")
    badge_color = level_colors.get(level, "#94A3B8")
    
    return f"""
    <div class="glass-card">
        <div class="glass-card-title" style="display:flex; justify-content:space-between; align-items:center;">
            <span>Confidence Index</span>
            <span style="background: {badge_bg}; color: {badge_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; border: 1px solid {badge_color}33;">
                {level}
            </span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 0.4rem;">
            <span style="font-size: 1.5rem; font-weight: 700; color: #F8FAFC;">{pct}%</span>
            <span style="font-size: 0.8rem; color: #94A3B8;">Overall Grounding Score</span>
        </div>
        <div class="confidence-gauge-bar">
            <div class="confidence-fill {level_lower}" style="width: {pct}%;"></div>
        </div>
    </div>
    """

def render_escalation_banner(reason: str) -> str:
    """Renders the attention-grabbing red escalation alert container."""
    safe_reason = html.escape(reason)
    return f"""
    <div class="escalation-banner">
        <div class="escalation-title">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #F43F5E;">
                <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>
                <line x1="12" y1="9" x2="12" y2="13"></line>
                <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
            Human Support Required
        </div>
        <div style="font-size: 0.95rem; color: #FDA4AF; line-height: 1.5; margin-bottom: 0.2rem;">
            This case matches active escalation rules and has been transferred to a live support manager.
        </div>
        <div style="font-size: 0.85rem; color: #E2E8F0; margin-top: 0.5rem; padding: 0.5rem; background: rgba(0,0,0,0.2); border-radius: 6px; border-left: 3px solid #F43F5E;">
            <strong>Trigger Reason:</strong> {safe_reason}
        </div>
    </div>
    """

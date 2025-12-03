# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import json
from datetime import datetime

# ---------------------------------------------------------
# Intentar importar el sistema de recomendaciones
# ---------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from recommender_system import GameRecommender
    from feedback_system import FeedbackSystem, ModelMonitoring
except Exception as e:
    # Si el import falla, guardamos el error para mostrarlo más abajo
    RecommenderImportError = e
    GameRecommender = None
    FeedbackSystem = None
    ModelMonitoring = None
else:
    RecommenderImportError = None

# ============================================
# INICIALIZACIÓN (después de cargar recommender)
# ============================================

@st.cache_resource
def load_feedback_system():
    if FeedbackSystem is None:
        st.error("⚠️ No se pudo cargar el sistema de feedback.")
        return None
    return FeedbackSystem()

@st.cache_resource
def load_monitoring():
    feedback_sys = load_feedback_system()
    if feedback_sys is None:
        return None
    return ModelMonitoring(feedback_sys)

# ---------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sistema Inteligente de Recomendación - Gaming",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Paleta y constantes (manteniendo el "vibe" original)
# ---------------------------------------------------------
GAMING_COLORS = {
    'primary': '#00FF88',      # Neon Green (accent)
    'secondary': '#0088FF',    # Electric Blue
    'accent': '#FF0088',       # Neon Pink
    'dark_bg': '#0A0A12',      # Dark background
    'card_bg': '#1A1A2E',      # Card background
    'input_bg': '#141420',     # NEW: For input areas
    'grid': '#2A2A3E',
    'text': '#E0E8FF',
    'highlight': '#FFFFFF',
    'light_text': '#F0F5FF',
    'medium_bg': '#252540',
}

PLAYSTYLE_PALETTE = ['#00FF88', '#0088FF', '#FF0088', '#FFAA00', '#AA00FF']
COMBAT_PALETTE = ['#FF5555', '#55FF55', '#5555FF', '#FFFF55', '#FF55FF']

# ---------------------------------------------------------
# CSS moderno y mejoras de UX integradas
# ---------------------------------------------------------
st.markdown(f"""
<style>
    :root {{
        --primary: {GAMING_COLORS['primary']};
        --secondary: {GAMING_COLORS['secondary']};
        --accent: {GAMING_COLORS['accent']};
        --bg-dark: {GAMING_COLORS['dark_bg']};
        --bg-card: {GAMING_COLORS['card_bg']};
        --bg-medium: {GAMING_COLORS['medium_bg']};
        --input-bg: {GAMING_COLORS['input_bg']}; /* New variable */
        --text-main: {GAMING_COLORS['text']};
        --text-light: {GAMING_COLORS['light_text']};
    }}

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Rajdhani:wght@600&display=swap');

    .stApp {{
        background: radial-gradient(circle at 20% 0%, #141420 0%, var(--bg-dark) 80%);
        color: var(--text-main);
        font-family: "Inter", sans-serif;
        letter-spacing: 0.2px;
    }}

    /* Headers */
    h1, h2, h3 {{
        font-family: "Rajdhani", sans-serif !important;
        font-weight: 600 !important;
        color: var(--primary) !important;
        text-shadow: 0 0 6px rgba(0,255,200,0.22);
        margin-bottom: 0.35rem;
    }}
    h1 {{ font-size: 2.2rem !important; }}
    h2 {{ font-size: 1.6rem !important; }}
    h3 {{ font-size: 1.15rem !important; }}

    /* Cards */
    .gaming-card {{
        background: linear-gradient(135deg, var(--bg-card), #191927);
        border-radius: 14px;
        padding: 1.25rem;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow:
            0 3px 8px rgba(0,0,0,0.45),
            0 0 8px rgba(0,255,180,0.06);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        animation: fadeIn 0.35s ease;
    }}
    .gaming-card:hover {{
        transform: translateY(-2px);
        box-shadow:
            0 6px 18px rgba(0,0,0,0.55),
            0 0 12px rgba(0,255,200,0.12);
    }}

    /* Metric cards */
    .metric-card {{
        background: #0f1016;
        padding: 1.1rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.07);
        box-shadow: 0 0 8px rgba(0,255,150,0.06);
    }}
    .metric-card h2 {{
        font-family: "Rajdhani", sans-serif !important;
        color: var(--accent);
        font-size: 2.2rem; /* Increased size for value */
        margin: 0.1rem 0;
    }}
    .metric-card h3 {{ 
        color: var(--text-light); 
        margin: 0; 
        font-size: 1rem; /* Increased size for label */
        font-weight: 400;
    }}
    .metric-card .sub-text {{ /* New class for the smallest text */
        font-size: 0.75rem; 
        color: #AFC3FF; 
        margin-top: 4px;
    }}

    /* Buttons */
    .stButton>button {{
        background: var(--secondary) !important;
        border-radius: 10px;
        border: none;
        color: white !important;
        padding: 0.55rem 1.6rem;
        font-weight: 600;
        box-shadow: 0 0 12px rgba(0,136,255,0.22);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    .stButton>button:hover {{
        transform: scale(1.03);
        box-shadow: 0 0 20px rgba(0,136,255,0.36);
        filter: brightness(1.1);
    }}

    /* Profile Buttons */
    .profile-button-primary {{
        background: var(--primary) !important;
        border-radius: 10px;
        border: none;
        color: var(--bg-dark) !important;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 0 15px rgba(0,255,136,0.25);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    .profile-button-primary:hover {{
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(0,255,136,0.4);
    }}

    .profile-button-secondary {{
        background: var(--accent) !important;
        border-radius: 10px;
        border: none;
        color: white !important;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 0 15px rgba(255,0,136,0.25);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    .profile-button-secondary:hover {{
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(255,0,136,0.4);
    }}

    /* Inputs & selects */
    .stSelectbox>div>div, .stTextInput>div>input, .stCheckbox>label {{
        background-color: var(--input-bg) !important; /* Changed to input_bg */
        color: var(--text-main) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        transition: border-color 0.2s ease;
    }}
    .stSelectbox>div:focus-within>div, .stTextInput>div>input:focus {{
        border-color: var(--primary) !important;
        box-shadow: 0 0 6px rgba(0,255,136,0.4);
    }}

    /* Slider Improvements - DUAL COLOR & LEGIBILITY FIX */
    .stSlider>div>div>div[data-baseweb="slider"] {{
        background: var(--bg-medium) !important; /* Inactive track */
        height: 6px; 
    }}
    .stSlider>div>div>div>div[data-testid="stThumbValue"] {{
        background-color: var(--secondary) !important; /* Filled portion */
        border-radius: 999px;
    }}
    .stSlider>div>div>div>div[data-testid="stThumb"] {{
        background-color: var(--primary) !important;
        border: 3px solid var(--bg-dark) !important; 
        box-shadow: 0 0 10px rgba(0,255,136,0.5);
    }}
    .stSlider>div>div>div>div[role="tooltip"] {{ /* FIX: Value Label Legibility */
        background: var(--bg-dark) !important; 
        color: var(--primary) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 6px;
        padding: 2px 8px;
        font-weight: 700;
        box-shadow: 0 0 6px rgba(0,255,136,0.3);
        top: -30px !important;
    }}

    /* Input Area - NO CARD BLEND-IN */
    .input-area-sleek {{
        background: var(--input-bg);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.05);
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background-color: var(--input-bg) !important; /* Use a darker background */
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.08) !important;
        padding: 10px 14px;
    }}

    /* Alerts */
    .stAlert {{
        background-color: var(--bg-medium) !important;
        border-left: 4px solid var(--accent) !important;
        border-radius: 8px;
    }}

    /* Footer tweaks */
    footer {{
        visibility: hidden;
    }}

    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Carga del modelo con caché
# ---------------------------------------------------------
@st.cache_resource
def load_recommender():
    if GameRecommender is None:
        raise RuntimeError(f"No se pudo importar GameRecommender: {RecommenderImportError}")
    return GameRecommender()

# ---------------------------------------------------------
# Funciones para gráficos (limpios y legibles)
# ---------------------------------------------------------

def show_feedback_section(session_id, predicted_style, player_data):
    """Muestra sección de feedback después del análisis"""
    
    st.markdown("---")
    st.markdown("### 💬 Tu Opinión Importa")
    st.markdown("""
    <div class='gaming-card' style='padding: 1.5rem;'>
        <p style='color: var(--text-light); margin-bottom: 1rem;'>
            Tu feedback nos ayuda a mejorar el sistema y hacerlo más preciso para todos los jugadores.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form(key=f'feedback_form_{session_id}'):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Calificación general
            rating = st.slider(
                "📊 Calificación general",
                min_value=1,
                max_value=5,
                value=3,
                help="1 = Muy malo, 5 = Excelente"
            )
            
            # ¿Predicción correcta?
            prediction_correct = st.radio(
                f"🎯 ¿La predicción '{predicted_style}' fue correcta?",
                options=["Sí", "No", "No estoy seguro"],
                horizontal=True
            )
            
            # Si la predicción no fue correcta, pedir el estilo real
            actual_style = None
            if prediction_correct == "No":
                actual_style = st.selectbox(
                    "¿Cuál es tu estilo real?",
                    options=['Casual', 'Aggressive', 'Explorer', 'Competitive', 'Strategic']
                )
        
        with col2:
            # ¿Recomendaciones útiles?
            recommendations_helpful = st.radio(
                "💡 ¿Las recomendaciones fueron útiles?",
                options=["Sí", "No", "Parcialmente"],
                horizontal=True
            )
            
            # Comentarios adicionales
            comments = st.text_area(
                "📝 Comentarios adicionales (opcional)",
                placeholder="Comparte cualquier sugerencia o comentario...",
                height=100
            )
        
        # Botón de envío
        submitted = st.form_submit_button(
            "📤 Enviar Feedback",
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            # Guardar feedback
            feedback_sys = load_feedback_system()
            
            if feedback_sys:
                feedback_data = {
                    'rating': rating,
                    'prediction_correct': prediction_correct == "Sí",
                    'actual_style': actual_style if prediction_correct == "No" else predicted_style,
                    'recommendations_helpful': recommendations_helpful == "Sí",
                    'comments': comments,
                    'feedback_type': 'post_analysis'
                }
                
                feedback_sys.save_feedback(session_id, feedback_data)
                
                # Guardar perfil para reentrenamiento
                feedback_sys.save_profile_for_training(
                    player_data,
                    predicted_style,
                    actual_style if prediction_correct == "No" else None
                )
                
                st.success("✅ ¡Gracias por tu feedback! Nos ayuda a mejorar el sistema.")
                
                # Mostrar estadísticas de feedback
                summary = feedback_sys.get_feedback_summary(days=30)
                if summary and summary['total_feedback'] > 0:
                    st.info(f"📊 En los últimos 30 días: {summary['total_feedback']} usuarios han compartido feedback. Calificación promedio: {summary['avg_rating']:.1f}/5")
            else:
                st.warning("⚠️ No se pudo guardar el feedback debido a un error del sistema.")

def create_radar_chart(player_data, predicted_style, recommender):
    categories = ['Tiempo', 'Sesiones', 'Dificultad', 'Win Rate', 'Logros', 'PvP']
    # Normalizar a 0-10
    player_values = [
        min(10, player_data['playtime_hours'] / 20),
        min(10, player_data['sessions_per_week']),
        player_data['difficulty_level'],
        player_data['win_rate'] * 10,
        min(10, player_data['achievements_unlocked'] / 10),
        min(10, player_data['pvp_matches'] / 40)
    ]

    ref_stats = getattr(recommender, 'reference_stats', {}).get(predicted_style, {})
    avg_values = [
        min(10, ref_stats.get('playtime_hours', 100) / 20),
        min(10, ref_stats.get('sessions_per_week', 8)),
        ref_stats.get('difficulty_level', 7),
        ref_stats.get('win_rate', 0.6) * 10,
        min(10, ref_stats.get('achievements_unlocked', 60) / 10),
        min(10, ref_stats.get('pvp_matches', 150) / 40)
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=player_values,
        theta=categories,
        fill='toself',
        name='Tu Perfil',
        line=dict(color=GAMING_COLORS['primary'], width=3),
        fillcolor='rgba(0,255,136,0.22)',
        hovertemplate='%{theta}: %{r:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=categories,
        fill='toself',
        name=f'Promedio {predicted_style}',
        line=dict(color=GAMING_COLORS['secondary'], width=2),
        fillcolor='rgba(0,136,255,0.14)',
        hovertemplate='%{theta}: %{r:.2f}<extra></extra>'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=GAMING_COLORS['input_bg'], # Use the slightly lighter background here
            radialaxis=dict(
                visible=True, 
                range=[0, 10], 
                gridcolor='rgba(255,255,255,0.15)', # Lighter, fainter grid
                linecolor='rgba(255,255,255,0.15)',
                color=GAMING_COLORS['light_text'] # Brighter axis text
            ),
            angularaxis=dict(
                color=GAMING_COLORS['text'], 
                gridcolor='rgba(255,255,255,0.15)'
            )
        ),
        paper_bgcolor=GAMING_COLORS['dark_bg'],
        font=dict(color=GAMING_COLORS['text']),
        showlegend=True,
        height=420,
        margin=dict(l=20, r=20, t=60, b=10)
    )

    return fig

def create_probability_chart(probabilities):
    styles = list(probabilities.keys())
    probs = [p * 100 for p in probabilities.values()]

    fig = go.Figure(data=[
        go.Bar(
            x=styles,
            y=probs,
            marker=dict(
                color='rgba(0, 136, 255, 0.7)', # Base color blue/secondary
                line=dict(width=1.5, color='rgba(0, 255, 136, 0.5)') # Primary neon border
            ),
            text=[f'{p:.1f}%' for p in probs],
            textposition='outside',
            textfont=dict(color=GAMING_COLORS['highlight'], size=12, family='Rajdhani, Arial')
        )
    ])

    fig.update_layout(
        title=dict(text="Probabilidad por Estilo", font=dict(color=GAMING_COLORS['primary'], size=15)),
        xaxis=dict(tickfont=dict(color=GAMING_COLORS['text']), gridcolor=GAMING_COLORS['grid']),
        yaxis=dict(title="%", tickfont=dict(color=GAMING_COLORS['text']), gridcolor=GAMING_COLORS['grid']),
        paper_bgcolor=GAMING_COLORS['dark_bg'],
        plot_bgcolor=GAMING_COLORS['card_bg'],
        height=360,
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=False
    )

    return fig

def create_metrics_comparison(player_data, predicted_style, recommender):
    ref_stats = getattr(recommender, 'reference_stats', {}).get(predicted_style, {})

    metrics_data = {
        'Métrica': ['Horas', 'Sesiones/Sem', 'Dificultad', 'Win Rate (%)', 'Logros'],
        'Tu Valor': [
            player_data['playtime_hours'],
            player_data['sessions_per_week'],
            player_data['difficulty_level'],
            player_data['win_rate'] * 100,
            player_data['achievements_unlocked']
        ],
        'Promedio': [
            ref_stats.get('playtime_hours', 100),
            ref_stats.get('sessions_per_week', 8),
            ref_stats.get('difficulty_level', 7),
            ref_stats.get('win_rate', 0.6) * 100,
            ref_stats.get('achievements_unlocked', 60)
        ]
    }

    df = pd.DataFrame(metrics_data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Tu Perfil',
        x=df['Métrica'],
        y=df['Tu Valor'],
        marker_color=GAMING_COLORS['primary'],
        text=df['Tu Valor'].apply(lambda v: f'{v:.0f}' if isinstance(v, (int,float)) else v),
        textposition='auto'
    ))
    fig.add_trace(go.Bar(
        name=f'Promedio {predicted_style}',
        x=df['Métrica'],
        y=df['Promedio'],
        marker_color=GAMING_COLORS['secondary'],
        text=df['Promedio'].apply(lambda v: f'{v:.0f}' if isinstance(v, (int,float)) else v),
        textposition='auto'
    ))

    fig.update_layout(
        title=dict(text="Comparación de Métricas Clave", font=dict(color=GAMING_COLORS['primary'], size=15)),
        barmode='group',
        paper_bgcolor=GAMING_COLORS['dark_bg'],
        plot_bgcolor=GAMING_COLORS['card_bg'],
        height=380,
        margin=dict(l=20, r=20, t=60, b=30)
    )

    return fig

def create_style_distribution_chart(recommender):
    # Si el recommender proporciona distribución real, usarla; si no, usar counts simulados
    if hasattr(recommender, 'style_distribution') and recommender.style_distribution:
        style_counts = recommender.style_distribution
    else:
        style_counts = {style: np.random.randint(1500, 2500) for style in getattr(recommender, 'classes', ['Aggressive','Strategic','Casual','Explorer','Competitive'])}

    fig = go.Figure(data=[
        go.Pie(
            labels=list(style_counts.keys()),
            values=list(style_counts.values()),
            marker=dict(colors=PLAYSTYLE_PALETTE),
            textinfo='percent+label',
            hole=0.32
        )
    ])

    fig.update_layout(
        title=dict(text="Distribución de Estilos en el Dataset", font=dict(color=GAMING_COLORS['primary'], size=15)),
        paper_bgcolor=GAMING_COLORS['dark_bg'],
        plot_bgcolor=GAMING_COLORS['card_bg'],
        height=460,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

# ---------------------------------------------------------
# Funciones para manejo de perfiles
# ---------------------------------------------------------
def get_default_player_data():
    """Return default player data structure"""
    return {
        'playtime_hours': 28.5,
        'sessions_per_week': 3.8,
        'avg_session_length': 1.4,
        'achievements_unlocked': 16.0,
        'difficulty_level': 3.0,
        'combat_style': 'Melee',
        'win_rate': 0.38,
        'pvp_matches': 18.0,
        'death_count': 74.0,
        'last_login_days_ago': 7,
        'premium_user': 0
    }

def validate_profile_data(data):
    """Validate uploaded profile data"""
    required_fields = [
        'playtime_hours', 'sessions_per_week', 'avg_session_length',
        'achievements_unlocked', 'difficulty_level', 'combat_style',
        'win_rate', 'pvp_matches', 'death_count', 'last_login_days_ago',
        'premium_user'
    ]
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate combat_style is one of expected values
    valid_combat_styles = ['Melee', 'Ranged', 'Magic', 'Hybrid', 'Stealth']
    if data['combat_style'] not in valid_combat_styles:
        return False, f"Invalid combat_style. Must be one of: {', '.join(valid_combat_styles)}"
    
    return True, "Valid profile data"

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    st.markdown("""
    <div style='display:flex; justify-content:center;'>
        <div style='display:flex; gap:1rem; align-items:center; max-width:800px; width:100%;'>
            <div style='flex:1; text-align:center;'>
                <h1 style='margin:0;'>🎮 Sistema Inteligente de Recomendación para optimización de experiencias de juego</h1>
                <p style='color:#BFC8FF; margin-top:6px;'>Mejorando tu experiencia de juego con IA — análisis, estilo y recomendaciones.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Cargar recommender con manejo de errores
    try:
        with st.spinner("🔄 Cargando sistema de recomendaciones..."):
            recommender = load_recommender()
    except Exception as e:
        st.error("⚠️ No se pudo cargar el sistema de recomendaciones.")
        st.exception(e)
        st.stop()

    # Sidebar con resumen del sistema
    with st.sidebar:
        st.markdown("<div style='padding:0.6rem 0.4rem;'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin:6px 0 10px 0; color:{GAMING_COLORS['primary']}'>🎯 Sistema Inteligente</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='gaming-card'>
            <p style='margin:0.15rem 0;'><strong>Modelo:</strong> {recommender.metadata.get('model_name', 'N/A')}</p>
            <p style='margin:0.15rem 0;'><strong>Precisión (test):</strong> {recommender.metadata.get('test_accuracy', 0):.2%}</p>
            <p style='margin:0.15rem 0;'><strong>Features:</strong> {len(getattr(recommender, 'feature_names', []))}</p>
            <p style='margin:0.15rem 0;'><strong>Estilos:</strong> {len(getattr(recommender, 'classes', []))}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div class='gaming-card'><h4 style='margin-top:0;color:var(--text-light)'>🎯 Cómo usar</h4><ol style='padding-left:1rem;margin:0.2rem 0 0 0;'><li>Completa tu perfil</li><li>Analiza</li><li>Revisa recomendaciones</li></ol></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<div style='text-align:center; font-size:12px; color:#AFC3FF; margin-top:0.6rem;'>UTP 2025 · Sistema Inteligente</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Perfil", "📈 Estadísticas", "❓ Ayuda", "⚙️ Admin"])

    # ---------- TAB 1: Perfil ----------
    with tab1:
        st.markdown("<h2 style='margin:0;'>📊 Perfil del Jugador</h2><p style='margin:6px 0 1rem 0; color:#BFC8FF'>Introduce tus datos para generar recomendaciones personalizadas.</p>", unsafe_allow_html=True)
        
        st.markdown("---")

        # Initialize session state for profile input method
        if 'profile_input_method' not in st.session_state:
            st.session_state.profile_input_method = None

        # Profile input method selection
        st.markdown("### 🎯 Selecciona cómo ingresar tu perfil")
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            if st.button("📝 Ingresar Perfil Manualmente", 
                        use_container_width=True, 
                        key="manual_btn"):
                st.session_state.profile_input_method = "manual"
        
        with col2:
            if st.button("📁 Cargar Perfil desde JSON", 
                        use_container_width=True, 
                        key="upload_btn"):
                st.session_state.profile_input_method = "upload"

        st.markdown("---")

        player_data = None

        # Manual input section
        if st.session_state.profile_input_method == "manual":
            st.markdown("### ⚙️ Configuración de perfil manual")
            
            # Obtener valores por defecto
            default_data = get_default_player_data()
            
            col1, col2, col3 = st.columns([1,1,1], gap="large")
            with col1:
                playtime = st.slider("Horas totales jugadas", 0, 500, int(default_data['playtime_hours']), 5, help="Total aproximado de horas jugadas")
                sessions = st.slider("Sesiones por semana", 1, 20, int(default_data['sessions_per_week']), 1, help="Cuántas sesiones sueles tener por semana")
                avg_length = st.slider("Duración promedio (horas)", 0.5, 8.0, float(default_data['avg_session_length']), 0.5, help="Duración media de una sesión")
            with col2:
                difficulty = st.slider("Nivel de dificultad (1-10)", 1, 10, int(default_data['difficulty_level']), 1, help="Dificultad en la que sueles jugar")
                win_rate = st.slider("Tasa de victoria (%)", 0, 100, int(default_data['win_rate'] * 100), 1, help="Porcentaje de victorias") / 100.0
                achievements = st.slider("Logros desbloqueados", 0, 200, int(default_data['achievements_unlocked']), 1, help="Cantidad de logros que completaste")
            with col3:
                combat_style = st.selectbox("Estilo de combate favorito", ['Melee','Ranged','Magic','Hybrid','Stealth'], 
                                        index=['Melee','Ranged','Magic','Hybrid','Stealth'].index(default_data['combat_style']), 
                                        help="Tipo de combate que prefieres")
                pvp_matches = st.slider("Partidas PvP", 0, 2000, int(default_data['pvp_matches']), 10, help="Número de partidas PvP")
                death_count = st.slider("Muertes totales", 0, 2000, int(default_data['death_count']), 10, help="Veces que has muerto en total")

            col4, col5 = st.columns([1,1], gap="large")
            with col4:
                last_login = st.slider("Días desde último login", 0, 365, int(default_data['last_login_days_ago']), 1)
            with col5:
                premium = st.checkbox("Usuario Premium", value=bool(default_data['premium_user']))

            player_data = {
                'playtime_hours': playtime,
                'sessions_per_week': sessions,
                'avg_session_length': avg_length,
                'achievements_unlocked': achievements,
                'difficulty_level': difficulty,
                'combat_style': combat_style,
                'win_rate': win_rate,
                'pvp_matches': pvp_matches,
                'death_count': death_count,
                'last_login_days_ago': last_login,
                'premium_user': 1 if premium else 0
            }

        # Upload section
        elif st.session_state.profile_input_method == "upload":
            st.markdown("### 📁 Cargar perfil desde archivo JSON")
            
            uploaded_file = st.file_uploader("Selecciona un archivo JSON con tu perfil", type=['json'])
            
            if uploaded_file is not None:
                try:
                    # Read and parse the JSON file
                    profile_data = json.load(uploaded_file)
                    
                    # Validate the profile data
                    is_valid, message = validate_profile_data(profile_data)
                    
                    if is_valid:
                        st.success("✅ Perfil cargado correctamente")
                        player_data = profile_data
                        
                        # Display the loaded data
                        with st.expander("📋 Ver datos cargados", expanded=True):
                            st.json(profile_data)
                    else:
                        st.error(f"❌ Error en el archivo: {message}")
                        st.info("""
                        **Formato esperado:**
                        ```json
                        {
                          "playtime_hours": 35,
                          "sessions_per_week": 3,
                          "avg_session_length": 1.2,
                          "achievements_unlocked": 18,
                          "difficulty_level": 3,
                          "combat_style": "Melee",
                          "win_rate": 0.38,
                          "pvp_matches": 15,
                          "death_count": 65,
                          "last_login_days_ago": 5,
                          "premium_user": 0
                        }
                        ```
                        """)
                        
                except json.JSONDecodeError:
                    st.error("❌ Error: El archivo no es un JSON válido")
                except Exception as e:
                    st.error(f"❌ Error al procesar el archivo: {str(e)}")
            else:
                st.info("""
                **📋 Formato esperado del archivo JSON:**
                ```json
                {
                  "playtime_hours": 35,
                  "sessions_per_week": 3,
                  "avg_session_length": 1.2,
                  "achievements_unlocked": 18,
                  "difficulty_level": 3,
                  "combat_style": "Melee",
                  "win_rate": 0.38,
                  "pvp_matches": 15,
                  "death_count": 65,
                  "last_login_days_ago": 5,
                  "premium_user": 0
                }
                ```
                """)

        # Analysis button (only show if we have player data)
        if player_data is not None:
            st.markdown("---")
            center_col = st.columns([1, 2, 1])[1]  # Get the middle column
            with center_col:
                analyze_btn = st.button("🔍 ANALIZAR Y GENERAR RECOMENDACIONES", type="primary", use_container_width=True)

            if analyze_btn:
                with st.spinner("🤖 Analizando tu perfil con IA..."):
                    # Asumimos que recommender.analyze_player devuelve diccionario con keys: prediction, metrics, recommendations
                    try:
                        result = recommender.analyze_player(player_data)
                    except Exception as e:
                        st.error("Error durante el análisis del perfil.")
                        st.exception(e)
                        result = None

                if result:
                    st.success("✅ Análisis completado")
                    # Header de resultados
                    st.markdown(f"<div class='gaming-card' style='margin-top:8px;'><h2 style='margin:0;color:var(--primary)'>🎯 Resultados del Análisis</h2></div>", unsafe_allow_html=True)

                    # Métricas principales
                    c1, c2, c3, c4 = st.columns([1,1,1,1], gap="large")
                    predicted_style = result['prediction'].get('predicted_style', 'N/A')
                    confidence = result['prediction'].get('confidence', 0.0)
                    engagement = result['metrics'].get('engagement_score', 0.0)
                    skill = result['metrics'].get('skill_level', 0.0)

                    with c1:
                        st.markdown(f"<div class='metric-card'><h3 style='margin:0;'>Estilo Predicho</h3><h2>{predicted_style}</h2><div class='sub-text'>Tu etiqueta de juego</div></div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<div class='metric-card'><h3 style='color:var(--text-light); margin:0;'>Confianza</h3><h2>{confidence:.1%}</h2><div style='font-size:12px;color:#AFC3FF'>Precisión del modelo</div></div>", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"<div class='metric-card'><h3 style='color:var(--text-light); margin:0;'>Engagement</h3><h2>{engagement:.1f}</h2><div style='font-size:12px;color:#AFC3FF'>Nivel de compromiso</div></div>", unsafe_allow_html=True)
                    with c4:
                        st.markdown(f"<div class='metric-card'><h3 style='color:var(--text-light); margin:0;'>Habilidad</h3><h2>{skill:.1f}</h2><div style='font-size:12px;color:#AFC3FF'>Nivel estimado</div></div>", unsafe_allow_html=True)

                    st.markdown("---")

                    # Visualizaciones
                    viz_left, viz_right = st.columns([1,1], gap="large")

                    with viz_left:
                        radar_fig = create_radar_chart(player_data, predicted_style, recommender)
                        st.plotly_chart(radar_fig, use_container_width=True, config={'displayModeBar': False})

                    with viz_right:
                        prob_fig = create_probability_chart(result['prediction'].get('probabilities', {predicted_style: 1.0}))
                        st.plotly_chart(prob_fig, use_container_width=True, config={'displayModeBar': False})

                    # Comparación de métricas (full-width)
                    metrics_fig = create_metrics_comparison(player_data, predicted_style, recommender)
                    st.plotly_chart(metrics_fig, use_container_width=True, config={'displayModeBar': False})

                    st.markdown("---")

                    # Recomendaciones - se muestran como expanders con prioridad marcada
                    st.markdown("<h3 style='color:var(--primary); margin-bottom:6px;'>💡 Recomendaciones Personalizadas</h3>", unsafe_allow_html=True)

                    priority_colors = {'Alta': GAMING_COLORS['accent'], 'Media': GAMING_COLORS['secondary'], 'Baja': GAMING_COLORS['primary']}

                    for i, rec in enumerate(result.get('recommendations', []), 1):
                        pr = rec.get('priority', 'Media')
                        color = priority_colors.get(pr, GAMING_COLORS['primary'])
                        icon = "🔥" if pr == 'Alta' else "✨" if pr == 'Media' else "⭐"
                        with st.expander(f"Recomendación {i}", expanded=(i <= 2)):
                            st.markdown(f"{icon} **{rec.get('title','Sin título')}** — <span style='color:{color};'>Prioridad: {pr}</span>", unsafe_allow_html=True)
                            st.markdown(f"**📝 Justificación:**")
                            st.markdown(f"<div style='background:{GAMING_COLORS['medium_bg']}; padding:10px; border-radius:6px; border-left:4px solid {color};'>{rec.get('reason','-')}</div>", unsafe_allow_html=True)
                            st.markdown(f"**💥 Impacto esperado:**")
                            st.markdown(f"<div style='background:{GAMING_COLORS['medium_bg']}; padding:10px; border-radius:6px; border-left:4px solid {GAMING_COLORS['primary']};'>{rec.get('impact','-')}</div>", unsafe_allow_html=True)
                            st.markdown(f"**🎯 Acción recomendada:**")
                            st.markdown(f"<div style='background:{GAMING_COLORS['medium_bg']}; padding:10px; border-radius:6px; border-left:4px solid {GAMING_COLORS['secondary']};'>{rec.get('action','-')}</div>", unsafe_allow_html=True)

                    # Descargar reporte de texto
                    report_lines = [
                        "REPORTE - SISTEMA INTELIGENTE DE RECOMENDACIÓN",
                        "=============================================",
                        "",
                        "== Información del Jugador ==",
                        f"Horas de Juego: {player_data['playtime_hours']}",
                        f"Sesiones/Semana: {player_data['sessions_per_week']}",
                        f"Dificultad: {player_data['difficulty_level']}",
                        f"Win Rate: {player_data['win_rate']:.1%}",
                        f"Estilo combate: {player_data['combat_style']}",
                        "",
                        "== Resultados ==",
                        f"Estilo predicho: {predicted_style}",
                        f"Confianza: {confidence:.1%}",
                        f"Engagement: {engagement:.2f}",
                        f"Skill level: {skill:.2f}",
                        "",
                        "== Recomendaciones =="
                    ]
                    for idx, r in enumerate(result.get('recommendations', []), 1):
                        report_lines += [
                            f"{idx}. {r.get('title','-')}",
                            f"   Prioridad: {r.get('priority','-')}",
                            f"   Razón: {r.get('reason','-')}",
                            f"   Impacto: {r.get('impact','-')}",
                            f"   Acción: {r.get('action','-')}",
                            ""
                        ]
                    report_text = "\n".join(report_lines)

                    st.download_button(
                        label="📥 Descargar reporte",
                        data=report_text,
                        file_name=f"reporte_gaming_{predicted_style}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                    # Guardar predicción y mostrar feedback
                    if analyze_btn and result:
                        feedback_sys = load_feedback_system()
                        if feedback_sys:
                            session_id = feedback_sys.save_prediction(player_data, result['prediction'])
                            
                            # Mostrar sección de feedback
                            show_feedback_section(session_id, predicted_style, player_data)
                        else:
                            st.warning("⚠️ El sistema de feedback no está disponible en este momento.")

                    else:
                        st.error("No se obtuvieron resultados del analizador.")

        # Show instruction if no method selected yet
        elif st.session_state.profile_input_method is None:
            st.markdown("""
            <div class='gaming-card' style='text-align: center; padding: 3rem;'>
                <h3 style='color: var(--primary); margin-bottom: 1rem;'>🎯 Selecciona un método para comenzar</h3>
                <p style='color: var(--text-light); margin-bottom: 2rem;'>
                    Elige cómo quieres ingresar los datos de tu perfil de jugador
                </p>
                <div style='display: flex; gap: 2rem; justify-content: center;'>
                    <div style='flex: 1; max-width: 200px;'>
                        <h4 style='color: var(--secondary);'>📝 Manual</h4>
                        <p style='font-size: 0.9rem; color: #AFC3FF;'>Completa los datos usando los controles interactivos</p>
                    </div>
                    <div style='flex: 1; max-width: 200px;'>
                        <h4 style='color: var(--accent);'>📁 JSON</h4>
                        <p style='font-size: 0.9rem; color: #AFC3FF;'>Carga un archivo JSON con tu perfil predefinido</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---------- TAB 2: Estadísticas del sistema ----------
    with tab2:
        st.markdown("<div class='gaming-card'><h2 style='margin:0;'>📈 Estadísticas del Sistema</h2><p style='margin:6px 0 0 0; color:#BFC8FF'>Métricas y distribución del dataset / modelo.</p></div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1,1], gap="large")
        with col1:
            st.markdown(f"<div class='gaming-card'><h4 style='margin:0 0 8px 0;'>Especificaciones Técnicas</h4><p style='margin:0.1rem 0;'><strong>Modelo:</strong> {recommender.metadata.get('model_name','N/A')}</p><p style='margin:0.1rem 0;'><strong>Precisión:</strong> {recommender.metadata.get('test_accuracy',0):.2%}</p><p style='margin:0.1rem 0;'><strong>Precision:</strong> {recommender.metadata.get('precision',0):.2%}</p><p style='margin:0.1rem 0;'><strong>Recall:</strong> {recommender.metadata.get('recall',0):.2%}</p></div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div class='gaming-card'><h4 style='margin:0 0 8px 0;'>Configuración</h4><p style='margin:0.1rem 0;'><strong>Features usados:</strong> {len(getattr(recommender,'feature_names',[]))}</p><p style='margin:0.1rem 0;'><strong>Clases detectables:</strong> {len(getattr(recommender,'classes',[]))}</p><p style='margin:0.1rem 0;'><strong>Fecha entrenamiento:</strong> {recommender.metadata.get('timestamp','N/A')}</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Distribución de estilos")
        dist_fig = create_style_distribution_chart(recommender)
        st.plotly_chart(dist_fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("---")

        col3, col4, col5 = st.columns([1,1,1], gap="large")
        with col3:
            st.markdown(f"<div class='metric-card'><h3 style='color:var(--text-light); margin:0;'>Precisión</h3><h2>{recommender.metadata.get('test_accuracy',0):.1%}</h2><div style='font-size:12px;color:#AFC3FF'>Exactitud global</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><h3 style='color:var(--text-light); margin:0;'>Precision</h3><h2>{recommender.metadata.get('precision',0):.1%}</h2><div style='font-size:12px;color:#AFC3FF'>Por clase</div></div>", unsafe_allow_html=True)
        with col5:
            st.markdown(f"<div class='metric-card'><h3 style='color:var(--text-light); margin:0;'>Recall</h3><h2>{recommender.metadata.get('recall',0):.1%}</h2><div style='font-size:12px;color:#AFC3FF'>Sensibilidad</div></div>", unsafe_allow_html=True)

    # ---------- TAB 3: Ayuda ----------
    with tab3:
        st.markdown("<div class='gaming-card'><h2 style='margin:0;'>❓ Ayuda y Documentación</h2><p style='margin:6px 0 0 0; color:#BFC8FF'>Guía rápida sobre el sistema.</p></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='gaming-card'>
            <h3 style='margin-top:0;color:var(--text-light)'>¿Qué es este sistema?</h3>
            <p>Un sistema que analiza tu comportamiento como jugador y genera recomendaciones personalizadas basadas en patrones detectados por modelos de machine learning.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='gaming-card'>
            <h3 style='margin-top:0;color:var(--text-light)'>Métricas clave</h3>
            <ul>
                <li><strong>Tiempo de juego:</strong> Horas totales jugadas.</li>
                <li><strong>Sesiones:</strong> Frecuencia semanal.</li>
                <li><strong>Dificultad:</strong> Nivel jugado (1-10).</li>
                <li><strong>Win Rate:</strong> Porcentaje de victorias.</li>
                <li><strong>Engagement:</strong> Indicador de compromiso.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='gaming-card'>
            <h3 style='margin-top:0;color:var(--text-light)'>Autores</h3>
            <div style='display:flex; gap:1rem;'>
                <div style='flex:1; text-align:center;'>
                    <h4 style='color:var(--primary); margin:0;'>Laura Rivera</h4><div style='font-size:12px;color:#AFC3FF'>8-969-1184</div>
                </div>
                <div style='flex:1; text-align:center;'>
                    <h4 style='color:var(--secondary); margin:0;'>Marco Rodríguez</h4><div style='font-size:12px;color:#AFC3FF'>8-956-932</div>
                </div>
                <div style='flex:1; text-align:center;'>
                    <h4 style='color:var(--accent); margin:0;'>David Tao</h4><div style='font-size:12px;color:#AFC3FF'>8-961-1083</div>
                </div>
            </div>
            <p style='text-align:center; margin-top:8px; color:#AFC3FF'><strong>Curso:</strong> Sistemas Inteligentes - UTP 2025</p>
        </div>
        """, unsafe_allow_html=True)
    
        # ---------- TAB 4: Panel de Administración ----------
    with tab4:
        st.markdown("<div class='gaming-card'><h2 style='margin:0;'>⚙️ Panel de Administración</h2><p style='margin:6px 0 0 0; color:#BFC8FF'>Monitoreo del sistema y feedback de usuarios.</p></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Verificar contraseña simple (en producción usar autenticación real)
        password = st.text_input("🔐 Contraseña de administrador", type="password")
        
        if password == "admin123":  # CAMBIAR EN PRODUCCIÓN
            st.success("✅ Acceso autorizado")
            
            feedback_sys = load_feedback_system()
            monitor = load_monitoring()
            
            if feedback_sys is None or monitor is None:
                st.error("⚠️ No se pudo cargar los sistemas de feedback/monitoreo.")
            else:
                # Métricas de monitoreo
                st.markdown("#### 📊 Métricas del Sistema")
                
                metrics = monitor.calculate_drift_metrics(window_days=30)
                
                if metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Predicciones Totales",
                            metrics['total_predictions']
                        )
                    
                    with col2:
                        st.metric(
                            "Tasa de Feedback",
                            f"{metrics['feedback_rate']:.1%}"
                        )
                    
                    with col3:
                        accuracy_val = metrics['accuracy'] if metrics['accuracy'] else 0
                        st.metric(
                            "Accuracy Actual",
                            f"{accuracy_val:.1%}"
                        )
                    
                    with col4:
                        st.metric(
                            "Confianza Promedio",
                            f"{metrics['avg_confidence']:.1%}"
                        )
                
                st.markdown("---")
                
                # Verificar necesidad de reentrenamiento
                st.markdown("#### 🔄 Estado del Modelo")
                
                needs_retrain, reason, current_metrics = monitor.check_retraining_needed()
                
                if needs_retrain:
                    st.warning(f"⚠️ **Se recomienda reentrenar el modelo**")
                    st.write(f"**Razón:** {reason}")
                else:
                    st.success("✅ **Modelo funcionando correctamente**")
                    st.write(f"**Estado:** {reason}")
                
                st.markdown("---")
                
                # Resumen de feedback
                st.markdown("#### 💬 Resumen de Feedback (30 días)")
                
                summary = feedback_sys.get_feedback_summary(days=30)
                
                if summary and summary['total_feedback'] > 0:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Total Feedback",
                            summary['total_feedback']
                        )
                    
                    with col2:
                        avg_rating = summary['avg_rating'] if summary['avg_rating'] else 0
                        st.metric(
                            "Calificación Promedio",
                            f"{avg_rating:.1f}/5"
                        )
                    
                    with col3:
                        pred_acc = summary['prediction_accuracy'] if summary['prediction_accuracy'] else 0
                        st.metric(
                            "Accuracy Reportada",
                            f"{pred_acc:.1%}"
                        )
                else:
                    st.info("No hay feedback reciente")
                
                st.markdown("---")
                
                # Exportar datos para reentrenamiento
                st.markdown("#### 📥 Exportar Datos")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Exportar Datos de Reentrenamiento", use_container_width=True):
                        if feedback_sys.export_training_data():
                            st.success("✅ Datos exportados a data/retraining_data.csv")
                        else:
                            st.warning("⚠️ No hay suficientes datos confirmados")
                
                with col2:
                    # Generar reporte de monitoreo
                    if st.button("📄 Generar Reporte Completo", use_container_width=True):
                        report = monitor.generate_monitoring_report()
                        
                        report_json = json.dumps(report, indent=2, default=str)
                        st.download_button(
                            label="💾 Descargar Reporte JSON",
                            data=report_json,
                            file_name=f"monitoring_report_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                
                # Mostrar feedback reciente
                st.markdown("---")
                st.markdown("#### 📝 Feedback Reciente")
                
                with st.expander("Ver últimos 10 feedbacks", expanded=False):
                    feedbacks = feedback_sys._read_jsonl(feedback_sys.feedback_file)
                    
                    if feedbacks:
                        recent = feedbacks[-10:][::-1]  # Últimos 10, orden inverso
                        
                        for i, fb in enumerate(recent, 1):
                            st.markdown(f"""
                            **Feedback #{i}**
                            - Fecha: {fb.get('timestamp', 'N/A')}
                            - Calificación: {fb.get('rating', 'N/A')}/5
                            - Predicción correcta: {fb.get('prediction_correct', 'N/A')}
                            - Comentario: {fb.get('comments', 'Sin comentarios')}
                            ---
                            """)
                    else:
                        st.info("No hay feedback registrado")
        
        elif password:
            st.error("❌ Contraseña incorrecta")
        else:
            st.info("🔐 Ingresa la contraseña de administrador para acceder al panel")

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align:center; padding:1rem 0; color:#AFC3FF'>
        <div style='font-weight:600; color:{GAMING_COLORS['text']};'>Sistema Inteligente de Recomendación para Videojuegos</div>
        <div style='font-size:12px;'>UTP 2025 · Desarrollado por: Laura Rivera, Marco Rodríguez, David Tao</div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
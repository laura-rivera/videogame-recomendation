"""
Aplicación Web - Sistema Inteligente de Recomendación para Videojuegos
Interfaz de Usuario con Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Añadir path para imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from recommender_system import GameRecommender
except:
    st.error("⚠️ No se pudo cargar el sistema de recomendaciones. Asegúrate de haber entrenado el modelo primero.")
    st.stop()

# Configuración de la página
st.set_page_config(
    page_title="Sistema Inteligente de Recomendación",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .recommendation-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ff4b4b;
    }
    h1 {
        color: #ff4b4b;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    """Carga el sistema de recomendaciones (con caché)"""
    try:
        return GameRecommender()
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

def create_radar_chart(player_data, predicted_style, recommender):
    """Crea un gráfico de radar comparando el jugador con el promedio"""
    
    categories = ['Tiempo de Juego', 'Sesiones/Semana', 'Dificultad', 
                  'Win Rate', 'Logros', 'PvP']
    
    # Normalizar valores a escala 0-10
    player_values = [
        min(10, player_data['playtime_hours'] / 20),
        min(10, player_data['sessions_per_week']),
        player_data['difficulty_level'],
        player_data['win_rate'] * 10,
        min(10, player_data['achievements_unlocked'] / 10),
        min(10, player_data['pvp_matches'] / 40)
    ]
    
    # Valores promedio para el estilo predicho
    ref_stats = recommender.reference_stats.get(predicted_style, {})
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
        line=dict(color='#ff4b4b', width=2)
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=categories,
        fill='toself',
        name=f'Promedio {predicted_style}',
        line=dict(color='#1f77b4', width=2),
        opacity=0.6
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title="Comparación con Jugadores de tu Estilo",
        height=450
    )
    
    return fig

def create_probability_chart(probabilities):
    """Crea gráfico de probabilidades por estilo"""
    
    styles = list(probabilities.keys())
    probs = list(probabilities.values())
    
    colors = ['#ff4b4b' if p == max(probs) else '#1f77b4' for p in probs]
    
    fig = go.Figure(data=[
        go.Bar(
            x=styles,
            y=[p * 100 for p in probs],
            marker_color=colors,
            text=[f'{p:.1%}' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Probabilidad por Estilo de Juego",
        xaxis_title="Estilo",
        yaxis_title="Probabilidad (%)",
        height=400,
        showlegend=False
    )
    
    return fig

def create_metrics_comparison(player_data, predicted_style, recommender):
    """Crea comparación de métricas"""
    
    ref_stats = recommender.reference_stats.get(predicted_style, {})
    
    metrics_data = {
        'Métrica': ['Horas', 'Sesiones/Sem', 'Dificultad', 'Win Rate', 'Logros'],
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
        marker_color='#ff4b4b'
    ))
    
    fig.add_trace(go.Bar(
        name=f'Promedio {predicted_style}',
        x=df['Métrica'],
        y=df['Promedio'],
        marker_color='#1f77b4'
    ))
    
    fig.update_layout(
        title="Comparación de Métricas Clave",
        barmode='group',
        height=400,
        xaxis_title="",
        yaxis_title="Valor"
    )
    
    return fig

def main():
    """Función principal de la aplicación"""
    
    # Header
    st.title("🎮 Sistema Inteligente de Recomendación para Videojuegos")
    st.markdown("### Optimiza tu experiencia de juego con Inteligencia Artificial")
    st.markdown("---")
    
    # Cargar sistema
    with st.spinner("Cargando sistema de IA..."):
        recommender = load_recommender()
    
    if recommender is None:
        st.error("❌ No se pudo cargar el sistema. Verifica que los modelos estén entrenados.")
        st.info("💡 Ejecuta primero: `python train_model.py`")
        st.stop()
    
    # Sidebar - Información del sistema
    with st.sidebar:
        st.header("ℹ️ Información del Sistema")
        st.markdown(f"""
        **Modelo:** {recommender.metadata['model_name']}
        
        **Precisión:** {recommender.metadata['test_accuracy']:.2%}
        
        **Características:** {len(recommender.feature_names)}
        
        **Estilos detectables:** {len(recommender.classes)}
        """)
        
        st.markdown("---")
        st.markdown("### 📚 ¿Cómo funciona?")
        st.markdown("""
        1. Ingresa tu perfil de jugador
        2. El sistema analiza tus datos
        3. Predice tu estilo de juego
        4. Genera recomendaciones personalizadas
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Estilos de Juego")
        st.markdown("""
        - **Aggressive**: Combate directo
        - **Strategic**: Planificación táctica
        - **Casual**: Diversión relajada
        - **Explorer**: Descubrimiento
        - **Competitive**: Alto rendimiento
        """)
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["📊 Análisis de Perfil", "📈 Estadísticas", "❓ Ayuda"])
    
    with tab1:
        st.header("📊 Perfil del Jugador")
        
        # Crear columnas para inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("⏱️ Tiempo de Juego")
            playtime = st.slider(
                "Horas totales jugadas",
                min_value=0, max_value=500, value=50, step=5,
                help="Total de horas que has jugado"
            )
            
            sessions = st.slider(
                "Sesiones por semana",
                min_value=1, max_value=20, value=5, step=1,
                help="Cuántas veces juegas por semana"
            )
            
            avg_length = st.slider(
                "Duración promedio (horas)",
                min_value=0.5, max_value=8.0, value=2.0, step=0.5,
                help="Cuánto dura cada sesión en promedio"
            )
        
        with col2:
            st.subheader("🎯 Rendimiento")
            difficulty = st.slider(
                "Nivel de dificultad",
                min_value=1, max_value=10, value=5, step=1,
                help="Dificultad en la que juegas (1=Muy Fácil, 10=Extremo)"
            )
            
            win_rate = st.slider(
                "Tasa de victoria (%)",
                min_value=0, max_value=100, value=50, step=5,
                help="Porcentaje de partidas que ganas"
            ) / 100
            
            achievements = st.slider(
                "Logros desbloqueados",
                min_value=0, max_value=200, value=25, step=5,
                help="Cantidad de logros que has completado"
            )
        
        with col3:
            st.subheader("⚔️ Combate")
            combat_style = st.selectbox(
                "Estilo de combate favorito",
                options=['Melee', 'Ranged', 'Magic', 'Hybrid', 'Stealth'],
                help="Tu estilo de combate preferido"
            )
            
            pvp_matches = st.slider(
                "Partidas PvP",
                min_value=0, max_value=1000, value=50, step=10,
                help="Partidas jugador vs jugador"
            )
            
            death_count = st.slider(
                "Muertes totales",
                min_value=0, max_value=1000, value=150, step=10,
                help="Veces que has muerto en el juego"
            )
        
        # Opciones adicionales
        col4, col5 = st.columns(2)
        with col4:
            last_login = st.slider(
                "Días desde último login",
                min_value=0, max_value=30, value=2, step=1
            )
        with col5:
            premium = st.checkbox("Usuario Premium", value=False)
        
        st.markdown("---")
        
        # Botón de análisis
        if st.button("🔍 Analizar Perfil y Generar Recomendaciones", type="primary", use_container_width=True):
            
            # Preparar datos
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
            
            # Realizar análisis
            with st.spinner("🤖 Analizando tu perfil con IA..."):
                result = recommender.analyze_player(player_data)
            
            st.success("✅ ¡Análisis completado!")
            
            # Mostrar resultados
            st.markdown("## 🎯 Resultados del Análisis")
            
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Estilo Predicho",
                    result['prediction']['predicted_style'],
                    help="Tu estilo de juego identificado por la IA"
                )
            
            with col2:
                confidence = result['prediction']['confidence']
                st.metric(
                    "Confianza",
                    f"{confidence:.1%}",
                    help="Qué tan seguro está el modelo"
                )
            
            with col3:
                st.metric(
                    "Score de Engagement",
                    f"{result['metrics']['engagement_score']:.1f}",
                    help="Tu nivel de compromiso con el juego"
                )
            
            with col4:
                st.metric(
                    "Nivel de Habilidad",
                    f"{result['metrics']['skill_level']:.1f}",
                    help="Tu nivel de habilidad estimado (0-100)"
                )
            
            st.markdown("---")
            
            # Visualizaciones
            st.markdown("### 📊 Análisis Visual")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Gráfico de radar
                radar_fig = create_radar_chart(
                    player_data,
                    result['prediction']['predicted_style'],
                    recommender
                )
                st.plotly_chart(radar_fig, use_container_width=True)
            
            with viz_col2:
                # Gráfico de probabilidades
                prob_fig = create_probability_chart(
                    result['prediction']['probabilities']
                )
                st.plotly_chart(prob_fig, use_container_width=True)
            
            # Comparación de métricas
            metrics_fig = create_metrics_comparison(
                player_data,
                result['prediction']['predicted_style'],
                recommender
            )
            st.plotly_chart(metrics_fig, use_container_width=True)
            
            st.markdown("---")
            
            # Recomendaciones
            st.markdown("## 💡 Recomendaciones Personalizadas")
            st.markdown("Basadas en análisis de datos y patrones de jugadores similares")
            
            for i, rec in enumerate(result['recommendations'], 1):
                
                # Emoji según prioridad
                priority_emoji = {
                    'Alta': '🔴',
                    'Media': '🟡',
                    'Baja': '🟢'
                }
                
                with st.expander(
                    f"{priority_emoji.get(rec['priority'], '⚪')} Recomendación {i}: {rec['title']}", 
                    expanded=(i <= 3)
                ):
                    st.markdown(f"**📝 Justificación:**")
                    st.info(rec['reason'])
                    
                    st.markdown(f"**💥 Impacto Esperado:**")
                    st.success(rec['impact'])
                    
                    st.markdown(f"**🎯 Acción Recomendada:**")
                    st.warning(rec['action'])
                    
                    st.markdown(f"**⚡ Prioridad:** {rec['priority']}")
            
            # Botón de descarga
            st.markdown("---")
            
            # Crear reporte en texto
            report_text = f"""
REPORTE DE ANÁLISIS - SISTEMA INTELIGENTE DE RECOMENDACIÓN
============================================================

INFORMACIÓN DEL JUGADOR
-----------------------
Horas de Juego: {playtime}
Sesiones por Semana: {sessions}
Dificultad: {difficulty}
Win Rate: {win_rate:.1%}
Estilo de Combate: {combat_style}

RESULTADOS DEL ANÁLISIS
-----------------------
Estilo Predicho: {result['prediction']['predicted_style']}
Confianza: {result['prediction']['confidence']:.1%}
Engagement Score: {result['metrics']['engagement_score']:.2f}
Nivel de Habilidad: {result['metrics']['skill_level']:.2f}

RECOMENDACIONES
--------------
"""
            for i, rec in enumerate(result['recommendations'], 1):
                report_text += f"\n{i}. {rec['title']}\n"
                report_text += f"   Prioridad: {rec['priority']}\n"
                report_text += f"   Razón: {rec['reason']}\n"
                report_text += f"   Impacto: {rec['impact']}\n"
                report_text += f"   Acción: {rec['action']}\n"
            
            st.download_button(
                label="📥 Descargar Reporte Completo",
                data=report_text,
                file_name=f"reporte_gaming_{result['prediction']['predicted_style']}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    with tab2:
        st.header("📈 Estadísticas del Sistema")
        
        st.markdown("### 🎯 Información del Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Tipo de Modelo:** {recommender.metadata['model_name']}
            
            **Precisión en Test:** {recommender.metadata['test_accuracy']:.2%}
            
            **Precision:** {recommender.metadata['precision']:.2%}
            
            **Recall:** {recommender.metadata['recall']:.2%}
            
            **F1-Score:** {recommender.metadata['f1_score']:.2%}
            """)
        
        with col2:
            st.markdown(f"""
            **Características Utilizadas:** {len(recommender.feature_names)}
            
            **Clases Detectables:** {len(recommender.classes)}
            
            **Fecha de Entrenamiento:** {recommender.metadata.get('timestamp', 'N/A')}
            
            **Dataset:** 10,000 jugadores
            """)
        
        st.markdown("---")
        
        st.markdown("### 📊 Distribución de Estilos (Dataset de Entrenamiento)")
        
        # Crear gráfico de distribución
        style_counts = {style: 2000 for style in recommender.classes}  # Datos de ejemplo
        
        fig = px.pie(
            values=list(style_counts.values()),
            names=list(style_counts.keys()),
            title="Distribución de Estilos en el Dataset"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("❓ Ayuda y Documentación")
        
        st.markdown("""
        ### 🎮 ¿Qué es este sistema?
        
        Este es un **Sistema Inteligente de Recomendación** diseñado para analizar
        tu comportamiento como jugador y proporcionarte sugerencias personalizadas
        para mejorar tu experiencia de juego.
        
        ### 🤖 ¿Cómo funciona?
        
        1. **Recopilación de Datos**: Ingresas información sobre tu comportamiento en el juego
        2. **Análisis con IA**: El sistema usa Machine Learning para identificar patrones
        3. **Predicción**: Clasifica tu estilo de juego usando un modelo entrenado
        4. **Recomendaciones**: Genera sugerencias basadas en miles de perfiles similares
        
        ### 📊 Métricas Clave
        
        - **Tiempo de Juego**: Total de horas invertidas
        - **Sesiones**: Frecuencia con la que juegas
        - **Dificultad**: Nivel de desafío que prefieres
        - **Win Rate**: Porcentaje de victorias
        - **Engagement**: Nivel de compromiso calculado
        - **Skill Level**: Habilidad estimada (0-100)
        
        ### 🎯 Estilos de Juego
        
        **Aggressive (Agresivo)**
        - Prefiere combate directo
        - Alta actividad PvP
        - Dificultad elevada
        
        **Strategic (Estratégico)**
        - Planificación cuidadosa
        - Alto win rate
        - Sesiones largas
        
        **Casual (Casual)**
        - Juego relajado
        - Sesiones cortas
        - Dificultad baja-media
        
        **Explorer (Explorador)**
        - Descubrimiento de contenido
        - Muchos logros
        - Tiempo de juego alto
        
        **Competitive (Competitivo)**
        - Máximo rendimiento
        - Muchas partidas PvP
        - Dificultad máxima
        
        ### 🔒 Privacidad
        
        Todos los datos se procesan localmente. No se almacena información personal.
        
        ### 📧 Soporte
        
        **Autores:**
        - Laura Rivera (8-969-1184)
        - Marco Rodríguez (8-956-932)
        - David Tao (8-961-1083)
        
        **Curso:** Sistemas Inteligentes - UTP
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Sistema Inteligente de Recomendación para Videojuegos | UTP 2025</p>
        <p>Desarrollado por: Laura Rivera, Marco Rodríguez, David Tao</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
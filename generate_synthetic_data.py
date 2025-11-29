"""
Sistema Inteligente de Recomendación para Videojuegos
Generador de Datos Sintéticos
Autores: Laura Rivera, Marco Rodríguez, David Tao
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configurar semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

def generate_gaming_dataset(n_samples=7563):
    """
    Genera un dataset sintético de comportamiento de jugadores
    """
    
    # Definir estilos de juego con distribución REALISTA (no uniforme)
    playstyles = ['Aggressive', 'Strategic', 'Casual', 'Explorer', 'Competitive']
    playstyle_weights = [0.25, 0.15, 0.40, 0.10, 0.10]  # Más casuales, menos competitivos
    
    # Definir estilos de combate
    combat_styles = ['Melee', 'Ranged', 'Magic', 'Hybrid', 'Stealth']
    
    # Listas para almacenar datos
    data = []
    
    for i in range(n_samples):
        # Seleccionar estilo de juego con distribución realista
        playstyle = random.choices(playstyles, weights=playstyle_weights)[0]
        
        # Generar datos basados en el estilo de juego
        if playstyle == 'Aggressive':
            playtime_hours = np.random.normal(120, 40)  # Más variación
            sessions_per_week = np.random.normal(10, 3)
            avg_session_length = np.random.normal(3.5, 1.2)
            achievements_unlocked = np.random.normal(60, 20)
            difficulty_level = np.random.normal(7, 2)  # Más dispersión
            combat_preference = random.choices(['Melee', 'Ranged', 'Hybrid', 'Magic'], 
                                              weights=[0.5, 0.25, 0.15, 0.1])[0]  # Melee dominante
            win_rate = np.random.normal(0.55, 0.15)  # Más variabilidad
            pvp_matches = np.random.normal(200, 80)
            death_count = np.random.normal(300, 120)
            
        elif playstyle == 'Strategic':
            playtime_hours = np.random.normal(150, 60)
            sessions_per_week = np.random.normal(8, 3)
            avg_session_length = np.random.normal(4.5, 1.5)
            achievements_unlocked = np.random.normal(75, 25)
            difficulty_level = np.random.normal(8, 1.5)
            combat_preference = random.choices(['Ranged', 'Magic', 'Stealth', 'Hybrid'], 
                                              weights=[0.4, 0.3, 0.2, 0.1])[0]
            win_rate = np.random.normal(0.65, 0.12)
            pvp_matches = np.random.normal(150, 60)
            death_count = np.random.normal(180, 70)
            
        elif playstyle == 'Casual':
            playtime_hours = np.random.lognormal(3.2, 0.8)  # Distribución sesgada
            sessions_per_week = np.random.poisson(4)  # Distribución de conteo
            avg_session_length = np.random.normal(1.5, 0.8)
            achievements_unlocked = np.random.poisson(20)  # Distribución realista para logros
            difficulty_level = np.random.normal(4, 1.5)
            combat_preference = random.choices(combat_styles, 
                                              weights=[0.3, 0.25, 0.2, 0.15, 0.1])[0]
            win_rate = np.random.beta(3, 4)  # Distribución beta para proporciones
            pvp_matches = np.random.poisson(25)
            death_count = np.random.poisson(80)
            
        elif playstyle == 'Explorer':
            playtime_hours = np.random.normal(180, 70)
            sessions_per_week = np.random.normal(9, 4)
            avg_session_length = np.random.normal(5, 2)
            achievements_unlocked = np.random.normal(90, 35)
            difficulty_level = np.random.normal(6, 2)
            combat_preference = random.choices(combat_styles, 
                                              weights=[0.2, 0.25, 0.25, 0.2, 0.1])[0]
            win_rate = np.random.normal(0.50, 0.15)
            pvp_matches = np.random.normal(80, 40)
            death_count = np.random.normal(200, 80)
            
        else:  # Competitive
            playtime_hours = np.random.normal(200, 80)
            sessions_per_week = np.random.normal(15, 5)
            avg_session_length = np.random.normal(4, 1.5)
            achievements_unlocked = np.random.normal(85, 30)
            difficulty_level = np.random.normal(9, 1.2)
            combat_preference = random.choices(['Hybrid', 'Melee', 'Ranged', 'Magic'], 
                                              weights=[0.4, 0.25, 0.25, 0.1])[0]
            win_rate = np.random.normal(0.70, 0.12)
            pvp_matches = np.random.normal(400, 150)
            death_count = np.random.normal(350, 120)
        
        # INTRODUCIR CORRELACIONES NATURALES
        # Jugadores con más horas tienden a tener mejor win rate (pero no siempre)
        win_rate = min(0.95, max(0.05, win_rate + (playtime_hours - 100) * 0.0001))
        
        # Jugadores competitivos tienden a ser premium
        premium_weight = 0.5  # base
        if playstyle == 'Competitive':
            premium_weight = 0.7
        elif playstyle == 'Casual':
            premium_weight = 0.2
            
        # Calcular métricas derivadas con algo de ruido
        total_playtime = max(1, playtime_hours)
        engagement_score = (sessions_per_week * avg_session_length * 10) / 100
        
        # Skill level con correlación realista pero imperfecta
        base_skill = (win_rate * 50) + (difficulty_level * 5)
        skill_level = base_skill + np.random.normal(0, 8)  # Ruido aleatorio
        
        # Calcular probabilidad de abandono más matizada
        if playtime_hours < 15 or sessions_per_week < 1.5:
            churn_risk = 'High'
        elif playtime_hours < 50 or sessions_per_week < 3:
            churn_risk = 'Medium'
        else:
            churn_risk = 'Low'
            
        # INTRODUCIR ALGUNOS OUTLIERS EXTREMOS (jugadores hardcore)
        if random.random() < 0.01:  # 1% de outliers
            playtime_hours *= 2.5
            pvp_matches *= 3
            achievements_unlocked *= 2
        
        # Crear registro
        record = {
            'player_id': f'P{i+1:05d}',
            'playstyle': playstyle,
            'playtime_hours': max(0, playtime_hours),
            'sessions_per_week': max(0.5, sessions_per_week),
            'avg_session_length': max(0.3, avg_session_length),
            'achievements_unlocked': max(0, int(achievements_unlocked)),
            'difficulty_level': np.clip(round(difficulty_level), 1, 10),
            'combat_style': combat_preference,
            'win_rate': np.clip(win_rate, 0.01, 0.99),
            'pvp_matches': max(0, int(pvp_matches)),
            'death_count': max(0, int(death_count)),
            'engagement_score': max(0, engagement_score),
            'skill_level': np.clip(skill_level, 0, 100),
            'churn_risk': churn_risk,
            'last_login_days_ago': np.random.poisson(7),  # Distribución realista
            'premium_user': random.choices([0, 1], weights=[1-premium_weight, premium_weight])[0]
        }
        
        data.append(record)
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Añadir algunos valores nulos aleatorios (más en columnas específicas)
    null_columns = {
        'achievements_unlocked': 0.03,
        'pvp_matches': 0.02, 
        'death_count': 0.02,
        'win_rate': 0.01
    }
    
    for col, null_rate in null_columns.items():
        null_indices = np.random.choice(df.index, size=int(len(df) * null_rate), replace=False)
        df.loc[null_indices, col] = np.nan
    
    # Añadir pequeña correlación entre variables
    df['playtime_hours'] = df['playtime_hours'] + np.random.normal(0, 10, len(df))
    
    return df

def main():
    """Función principal para generar y guardar el dataset"""
    
    print("=" * 60)
    print("GENERADOR DE DATOS SINTÉTICOS")
    print("Sistema Inteligente de Recomendación para Videojuegos")
    print("=" * 60)
    print()
    
    # Generar dataset
    print("Generando dataset con 7,563 registros...")
    df = generate_gaming_dataset(7563)
    
    # Mostrar información básica
    print(f"Dataset generado con {len(df)} registros")
    print(f"Columnas: {list(df.columns)}")
    print()
    
    # Estadísticas básicas
    print("Distribución de estilos de juego:")
    print(df['playstyle'].value_counts())
    print()
    
    print("Guardando datos...")
    
    # Guardar dataset completo
    df.to_csv('data/gaming_behavior_raw.csv', index=False)
    print("Guardado: data/gaming_behavior_raw.csv")
    
    # Guardar muestra para pruebas rápidas
    df_sample = df.sample(n=1000, random_state=42)
    df_sample.to_csv('data/gaming_behavior_sample.csv', index=False)
    print("Guardado: data/gaming_behavior_sample.csv (muestra de 1000)")
    
    # Mostrar primeras filas
    print()
    print("Primeras 5 filas del dataset:")
    print(df.head())
    
    print()
    print("=" * 60)
    print("Generación completada exitosamente.")
    print("=" * 60)

if __name__ == "__main__":
    # Crear directorio si no existe
    import os
    os.makedirs('data', exist_ok=True)
    
    main()
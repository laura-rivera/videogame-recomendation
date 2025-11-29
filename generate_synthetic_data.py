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

def generate_gaming_dataset(n_samples=10000):
    """
    Genera un dataset sintético de comportamiento de jugadores
    """
    
    # Definir estilos de juego
    playstyles = ['Aggressive', 'Strategic', 'Casual', 'Explorer', 'Competitive']
    
    # Definir estilos de combate
    combat_styles = ['Melee', 'Ranged', 'Magic', 'Hybrid', 'Stealth']
    
    # Listas para almacenar datos
    data = []
    
    for i in range(n_samples):
        # Seleccionar estilo de juego (esto afectará otras variables)
        playstyle = random.choice(playstyles)
        
        # Generar datos basados en el estilo de juego
        if playstyle == 'Aggressive':
            playtime_hours = np.random.normal(120, 30)
            sessions_per_week = np.random.normal(10, 2)
            avg_session_length = np.random.normal(3.5, 1)
            achievements_unlocked = np.random.normal(60, 15)
            difficulty_level = np.random.normal(7, 1.5)
            combat_preference = random.choices(['Melee', 'Ranged', 'Hybrid'], 
                                              weights=[0.5, 0.3, 0.2])[0]
            win_rate = np.random.normal(0.55, 0.1)
            pvp_matches = np.random.normal(200, 50)
            death_count = np.random.normal(300, 80)
            
        elif playstyle == 'Strategic':
            playtime_hours = np.random.normal(150, 40)
            sessions_per_week = np.random.normal(8, 2)
            avg_session_length = np.random.normal(4.5, 1.2)
            achievements_unlocked = np.random.normal(75, 20)
            difficulty_level = np.random.normal(8, 1)
            combat_preference = random.choices(['Ranged', 'Magic', 'Stealth'], 
                                              weights=[0.4, 0.35, 0.25])[0]
            win_rate = np.random.normal(0.65, 0.08)
            pvp_matches = np.random.normal(150, 40)
            death_count = np.random.normal(180, 50)
            
        elif playstyle == 'Casual':
            playtime_hours = np.random.normal(40, 15)
            sessions_per_week = np.random.normal(4, 1.5)
            avg_session_length = np.random.normal(1.5, 0.5)
            achievements_unlocked = np.random.normal(20, 10)
            difficulty_level = np.random.normal(4, 1)
            combat_preference = random.choice(combat_styles)
            win_rate = np.random.normal(0.45, 0.12)
            pvp_matches = np.random.normal(30, 15)
            death_count = np.random.normal(80, 30)
            
        elif playstyle == 'Explorer':
            playtime_hours = np.random.normal(180, 50)
            sessions_per_week = np.random.normal(9, 2)
            avg_session_length = np.random.normal(5, 1.5)
            achievements_unlocked = np.random.normal(90, 25)
            difficulty_level = np.random.normal(6, 1.5)
            combat_preference = random.choice(combat_styles)
            win_rate = np.random.normal(0.50, 0.1)
            pvp_matches = np.random.normal(80, 30)
            death_count = np.random.normal(200, 60)
            
        else:  # Competitive
            playtime_hours = np.random.normal(200, 60)
            sessions_per_week = np.random.normal(15, 3)
            avg_session_length = np.random.normal(4, 1)
            achievements_unlocked = np.random.normal(85, 20)
            difficulty_level = np.random.normal(9, 0.8)
            combat_preference = random.choices(['Hybrid', 'Melee', 'Ranged'], 
                                              weights=[0.4, 0.3, 0.3])[0]
            win_rate = np.random.normal(0.70, 0.08)
            pvp_matches = np.random.normal(400, 100)
            death_count = np.random.normal(350, 90)
        
        # Calcular métricas derivadas
        total_playtime = max(1, playtime_hours)
        engagement_score = (sessions_per_week * avg_session_length * 10) / 100
        skill_level = (win_rate * 50) + (difficulty_level * 5)
        
        # Calcular probabilidad de abandono (churn)
        if playtime_hours < 20 or sessions_per_week < 2:
            churn_risk = 'High'
        elif playtime_hours < 80 or sessions_per_week < 5:
            churn_risk = 'Medium'
        else:
            churn_risk = 'Low'
        
        # Crear registro
        record = {
            'player_id': f'P{i+1:05d}',
            'playstyle': playstyle,
            'playtime_hours': max(0, playtime_hours),
            'sessions_per_week': max(1, sessions_per_week),
            'avg_session_length': max(0.5, avg_session_length),
            'achievements_unlocked': max(0, int(achievements_unlocked)),
            'difficulty_level': np.clip(difficulty_level, 1, 10),
            'combat_style': combat_preference,
            'win_rate': np.clip(win_rate, 0, 1),
            'pvp_matches': max(0, int(pvp_matches)),
            'death_count': max(0, int(death_count)),
            'engagement_score': max(0, engagement_score),
            'skill_level': np.clip(skill_level, 0, 100),
            'churn_risk': churn_risk,
            'last_login_days_ago': np.random.randint(0, 30),
            'premium_user': random.choices([0, 1], weights=[0.7, 0.3])[0]
        }
        
        data.append(record)
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Añadir algunos valores nulos aleatorios (realismo)
    null_columns = ['achievements_unlocked', 'pvp_matches', 'death_count']
    for col in null_columns:
        null_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
        df.loc[null_indices, col] = np.nan
    
    return df

def main():
    """Función principal para generar y guardar el dataset"""
    
    print("=" * 60)
    print("GENERADOR DE DATOS SINTÉTICOS")
    print("Sistema Inteligente de Recomendación para Videojuegos")
    print("=" * 60)
    print()
    
    # Generar dataset
    print("Generando dataset con 10,000 registros...")
    df = generate_gaming_dataset(10000)
    
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
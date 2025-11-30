import pandas as pd

def calculate_current_reference_stats():
    """Calcula las estadísticas actuales de tu dataset"""
    df = pd.read_csv('data/gaming_behavior_raw.csv')
    
    reference_stats = {}
    
    for playstyle in df['playstyle'].unique():
        style_data = df[df['playstyle'] == playstyle]
        
        reference_stats[playstyle] = {
            'playtime_hours': round(style_data['playtime_hours'].mean(), 1),
            'sessions_per_week': round(style_data['sessions_per_week'].mean(), 1),
            'difficulty_level': round(style_data['difficulty_level'].mean(), 1),
            'win_rate': round(style_data['win_rate'].mean(), 3),
            'pvp_matches': round(style_data['pvp_matches'].mean(), 1),
            'achievements_unlocked': round(style_data['achievements_unlocked'].mean(), 1),
            'avg_session_length': round(style_data['avg_session_length'].mean(), 1),
            'death_count': round(style_data['death_count'].mean(), 1),
            'engagement_score': round(style_data['engagement_score'].mean(), 2),
            'skill_level': round(style_data['skill_level'].mean(), 1)
        }
    
    # Imprimir para copiar y pegar
    print("ESTADÍSTICAS ACTUALES:")
    for style, stats in reference_stats.items():
        print(f"    '{style}': {{")
        for key, value in stats.items():
            print(f"        '{key}': {value},")
        print("    },")
    
    return reference_stats

if __name__ == "__main__":
    stats = calculate_current_reference_stats()
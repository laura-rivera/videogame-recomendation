"""
Sistema Inteligente de Recomendaciones para Videojuegos - FIXED
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class GameRecommender:
    """Sistema de recomendaciones personalizado para jugadores"""
    
    def __init__(self, model_path='models/best_model.pkl',
                 scaler_path='models/scaler.pkl',
                 metadata_path='models/model_metadata.pkl'):
        """Inicializa el sistema de recomendaciones"""
        
        # Cargar modelo y componentes
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.metadata = joblib.load(metadata_path)
        self.feature_names = self.metadata['feature_names']
        self.classes = self.metadata['classes']
        
        # Estadísticas de referencia (promedios por estilo)
        self.reference_stats = self._load_reference_stats()
        
        print("Sistema de recomendaciones inicializado")
        print(f"   Modelo: {self.metadata['model_name']}")
        print(f"   Precisión: {self.metadata['test_accuracy']:.2%}")
        print(f"   Características esperadas: {self.feature_names}")
    
    def _load_reference_stats(self):
        """Carga estadísticas de referencia por estilo de juego - ACTUALIZADO"""
        return {
            'Casual': {
                'playtime_hours': 35.6,
                'sessions_per_week': 4.0,
                'difficulty_level': 4.0,
                'win_rate': 0.423,
                'pvp_matches': 25.7,
                'achievements_unlocked': 20.4,
                'avg_session_length': 1.5,
                'death_count': 79.9,
                'engagement_score': 0.59,
                'skill_level': 41.3,
            },
            'Aggressive': {
                'playtime_hours': 120.4,
                'sessions_per_week': 10.1,
                'difficulty_level': 7.0,
                'win_rate': 0.555,
                'pvp_matches': 204.9,
                'achievements_unlocked': 61.2,
                'avg_session_length': 3.5,
                'death_count': 295.5,
                'engagement_score': 3.52,
                'skill_level': 63.1,
            },
            'Explorer': {
                'playtime_hours': 186.8,
                'sessions_per_week': 9.0,
                'difficulty_level': 6.0,
                'win_rate': 0.498,
                'pvp_matches': 78.2,
                'achievements_unlocked': 89.3,
                'avg_session_length': 5.0,
                'death_count': 197.3,
                'engagement_score': 4.46,
                'skill_level': 55.5,
            },
            'Competitive': {
                'playtime_hours': 203.8,
                'sessions_per_week': 15.1,
                'difficulty_level': 8.9,
                'win_rate': 0.715,
                'pvp_matches': 406.4,
                'achievements_unlocked': 84.5,
                'avg_session_length': 4.0,
                'death_count': 348.4,
                'engagement_score': 5.96,
                'skill_level': 80.9,
            },
            'Strategic': {
                'playtime_hours': 151.9,
                'sessions_per_week': 8.0,
                'difficulty_level': 8.0,
                'win_rate': 0.656,
                'pvp_matches': 147.4,
                'achievements_unlocked': 75.4,
                'avg_session_length': 4.5,
                'death_count': 178.6,
                'engagement_score': 3.55,
                'skill_level': 73.6,
            }
        }
    
    def prepare_input(self, player_data):
        """Prepara los datos del jugador para predicción - CORREGIDO"""
        
        # Calcular características derivadas si no están presentes
        player_data['kd_ratio'] = player_data.get('pvp_matches', 100) / max(player_data.get('death_count', 150), 1)
        player_data['play_intensity'] = player_data.get('playtime_hours', 50) / (player_data.get('sessions_per_week', 5) * 4 + 1)
        player_data['commitment_score'] = (
            player_data.get('playtime_hours', 50) * 0.3 + 
            player_data.get('sessions_per_week', 5) * 10 + 
            player_data.get('achievements_unlocked', 25) * 0.5
        )
        player_data['pvp_experience'] = np.log1p(player_data.get('pvp_matches', 50))
        player_data['achievement_rate'] = player_data.get('achievements_unlocked', 25) / (player_data.get('playtime_hours', 50) + 1)
        player_data['engagement_score'] = (player_data.get('sessions_per_week', 5) * player_data.get('avg_session_length', 2) * 10) / 100
        player_data['skill_level'] = (player_data.get('win_rate', 0.5) * 50) + (player_data.get('difficulty_level', 5) * 5)

        # Mapeo de estilos de combate a numérico
        combat_style_map = {'Melee': 0, 'Ranged': 1, 'Magic': 2, 'Hybrid': 3, 'Stealth': 4}
        combat_style = player_data.get('combat_style', 'Melee')
        player_data['combat_style_encoded'] = combat_style_map.get(combat_style, 0)
        
        # Crear DataFrame con SOLO las características que el modelo espera
        # y en el orden exacto que fueron entrenadas
        feature_dict = {}
        
        for feature in self.feature_names:
            if feature in player_data:
                feature_dict[feature] = [player_data[feature]]
            else:
                # Valores por defecto razonables basados en el tipo de característica
                default_values = {
                    'playtime_hours': 50,
                    'sessions_per_week': 5,
                    'avg_session_length': 2.0,
                    'achievements_unlocked': 25,
                    'difficulty_level': 5,
                    'win_rate': 0.5,
                    'pvp_matches': 50,
                    'death_count': 100,
                    'engagement_score': 1.0,
                    'skill_level': 50,
                    'last_login_days_ago': 7,
                    'premium_user': 0,
                    'kd_ratio': 1.0,
                    'play_intensity': 2.5,
                    'commitment_score': 100,
                    'pvp_experience': 3.9,
                    'achievement_rate': 0.5,
                    'combat_style_encoded': 0
                }
                feature_dict[feature] = [default_values.get(feature, 0)]
        
        # Crear DataFrame con el orden exacto de las características del modelo
        X = pd.DataFrame(feature_dict)
        X = X[self.feature_names]
        
        print(f"Características preparadas: {list(X.columns)}")
        print(f"Valores: {X.iloc[0].to_dict()}")
    
        # Escalar si el scaler está disponible
        if hasattr(self, 'scaler') and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            return X_scaled
        else:
            return X.values
    
    def predict_playstyle(self, player_data):
        """Predice el estilo de juego del jugador"""
        
        X = self.prepare_input(player_data)
        
        # Predicción
        predicted_style = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = probabilities.max()
        
        # Obtener probabilidades por clase
        prob_dict = {cls: prob for cls, prob in zip(self.classes, probabilities)}
        
        return {
            'predicted_style': predicted_style,
            'confidence': confidence,
            'probabilities': prob_dict
        }
    
    def generate_recommendations(self, player_data, prediction_result):
        """Genera recomendaciones personalizadas con umbrales optimizados"""
        
        predicted_style = prediction_result['predicted_style']
        confidence = prediction_result['confidence']
        
        recommendations = []
        
        # Obtener estadísticas de referencia para el estilo predicho
        ref_stats = self.reference_stats.get(predicted_style, {})
        
        # UMBRALES OPTIMIZADOS BASADOS EN DATOS REALES:
        
        # 1. Dificultad - umbral más estricto (1.5 en lugar de 2)
        current_difficulty = player_data.get('difficulty_level', 5)
        optimal_difficulty = ref_stats.get('difficulty_level', 7)
        
        if abs(current_difficulty - optimal_difficulty) >= 1.5:
            if current_difficulty < optimal_difficulty:
                recommendations.append({
                    'title': 'Incrementa el nivel de dificultad',
                    'reason': f'Los jugadores {predicted_style} típicamente juegan en dificultad {optimal_difficulty:.1f}, mientras que tu nivel actual es {current_difficulty:.1f}.',
                    'impact': f'Aumentar la dificultad podría mejorar tu engagement en un {min(20, int((optimal_difficulty - current_difficulty) * 6)):.0f}%.',
                    'priority': 'Alta',
                    'action': f'Ajusta la dificultad de {current_difficulty:.1f} a {optimal_difficulty:.1f}'
                })
            else:
                recommendations.append({
                    'title': 'Considera reducir la dificultad temporalmente',
                    'reason': f'Tu nivel de dificultad ({current_difficulty:.1f}) está por encima del promedio para jugadores {predicted_style} ({optimal_difficulty:.1f}).',
                    'impact': 'Reducir ligeramente la dificultad podría mejorar tu win rate y satisfacción.',
                    'priority': 'Media',
                    'action': f'Prueba ajustar de {current_difficulty:.1f} a {optimal_difficulty:.1f}'
                })
        
        # 2. Tiempo de juego - umbral más sensible (65% en lugar de 50%)
        current_playtime = player_data.get('playtime_hours', 50)
        optimal_playtime = ref_stats.get('playtime_hours', 100)
        
        if current_playtime < optimal_playtime * 0.65:
            recommendations.append({
                'title': 'Aumenta tu tiempo de práctica',
                'reason': f'Con {current_playtime:.0f} horas jugadas, estás por debajo del promedio de jugadores {predicted_style} ({optimal_playtime:.0f} horas).',
                'impact': f'Incrementar tu tiempo de juego podría mejorar tu win rate en un {min(15, int((optimal_playtime - current_playtime) / 15))}%.',
                'priority': 'Media',
                'action': f'Objetivo: alcanzar {optimal_playtime * 0.8:.0f} horas'
            })
        
        # 3. Win rate - umbral más ajustado (0.04 en lugar de 0.05)
        current_win_rate = player_data.get('win_rate', 0.5)
        optimal_win_rate = ref_stats.get('win_rate', 0.6)
        
        if current_win_rate < optimal_win_rate - 0.04:
            if predicted_style == 'Aggressive':
                recommendations.append({
                    'title': 'Refina tu estrategia de combate',
                    'reason': f'Tu win rate ({current_win_rate:.1%}) está por debajo del promedio para jugadores agresivos ({optimal_win_rate:.1%}).',
                    'impact': 'Combinar agresividad con timing estratégico podría aumentar tu win rate en 10-15%.',
                    'priority': 'Alta',
                    'action': 'Practica combos avanzados y aprende patrones de enemigos'
                })
            elif predicted_style == 'Strategic':
                recommendations.append({
                    'title': 'Optimiza tu planificación',
                    'reason': f'Como jugador estratégico con win rate de {current_win_rate:.1%}, tienes potencial para alcanzar {optimal_win_rate:.1%}.',
                    'impact': 'Analizar tus partidas podría incrementar tu efectividad significativamente.',
                    'priority': 'Alta',
                    'action': 'Revisa replays de tus derrotas e identifica patrones de error'
                })
            elif predicted_style == 'Casual':
                recommendations.append({
                    'title': 'Experimenta con nuevas estrategias',
                    'reason': f'Tu win rate actual ({current_win_rate:.1%}) indica que estás en proceso de aprendizaje.',
                    'impact': 'Probar diferentes enfoques te ayudará a descubrir qué funciona mejor para ti.',
                    'priority': 'Baja',
                    'action': 'Experimenta con diferentes estilos de combate'
                })
            elif predicted_style == 'Competitive':
                recommendations.append({
                    'title': 'Enfócate en la consistencia',
                    'reason': f'Para un jugador competitivo, tu win rate ({current_win_rate:.1%}) debería acercarse más a {optimal_win_rate:.1%}.',
                    'impact': 'Mejorar la consistencia te posicionará mejor en rankings competitivos.',
                    'priority': 'Alta',
                    'action': 'Trabaja en mantener un rendimiento estable partida tras partida'
                })
        
        # 4. Actividad PvP - umbral específico por estilo
        current_pvp = player_data.get('pvp_matches', 50)
        optimal_pvp = ref_stats.get('pvp_matches', 100)
        
        pvp_thresholds = {
            'Competitive': 0.7,    # 70% del óptimo
            'Aggressive': 0.6,     # 60% del óptimo  
            'Strategic': 0.5,      # 50% del óptimo
            'Explorer': 0.4,       # 40% del óptimo
            'Casual': 0.3          # 30% del óptimo
        }
        
        threshold = pvp_thresholds.get(predicted_style, 0.5)
        if current_pvp < optimal_pvp * threshold:
            recommendations.append({
                'title': 'Incrementa tu actividad en PvP',
                'reason': f'Con {current_pvp:.0f} partidas PvP, estás por debajo del esperado para {predicted_style} ({optimal_pvp:.0f}+ partidas).',
                'impact': f'Aumentar tu actividad PvP mejoraría tus habilidades y te daría más experiencia competitiva.',
                'priority': 'Alta' if predicted_style in ['Competitive', 'Aggressive'] else 'Media',
                'action': f'Objetivo: alcanzar {optimal_pvp * threshold:.0f} partidas PvP'
            })
        
        # 5. Logros - umbral ajustado para Explorers (75% en lugar de 70%)
        current_achievements = player_data.get('achievements_unlocked', 25)
        optimal_achievements = ref_stats.get('achievements_unlocked', 50)
        
        if predicted_style == 'Explorer' and current_achievements < optimal_achievements * 0.75:
            recommendations.append({
                'title': 'Explora más contenido del juego',
                'reason': f'Como Explorer con {current_achievements:.0f} logros, tienes contenido por descubrir (promedio: {optimal_achievements:.0f}).',
                'impact': f'Desbloquear {int(optimal_achievements - current_achievements)} logros adicionales te dará acceso a recompensas exclusivas.',
                'priority': 'Media',
                'action': 'Consulta la guía de logros y establece metas semanales'
            })
        
        # 6. Sesiones de juego - umbral más bajo (2 sesiones)
        current_sessions = player_data.get('sessions_per_week', 5)
        optimal_sessions = ref_stats.get('sessions_per_week', 8)
        
        if current_sessions < 2:
            recommendations.append({
                'title': 'Establece una rutina de juego más consistente',
                'reason': f'Con {current_sessions:.0f} sesiones por semana, tu progreso puede ser lento. Ideal: {optimal_sessions:.0f} sesiones.',
                'impact': 'Sesiones más frecuentes mejoran la retención de habilidades.',
                'priority': 'Media',
                'action': f'Intenta aumentar a {min(optimal_sessions, current_sessions + 2):.0f} sesiones semanales'
            })
        
        # Si hay pocas recomendaciones, añadir generales
        if len(recommendations) < 3:
            recommendations.append({
                'title': 'Mantén tu excelente desempeño',
                'reason': f'Tu perfil está bien alineado con el estilo {predicted_style}. Confianza: {confidence:.1%}.',
                'impact': 'Continúa con tu enfoque actual y considera pequeños ajustes.',
                'priority': 'Baja',
                'action': 'Sigue mejorando y explora contenido avanzado'
            })
        
        # Ordenar por prioridad
        priority_order = {'Alta': 0, 'Media': 1, 'Baja': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations
    
    def analyze_player(self, player_data):
        """Análisis completo del jugador con predicción y recomendaciones"""
        
        # Predecir estilo
        prediction = self.predict_playstyle(player_data)
        
        # Generar recomendaciones
        recommendations = self.generate_recommendations(player_data, prediction)
        
        # Calcular métricas adicionales
        engagement_score = (
            player_data.get('sessions_per_week', 5) * 
            player_data.get('avg_session_length', 2) * 10
        ) / 100
        
        skill_level = (
            player_data.get('win_rate', 0.5) * 50 + 
            player_data.get('difficulty_level', 5) * 5
        )
        
        return {
            'prediction': prediction,
            'recommendations': recommendations,
            'metrics': {
                'engagement_score': engagement_score,
                'skill_level': skill_level,
                'total_playtime': player_data.get('playtime_hours', 0),
                'win_rate': player_data.get('win_rate', 0)
            }
        }

def main():
    """Función de prueba - CORREGIDA"""
    print("=" * 60)
    print("SISTEMA DE RECOMENDACIONES - PRUEBA")
    print("=" * 60)
    
    # Crear sistema
    recommender = GameRecommender()
    
    # Perfil de prueba - Asegurar que tenemos todas las características necesarias
    test_player = {
        'playtime_hours': 45,
        'sessions_per_week': 3,
        'avg_session_length': 2.0,
        'achievements_unlocked': 15,
        'difficulty_level': 4,
        'combat_style': 'Melee',
        'win_rate': 0.42,
        'pvp_matches': 25,
        'death_count': 80,
        'last_login_days_ago': 2,
        'premium_user': 0
        # Las características derivadas se calcularán automáticamente
    }
    
    print("\nAnalizando perfil de jugador...")
    result = recommender.analyze_player(test_player)
    
    print(f"\nEstilo Predicho: {result['prediction']['predicted_style']}")
    print(f"   Confianza: {result['prediction']['confidence']:.1%}")
    
    print("\nProbabilidades por estilo:")
    for style, prob in result['prediction']['probabilities'].items():
        print(f"   {style}: {prob:.1%}")
    
    print("\nRecomendaciones:")
    for i, rec in enumerate(result['recommendations'][:5], 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Razón: {rec['reason']}")
        print(f"   Impacto: {rec['impact']}")
        print(f"   Prioridad: {rec['priority']}")
        print(f"   Acción: {rec['action']}")

if __name__ == "__main__":
    main()
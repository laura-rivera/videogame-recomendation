"""
Sistema Inteligente de Recomendaciones para Videojuegos
"""

import joblib
import pandas as pd
import numpy as np

class GameRecommender:
    """Sistema de recomendaciones personalizado para jugadores"""
    
    def __init__(self, model_path=r"C:\Users\laura\Source\Repos\videogame-recomendation\models\best_model.pkl",
                 metadata_path=r"C:\Users\laura\Source\Repos\videogame-recomendation\models\model_metadata.pkl"):
        """Inicializa el sistema de recomendaciones SIN SCALER"""
        
        try:
            # Cargar modelo y componentes (sin scaler)
            self.model = joblib.load(model_path)
            self.metadata = joblib.load(metadata_path)
            self.feature_names = self.metadata['feature_names']
            self.classes = self.metadata['classes']
            
            # Estadísticas de referencia (promedios por estilo)
            self.reference_stats = self._load_reference_stats()
            
            print("Sistema de recomendaciones inicializado (SIN SCALER)")
            print(f"   Modelo: {self.metadata['model_name']}")
            print(f"   Precisión: {self.metadata['test_accuracy']:.2%}")
            print(f"   Características esperadas: {self.feature_names}")
            print(f"   Clases disponibles: {self.classes}")
            
        except Exception as e:
            print(f"Error cargando el modelo: {e}")
            # Crear un modelo dummy para testing
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Crea un modelo dummy para testing cuando no hay modelo real"""
        from sklearn.ensemble import RandomForestClassifier
        
        print("Creando modelo dummy para testing...")
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.feature_names = ['playtime_hours', 'sessions_per_week', 'avg_session_length', 
                            'achievements_unlocked', 'difficulty_level', 'win_rate', 
                            'pvp_matches', 'death_count', 'last_login_days_ago', 'premium_user',
                            'kd_ratio', 'engagement_score', 'skill_level']
        self.classes = ['Casual', 'Aggressive', 'Explorer', 'Competitive', 'Strategic']
        self.metadata = {
            'model_name': 'DummyModel',
            'test_accuracy': 0.75,
            'precision': 0.72,
            'recall': 0.70,
            'timestamp': '2024-01-01'
        }
        
        # Crear datos dummy para entrenar el modelo
        np.random.seed(42)
        n_samples = 1000
        X_dummy = np.random.randn(n_samples, len(self.feature_names))
        y_dummy = np.random.choice(self.classes, n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Entrenar el modelo directamente con datos sin escalar
        self.model.fit(X_dummy, y_dummy)
        
        self.reference_stats = self._load_reference_stats()
    
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
        """Prepara los datos del jugador"""
        
        player_data = player_data.copy()
        
        # Calcular características derivadas
        player_data['kd_ratio'] = player_data.get('pvp_matches', 1) / max(player_data.get('death_count', 1), 1)
        player_data['engagement_score'] = (player_data.get('sessions_per_week', 5) * player_data.get('avg_session_length', 2) * 10) / 100
        player_data['skill_level'] = (player_data.get('win_rate', 0.5) * 50) + (player_data.get('difficulty_level', 5) * 5)
        player_data['play_intensity'] = player_data.get('playtime_hours', 50) / max(player_data.get('sessions_per_week', 1) * 4, 1)
        player_data['commitment_score'] = (
            player_data.get('playtime_hours', 50) * 0.3 + 
            player_data.get('sessions_per_week', 5) * 10 + 
            player_data.get('achievements_unlocked', 25) * 0.5
        )
        player_data['pvp_experience'] = np.log1p(player_data.get('pvp_matches', 50))
        player_data['achievement_rate'] = player_data.get('achievements_unlocked', 25) / max(player_data.get('playtime_hours', 1), 1)

        combat_style_map = {'Melee': 0, 'Ranged': 1, 'Magic': 2, 'Hybrid': 3, 'Stealth': 4}
        combat_style = player_data.get('combat_style', 'Melee')
        player_data['combat_style_encoded'] = combat_style_map.get(combat_style, 0)
        
        # Crear DataFrame con el orden exacto
        feature_dict = {}
        for feature in self.feature_names:
            if feature in player_data:
                feature_dict[feature] = [player_data[feature]]
            else:
                default_values = {
                    'playtime_hours': 50, 'sessions_per_week': 5, 'avg_session_length': 2.0,
                    'achievements_unlocked': 25, 'difficulty_level': 5, 'win_rate': 0.5,
                    'pvp_matches': 50, 'death_count': 100, 'engagement_score': 1.0,
                    'skill_level': 50, 'kd_ratio': 1.0, 'play_intensity': 2.5,
                    'commitment_score': 100, 'pvp_experience': 3.9, 'achievement_rate': 0.5,
                    'last_login_days_ago': 7, 'combat_style_encoded': 0, 'premium_user': 0
                }
                feature_dict[feature] = [default_values.get(feature, 0)]
        
        X = pd.DataFrame(feature_dict)
        X = X[self.feature_names]
        
        print(f"Datos preparados:")
        print(f"  playtime_hours: {X['playtime_hours'].iloc[0]}")
        print(f"  win_rate: {X['win_rate'].iloc[0]}")
        print(f"  pvp_matches: {X['pvp_matches'].iloc[0]}")
        
        return X 
    
    def predict_playstyle(self, player_data):
        """Predice el estilo de juego"""
        
        X = self.prepare_input(player_data)
        
        try:
            X_values = X.values
            
            predicted_style = self.model.predict(X_values)[0]
            probabilities = self.model.predict_proba(X_values)[0]
            confidence = probabilities.max()
            
            prob_dict = {cls: prob for cls, prob in zip(self.classes, probabilities)}
            
            print(f"DEBUG - Predicción:")
            print(f"  Input shape: {X_values.shape}")
            print(f"  Predicción: {predicted_style}")
            print(f"  Confianza: {confidence:.2%}")
            
            return {
                'predicted_style': predicted_style,
                'confidence': confidence,
                'probabilities': prob_dict
            }
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            # Fallback a reglas
            predicted_style, confidence, prob_dict = self._rule_based_prediction(player_data)
            return {
                'predicted_style': predicted_style,
                'confidence': confidence,
                'probabilities': prob_dict
            }

    def _analyze_prediction(self, player_data, probabilities):
        """Analiza por qué se hizo cierta predicción"""
        print(f"ANÁLISIS DE PREDICCIÓN:")
        
        playtime = player_data.get('playtime_hours', 50)
        win_rate = player_data.get('win_rate', 0.5)
        pvp_matches = player_data.get('pvp_matches', 50)
        achievements = player_data.get('achievements_unlocked', 25)
        difficulty = player_data.get('difficulty_level', 5)
        
        style_indicators = {
            'Casual': playtime < 50 and win_rate < 0.45,
            'Aggressive': pvp_matches > 150 and win_rate > 0.55,
            'Explorer': achievements > 70 and playtime > 100,
            'Competitive': pvp_matches > 200 and win_rate > 0.65,
            'Strategic': difficulty > 7 and win_rate > 0.6
        }
        
        for style, indicator in style_indicators.items():
            if indicator:
                print(f"  ✓ Indicador de {style}: {indicator}")
    
    def _rule_based_prediction(self, player_data):
        """Predicción basada en reglas como fallback"""
        playtime = player_data.get('playtime_hours', 50)
        win_rate = player_data.get('win_rate', 0.5)
        difficulty = player_data.get('difficulty_level', 5)
        pvp_matches = player_data.get('pvp_matches', 50)
        achievements = player_data.get('achievements_unlocked', 25)
        
        # Reglas simples para determinar el estilo
        if playtime < 50 and win_rate < 0.4:
            style = 'Casual'
            confidence = 0.8
        elif pvp_matches > 300 and win_rate > 0.7:
            style = 'Competitive' 
            confidence = 0.9
        elif achievements > 80 and playtime > 150:
            style = 'Explorer'
            confidence = 0.7
        elif difficulty > 7 and win_rate > 0.6:
            style = 'Strategic'
            confidence = 0.75
        elif pvp_matches > 150:
            style = 'Aggressive'
            confidence = 0.7
        else:
            style = 'Casual'
            confidence = 0.6
        
        # Crear distribución de probabilidades
        prob_dict = {cls: 0.05 for cls in self.classes}
        prob_dict[style] = confidence
        remaining_prob = 1.0 - confidence
        other_styles = [cls for cls in self.classes if cls != style]
        if other_styles:
            prob_per_other = remaining_prob / len(other_styles)
            for cls in other_styles:
                prob_dict[cls] = prob_per_other
        
        return style, confidence, prob_dict
    
    def generate_recommendations(self, player_data, prediction_result):
        """Genera recomendaciones personalizadas"""
        
        predicted_style = prediction_result['predicted_style']
        confidence = prediction_result['confidence']
        
        recommendations = []
        
        ref_stats = self.reference_stats.get(predicted_style, {})
        
        metrics_to_check = [
            ('playtime_hours', 'Horas de juego', 'Alta', 'Media'),
            ('win_rate', 'Tasa de victoria', 'Alta', 'Media'), 
            ('difficulty_level', 'Nivel de dificultad', 'Media', 'Baja'),
            ('pvp_matches', 'Partidas PvP', 'Media', 'Baja'),
            ('achievements_unlocked', 'Logros desbloqueados', 'Baja', 'Baja')
        ]
        
        for metric, name, priority_high, priority_low in metrics_to_check:
            current = player_data.get(metric, 0)
            optimal = ref_stats.get(metric, current)
            
            if optimal == 0:
                continue
                
            ratio = current / optimal if optimal != 0 else 1
            
            if ratio < 0.6:
                recommendations.append({
                    'title': f'Mejora tu {name}',
                    'reason': f'Tu {name} ({current:.1f}) está significativamente por debajo del promedio para jugadores {predicted_style} ({optimal:.1f}).',
                    'impact': f'Incrementar tu {name} mejorará tu alineación con el estilo {predicted_style}.',
                    'priority': priority_high,
                    'action': f'Establece metas para alcanzar al menos {optimal * 0.8:.1f}'
                })
            elif ratio > 1.5:
                recommendations.append({
                    'title': f'Ajusta tu enfoque de {name}',
                    'reason': f'Tu {name} ({current:.1f}) excede el promedio para jugadores {predicted_style} ({optimal:.1f}).',
                    'impact': f'Un balance mejorado podría optimizar tu experiencia de juego.',
                    'priority': priority_low,
                    'action': f'Considera diversificar tu enfoque de juego'
                })
        
        style_specific_recs = {
            'Competitive': {
                'title': 'Enfócate en el meta actual',
                'reason': 'Mantente actualizado con las estrategias competitivas y tier lists.',
                'impact': 'Mejorarás tu ranking y efectividad en partidas.',
                'priority': 'Alta',
                'action': 'Estudia replays de jugadores profesionales'
            },
            'Casual': {
                'title': 'Disfruta el juego a tu ritmo',
                'reason': 'El enfoque casual permite disfrutar la experiencia sin presión.',
                'impact': 'Mayor satisfacción y menos frustración.',
                'priority': 'Baja', 
                'action': 'Explora modos de juego relajados y eventos sociales'
            },
            'Explorer': {
                'title': 'Sigue descubriendo contenido',
                'reason': 'Tu curiosidad es tu mayor fortaleza.',
                'impact': 'Descubrirás secretos y contenido único.',
                'priority': 'Media',
                'action': 'Completa logros secundarios y busca easter eggs'
            },
            'Strategic': {
                'title': 'Profundiza en el análisis',
                'reason': 'Tu enfoque metódico te da ventaja.',
                'impact': 'Tomarás mejores decisiones en situaciones complejas.',
                'priority': 'Alta',
                'action': 'Crea guías propias y analiza mecánicas avanzadas'
            },
            'Aggressive': {
                'title': 'Perfecciona tu agresividad',
                'reason': 'Tu estilo agresivo puede ser optimizado con timing.',
                'impact': 'Mayor efectividad en combates intensos.',
                'priority': 'Media',
                'action': 'Practica combos y timing de ataques'
            }
        }
        
        if predicted_style in style_specific_recs:
            recommendations.append(style_specific_recs[predicted_style])
        
        if len(recommendations) < 2:
            recommendations.append({
                'title': 'Mantén tu progreso actual',
                'reason': f'Tu perfil muestra buena alineación con el estilo {predicted_style}.',
                'impact': 'Continúa desarrollándote de manera consistente.',
                'priority': 'Baja',
                'action': 'Sigue jugando regularmente y establece pequeñas metas'
            })
        
        priority_order = {'Alta': 0, 'Media': 1, 'Baja': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations
    
    def analyze_player(self, player_data):
        """Análisis completo del jugador con predicción y recomendaciones"""
        
        print(f"Analizando perfil con datos: {player_data}")
        
        prediction = self.predict_playstyle(player_data)
        recommendations = self.generate_recommendations(player_data, prediction)
        
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


def test_profiles():
    """Probar los perfiles específicos que mencionas"""
    print("=" * 60)
    print("PRUEBA DE PERFILES DIVERSOS")
    print("=" * 60)
    
    # Crear sistema SIN pasar scaler_path
    recommender = GameRecommender(
        model_path='models/best_model.pkl',
        metadata_path='models/model_metadata.pkl'
    )
    
    profiles = {
        'p1.json': {
            'playtime_hours': 35,
            'sessions_per_week': 3,
            'avg_session_length': 1.2,
            'achievements_unlocked': 18,
            'difficulty_level': 3,
            'combat_style': 'Melee',
            'win_rate': 0.38,
            'pvp_matches': 15,
            'death_count': 65,
            'last_login_days_ago': 5,
            'premium_user': 0
        },
        'p2.json': {
            'playtime_hours': 220,
            'sessions_per_week': 14,
            'avg_session_length': 4.2,
            'achievements_unlocked': 92,
            'difficulty_level': 9,
            'combat_style': 'Hybrid', 
            'win_rate': 0.72,
            'pvp_matches': 380,
            'death_count': 320,
            'last_login_days_ago': 0,
            'premium_user': 1
        },
        'p3.json': {
            'playtime_hours': 180,
            'sessions_per_week': 8,
            'avg_session_length': 5.5,
            'achievements_unlocked': 95,
            'difficulty_level': 6,
            'combat_style': 'Magic',
            'win_rate': 0.51,
            'pvp_matches': 70,
            'death_count': 180,
            'last_login_days_ago': 2,
            'premium_user': 1
        },
        'p4.json': {
            'playtime_hours': 160,
            'sessions_per_week': 7,
            'avg_session_length': 4.8,
            'achievements_unlocked': 82,
            'difficulty_level': 8,
            'combat_style': 'Stealth',
            'win_rate': 0.68,
            'pvp_matches': 130,
            'death_count': 165,
            'last_login_days_ago': 1,
            'premium_user': 1
        }
    }
    
    for profile_name, profile_data in profiles.items():
        print(f"\n{'='*40}")
        print(f"ANALIZANDO: {profile_name}")
        print(f"{'='*40}")
        
        result = recommender.analyze_player(profile_data)
        
        print(f"Estilo Predicho: {result['prediction']['predicted_style']}")
        print(f"Confianza: {result['prediction']['confidence']:.1%}")
        
        print("\nProbabilidades:")
        for style, prob in result['prediction']['probabilities'].items():
            print(f"  {style}: {prob:.1%}")
        
        print(f"\nMétricas:")
        print(f"  Engagement: {result['metrics']['engagement_score']:.2f}")
        print(f"  Skill Level: {result['metrics']['skill_level']:.2f}")
        
        print(f"\nRecomendaciones principales:")
        for i, rec in enumerate(result['recommendations'][:3], 1):
            print(f"  {i}. {rec['title']} [{rec['priority']}]")


if __name__ == "__main__":
    # Inicializar el sistema SIN scaler
    recommender = GameRecommender(
        model_path='models/best_model.pkl',
        metadata_path='models/model_metadata.pkl'
    )
    
    # Probar con un perfil de ejemplo
    test_profile = {
        'playtime_hours': 35,
        'sessions_per_week': 3,
        'avg_session_length': 1.2,
        'achievements_unlocked': 18,
        'difficulty_level': 3,
        'win_rate': 0.38,
        'pvp_matches': 15,
        'death_count': 65,
        'last_login_days_ago': 5,
        'premium_user': 0
    }
    
    result = recommender.analyze_player(test_profile)
    print(f"\nResultado: {result['prediction']['predicted_style']}")
    print(f"Confianza: {result['prediction']['confidence']:.1%}")
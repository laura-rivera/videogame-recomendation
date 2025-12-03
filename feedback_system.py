"""
Sistema de Feedback y Mejora Continua
Etapa 4: Despliegue y Mantenimiento
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path

class FeedbackSystem:
    """Sistema para recopilar feedback y almacenar datos para reentrenamiento"""
    
    def __init__(self, data_dir='data/feedback'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivos de almacenamiento
        self.feedback_file = self.data_dir / 'user_feedback.jsonl'
        self.predictions_file = self.data_dir / 'prediction_history.jsonl'
        self.profiles_file = self.data_dir / 'analyzed_profiles.jsonl'
        
    def save_prediction(self, player_data, prediction_result, session_id=None):
        """
        Guarda una predicción realizada por el sistema
        
        Args:
            player_data: Datos del jugador
            prediction_result: Resultado de la predicción
            session_id: ID de sesión (opcional)
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id or self._generate_session_id(),
            'player_data': player_data,
            'predicted_style': prediction_result['predicted_style'],
            'confidence': float(prediction_result['confidence']),
            'probabilities': {k: float(v) for k, v in prediction_result['probabilities'].items()},
            'metrics': prediction_result.get('metrics', {})
        }
        
        self._append_jsonl(self.predictions_file, record)
        return record['session_id']
    
    def save_feedback(self, session_id, feedback_data):
        """
        Guarda el feedback del usuario sobre una recomendación
        
        Args:
            session_id: ID de la sesión
            feedback_data: Dict con el feedback del usuario
                - rating: calificación 1-5
                - prediction_correct: bool si la predicción fue correcta
                - actual_style: estilo real (si prediction_correct=False)
                - recommendations_helpful: bool
                - comments: texto libre
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'rating': feedback_data.get('rating'),
            'prediction_correct': feedback_data.get('prediction_correct'),
            'actual_style': feedback_data.get('actual_style'),
            'recommendations_helpful': feedback_data.get('recommendations_helpful'),
            'comments': feedback_data.get('comments', ''),
            'feedback_type': feedback_data.get('feedback_type', 'general')
        }
        
        self._append_jsonl(self.feedback_file, record)
        print(f"✅ Feedback guardado para sesión {session_id}")
    
    def save_profile_for_training(self, player_data, predicted_style, actual_style=None):
        """
        Guarda un perfil para futuro entrenamiento del modelo
        
        Args:
            player_data: Datos del jugador
            predicted_style: Estilo predicho por el modelo
            actual_style: Estilo real confirmado por el usuario (si existe)
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'player_data': player_data,
            'predicted_style': predicted_style,
            'actual_style': actual_style or predicted_style,
            'confirmed': actual_style is not None
        }
        
        self._append_jsonl(self.profiles_file, record)
        print(f"✅ Perfil guardado para entrenamiento futuro")
    
    def get_feedback_summary(self, days=30):
        """Obtiene resumen del feedback reciente"""
        feedbacks = self._read_jsonl(self.feedback_file, days=days)
        
        if not feedbacks:
            return None
        
        df = pd.DataFrame(feedbacks)
        
        summary = {
            'total_feedback': len(df),
            'avg_rating': df['rating'].mean() if 'rating' in df else None,
            'prediction_accuracy': df['prediction_correct'].mean() if 'prediction_correct' in df else None,
            'recommendations_helpful_rate': df['recommendations_helpful'].mean() if 'recommendations_helpful' in df else None,
            'most_common_issues': self._extract_common_issues(df)
        }
        
        return summary
    
    def get_profiles_for_retraining(self, min_confirmed=100):
        """
        Obtiene perfiles para reentrenamiento del modelo
        
        Args:
            min_confirmed: Número mínimo de perfiles confirmados necesarios
            
        Returns:
            DataFrame con perfiles listos para reentrenamiento
        """
        profiles = self._read_jsonl(self.profiles_file)
        
        if not profiles:
            print("⚠️ No hay perfiles guardados aún")
            return None
        
        df = pd.DataFrame(profiles)
        
        # Filtrar solo perfiles confirmados
        confirmed = df[df['confirmed'] == True]
        
        if len(confirmed) < min_confirmed:
            print(f"⚠️ Solo {len(confirmed)} perfiles confirmados. Se necesitan al menos {min_confirmed}")
            return None
        
        print(f"✅ {len(confirmed)} perfiles confirmados disponibles para reentrenamiento")
        return confirmed
    
    def export_training_data(self, output_path='data/retraining_data.csv'):
        """
        Exporta datos para reentrenamiento en formato CSV
        """
        profiles = self.get_profiles_for_retraining(min_confirmed=10)
        
        if profiles is None:
            return False
        
        # Expandir player_data a columnas individuales
        player_data_df = pd.json_normalize(profiles['player_data'])
        player_data_df['playstyle'] = profiles['actual_style'].values
        
        player_data_df.to_csv(output_path, index=False)
        print(f"✅ Datos exportados a {output_path}")
        return True
    
    def _append_jsonl(self, filepath, record):
        """Agrega un registro a un archivo JSONL"""
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    def _read_jsonl(self, filepath, days=None):
        """Lee registros de un archivo JSONL"""
        if not filepath.exists():
            return []
        
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    
                    # Filtrar por fecha si se especifica
                    if days:
                        record_date = datetime.fromisoformat(record['timestamp'])
                        cutoff_date = datetime.now() - pd.Timedelta(days=days)
                        if record_date < cutoff_date:
                            continue
                    
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        
        return records
    
    def _generate_session_id(self):
        """Genera un ID único de sesión"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
    
    def _extract_common_issues(self, df):
        """Extrae problemas comunes del feedback"""
        # Aquí podrías implementar análisis de texto de los comentarios
        # Por ahora retornamos un placeholder
        return []


class ModelMonitoring:
    """Sistema de monitoreo del rendimiento del modelo"""
    
    def __init__(self, feedback_system):
        self.feedback_system = feedback_system
    
    def calculate_drift_metrics(self, window_days=30):
        """
        Calcula métricas de drift del modelo
        
        Returns:
            Dict con métricas de rendimiento
        """
        predictions = self.feedback_system._read_jsonl(
            self.feedback_system.predictions_file, 
            days=window_days
        )
        
        feedbacks = self.feedback_system._read_jsonl(
            self.feedback_system.feedback_file,
            days=window_days
        )
        
        if not predictions or not feedbacks:
            return None
        
        pred_df = pd.DataFrame(predictions)
        feed_df = pd.DataFrame(feedbacks)
        
        # Unir predicciones con feedback
        merged = pred_df.merge(feed_df, on='session_id', how='inner')
        
        metrics = {
            'period': f'Last {window_days} days',
            'total_predictions': len(pred_df),
            'predictions_with_feedback': len(merged),
            'feedback_rate': len(merged) / len(pred_df) if len(pred_df) > 0 else 0,
            'accuracy': merged['prediction_correct'].mean() if len(merged) > 0 else None,
            'avg_confidence': pred_df['confidence'].mean(),
            'low_confidence_rate': (pred_df['confidence'] < 0.7).mean()
        }
        
        return metrics
    
    def check_retraining_needed(self, accuracy_threshold=0.75, drift_threshold=0.1):
        """
        Verifica si se necesita reentrenar el modelo
        
        Args:
            accuracy_threshold: Umbral mínimo de accuracy
            drift_threshold: Cambio máximo aceptable en accuracy
            
        Returns:
            Tuple (needs_retraining, reason, metrics)
        """
        current_metrics = self.calculate_drift_metrics(window_days=30)
        past_metrics = self.calculate_drift_metrics(window_days=90)
        
        if not current_metrics or not past_metrics:
            return False, "Insufficient data", None
        
        # Verificar accuracy absoluta
        if current_metrics['accuracy'] and current_metrics['accuracy'] < accuracy_threshold:
            return True, f"Accuracy below threshold ({current_metrics['accuracy']:.2%} < {accuracy_threshold:.2%})", current_metrics
        
        # Verificar drift
        if current_metrics['accuracy'] and past_metrics['accuracy']:
            drift = past_metrics['accuracy'] - current_metrics['accuracy']
            if drift > drift_threshold:
                return True, f"Significant drift detected ({drift:.2%})", current_metrics
        
        # Verificar tasa de confianza baja
        if current_metrics['low_confidence_rate'] > 0.3:
            return True, f"High rate of low-confidence predictions ({current_metrics['low_confidence_rate']:.2%})", current_metrics
        
        return False, "Model performing well", current_metrics
    
    def generate_monitoring_report(self):
        """Genera un reporte completo de monitoreo"""
        metrics = self.calculate_drift_metrics()
        needs_retraining, reason, _ = self.check_retraining_needed()
        feedback_summary = self.feedback_system.get_feedback_summary()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_metrics': metrics,
            'needs_retraining': needs_retraining,
            'retraining_reason': reason,
            'feedback_summary': feedback_summary
        }
        
        return report


def generate_documentation():
    """Genera documentación técnica del sistema"""
    
    doc = """
# SISTEMA INTELIGENTE DE RECOMENDACIÓN - DOCUMENTACIÓN TÉCNICA

## Arquitectura del Sistema

### Componentes Principales

1. **GameRecommender**: Sistema principal de recomendaciones
   - Carga modelo entrenado (.pkl)
   - Prepara datos de entrada
   - Genera predicciones y recomendaciones

2. **FeedbackSystem**: Sistema de feedback y almacenamiento
   - Guarda predicciones realizadas
   - Recopila feedback de usuarios
   - Almacena perfiles para reentrenamiento

3. **ModelMonitoring**: Sistema de monitoreo
   - Calcula métricas de rendimiento
   - Detecta drift del modelo
   - Genera alertas de reentrenamiento

### Flujo de Datos

```
Usuario → Input → GameRecommender → Predicción → Usuario
                         ↓
                  FeedbackSystem
                         ↓
                  Almacenamiento
                         ↓
                  ModelMonitoring
                         ↓
              ¿Reentrenamiento necesario?
```

## Almacenamiento de Datos

### Estructura de Archivos

```
data/
├── feedback/
│   ├── user_feedback.jsonl          # Feedback de usuarios
│   ├── prediction_history.jsonl     # Historial de predicciones
│   └── analyzed_profiles.jsonl      # Perfiles para reentrenamiento
├── gaming_behavior_processed.csv    # Dataset original
└── retraining_data.csv             # Datos para reentrenamiento
```

### Formato de Registros

#### Predicción
```json
{
  "timestamp": "2025-01-15T10:30:00",
  "session_id": "session_20250115_103000_1234",
  "player_data": {...},
  "predicted_style": "Competitive",
  "confidence": 0.85,
  "probabilities": {...}
}
```

#### Feedback
```json
{
  "timestamp": "2025-01-15T10:35:00",
  "session_id": "session_20250115_103000_1234",
  "rating": 4,
  "prediction_correct": true,
  "recommendations_helpful": true,
  "comments": "Muy acertado"
}
```

## Dependencias

### Python Packages
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
joblib>=1.2.0
streamlit>=1.28.0
plotly>=5.14.0
```

## Configuración de Producción

### Variables de Entorno
```bash
MODEL_PATH=models/best_model.pkl
METADATA_PATH=models/model_metadata.pkl
FEEDBACK_DIR=data/feedback
LOG_LEVEL=INFO
```

### Configuración de Servidor
- Puerto: 8501 (Streamlit default)
- Memoria recomendada: 2GB
- CPU: 2 cores mínimo

## Mantenimiento

### Ciclo de Actualización Recomendado

1. **Diario**: Recopilación de feedback
2. **Semanal**: Revisión de métricas de monitoreo
3. **Mensual**: Evaluación de necesidad de reentrenamiento
4. **Trimestral**: Reentrenamiento del modelo (si es necesario)

### Métricas Clave a Monitorear

- Accuracy de predicciones
- Tasa de feedback
- Confianza promedio
- Drift del modelo

## Contacto y Soporte

Desarrollado por:
- Laura Rivera (8-969-1184)
- Marco Rodríguez (8-956-932)
- David Tao (8-961-1083)

Curso: Sistemas Inteligentes - UTP 2025
"""
    
    return doc


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar sistemas
    feedback_sys = FeedbackSystem()
    monitor = ModelMonitoring(feedback_sys)
    
    print("=" * 60)
    print("SISTEMA DE FEEDBACK Y MONITOREO")
    print("=" * 60)
    
    # Ejemplo: Guardar una predicción
    player_data = {
        'playtime_hours': 150,
        'sessions_per_week': 10,
        'win_rate': 0.65
    }
    
    prediction_result = {
        'predicted_style': 'Competitive',
        'confidence': 0.85,
        'probabilities': {
            'Competitive': 0.85,
            'Aggressive': 0.10,
            'Strategic': 0.05
        }
    }
    
    session_id = feedback_sys.save_prediction(player_data, prediction_result)
    print(f"\n✅ Predicción guardada: {session_id}")
    
    # Ejemplo: Guardar feedback
    feedback_data = {
        'rating': 4,
        'prediction_correct': True,
        'recommendations_helpful': True,
        'comments': 'Las recomendaciones fueron muy útiles'
    }
    
    feedback_sys.save_feedback(session_id, feedback_data)
    
    # Verificar si se necesita reentrenamiento
    needs_retrain, reason, metrics = monitor.check_retraining_needed()
    print(f"\n¿Necesita reentrenamiento? {needs_retrain}")
    print(f"Razón: {reason}")
    
    # Generar documentación
    doc = generate_documentation()
    with open('DOCUMENTATION.md', 'w', encoding='utf-8') as f:
        f.write(doc)
    print("\n✅ Documentación generada: DOCUMENTATION.md")
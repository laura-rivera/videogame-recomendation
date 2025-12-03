
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

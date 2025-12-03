# 🎮 Sistema Inteligente de Recomendación para Videojuegos

## 📋 Tabla de Contenidos
- [Descripción General](#descripción-general)
- [Características Principales](#características-principales)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Guía de Uso](#guía-de-uso)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Mantenimiento y Monitoreo](#mantenimiento-y-monitoreo)
- [Solución de Problemas](#solución-de-problemas)
- [Créditos](#créditos)

---

## 🎯 Descripción General

El **Sistema Inteligente de Recomendación para Videojuegos** es una plataforma avanzada de análisis de comportamiento de jugadores que utiliza técnicas de Machine Learning para:

- **Clasificar automáticamente** a los jugadores en 5 estilos de juego distintos
- **Generar recomendaciones personalizadas** basadas en patrones de comportamiento
- **Predecir riesgo de abandono** (churn) y proporcionar estrategias de retención
- **Optimizar la experiencia del jugador** mediante insights basados en datos

### 🏆 Estilos de Juego Identificados

| Estilo | Características Principales |
|--------|---------------------------|
| **Casual** | Bajo compromiso, juego relajado, dificultad baja |
| **Aggressive** | Alto PvP, combate directo, orientado a la acción |
| **Explorer** | Alto completismo, descubrimiento de contenido, logros |
| **Competitive** | Máximo rendimiento, alta tasa de victoria, jugador premium |
| **Strategic** | Planificación táctica, dificultad alta, juego eficiente |

---

## ✨ Características Principales

### 🔍 Análisis Inteligente
- Clasificación con **91. 47% de precisión** (Gradient Boosting)
- Análisis de **18 métricas clave** de comportamiento
- Visualizaciones interactivas con gráficos radar y comparativas

### 💡 Recomendaciones Justificadas
- Motor de reglas basado en desviación métrica
- Explicaciones transparentes para cada sugerencia
- Priorización automática (Alta/Media/Baja)

### 📊 Panel de Administración
- Monitoreo de precisión del modelo en tiempo real
- Sistema de feedback de usuarios
- Alertas automáticas de reentrenamiento
- Exportación de reportes analíticos

### 🔄 Ciclo MLOps Completo
- Reentrenamiento automático basado en feedback
- Versionado de modelos con rollback
- Detección de drift de datos y concepto

---

## 💻 Requisitos del Sistema

### Software Necesario
- **Python**: 3.8 o superior
- **Espacio en disco**: Mínimo 500 MB
- **RAM**: Recomendado 2 GB
- **Navegador**: Chrome, Firefox, Edge o Safari (versión reciente)

### Dependencias Principales
```
pandas>=2.0.3
numpy>=1.24.3
scikit-learn>=1.3.0
streamlit>=1.28.0
plotly>=5.16.1
joblib>=1.3.2
```

---

## 🚀 Instalación

### Paso 1: Clonar el Repositorio
```bash
git clone https://github.com/laura-rivera/videogame-recomendation. git
cd videogame-recomendation
```

### Paso 2: Crear Entorno Virtual (Recomendado)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias
```bash
pip install -r requirements.txt
```

### Paso 4: Verificar Archivos Necesarios
Asegúrese de que existan estos archivos críticos:
```
models/
├── best_model.pkl          # Modelo entrenado
└── model_metadata.pkl      # Metadatos del modelo

data/
├── gaming_behavior_processed.csv  # Dataset procesado
└── feedback/               # Carpeta para feedback (se crea automáticamente)
```

---

## 📖 Guía de Uso

### Iniciar la Aplicación

```bash
streamlit run app. py
```

La interfaz se abrirá automáticamente en `http://localhost:8501`

### 🎮 Pestaña 1: Perfil del Jugador

#### **Opción A: Entrada Manual**
1. Haga clic en **"📝 Ingresar Perfil Manualmente"**
2. Ajuste los controles deslizantes para configurar el perfil:
   - **Horas de juego**: Tiempo total invertido
   - **Sesiones por semana**: Frecuencia de juego
   - **Tasa de victoria**: Porcentaje de victorias
   - **Partidas PvP**: Cantidad de combates competitivos
   - Y más... 
3. Presione **"🔍 ANALIZAR Y GENERAR RECOMENDACIONES"**

#### **Opción B: Carga desde JSON**
1. Haga clic en **"📁 Cargar Perfil desde JSON"**
2. Suba un archivo con esta estructura:
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
3. Presione el botón de análisis

#### **Interpretación de Resultados**

##### Métricas Principales
- **Estilo Predicho**: Clasificación del jugador
- **Confianza**: Precisión de la predicción (0-100%)
- **Engagement**: Nivel de compromiso calculado
- **Habilidad**: Nivel de destreza estimado

##### Gráfico Radar
- **Línea Verde (Tu Perfil)**: Tus métricas actuales
- **Línea Azul (Promedio)**: Perfil óptimo de tu estilo
- **Diferencias**: Áreas de mejora potencial

##### Recomendaciones
Cada recomendación incluye:
- **📝 Justificación**: Por qué se genera
- **💥 Impacto Esperado**: Beneficio de seguirla
- **🎯 Acción Recomendada**: Pasos concretos

#### **Sistema de Feedback**
Después del análisis, califique:
1. **Calificación general** (1-5 estrellas)
2.  **¿Predicción correcta?** (Sí/No/No estoy seguro)
3. **¿Recomendaciones útiles?** (Sí/No/Parcialmente)
4. **Comentarios adicionales** (opcional)

Su feedback mejora el sistema para futuros usuarios.

---

### 📈 Pestaña 2: Estadísticas del Sistema

Visualice métricas globales:
- **Especificaciones técnicas** del modelo
- **Distribución de estilos** en el dataset
- **Métricas de rendimiento**: Precisión, Precision, Recall

---

### ❓ Pestaña 3: Ayuda

Consulte:
- Definición de métricas clave
- Información sobre el proyecto
- Datos de contacto de los desarrolladores

---

### ⚙️ Pestaña 4: Panel de Administración

> **Nota**: Requiere contraseña de administrador (predeterminada: `admin123`)

#### Funcionalidades Administrativas

##### **Monitoreo del Sistema**
- **Predicciones Totales**: Cantidad de análisis realizados
- **Tasa de Feedback**: Porcentaje de usuarios que opinaron
- **Accuracy Actual**: Precisión validada por usuarios
- **Confianza Promedio**: Seguridad de las predicciones

##### **Estado del Modelo**
El sistema detecta automáticamente si necesita reentrenamiento por:
- Caída de precisión (<75%)
- Drift significativo (>10% de variación)
- Alta tasa de predicciones de baja confianza (>30%)

##### **Exportación de Datos**
- **Datos de Reentrenamiento**: Perfiles confirmados para actualizar el modelo
- **Reporte Completo**: Análisis detallado en formato JSON

---

## 🏗️ Arquitectura del Sistema

### Componentes Principales

```
┌─────────────────────────────────────────┐
│         Interfaz Streamlit (app.py)     │
│  - Input de datos                       │
│  - Visualizaciones                      │
│  - Panel de administración              │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│   Recommender System (recommender_system.py)  │
│  - Carga del modelo                     │
│  - Predicción de estilo                 │
│  - Generación de recomendaciones        │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│   Feedback System (feedback_system.py)  │
│  - Almacenamiento de predicciones       │
│  - Registro de feedback                 │
│  - Preparación para reentrenamiento     │
└─────────────────────────────────────────┘
```

### Flujo de Datos

```
Usuario → Input → Preprocessor → Modelo ML → Predicción
                                                  ↓
                                        Recomendaciones
                                                  ↓
                                             Feedback
                                                  ↓
                                        FeedbackSystem
                                                  ↓
                                    ModelMonitoring ←→ Alertas
                                                  ↓
                                        Reentrenamiento
```

---

## 🔧 Mantenimiento y Monitoreo

### Ciclo de Actualización Recomendado

| Frecuencia | Actividad | Responsable |
|-----------|-----------|-------------|
| **Diario** | Revisión de feedback nuevo | Admin |
| **Semanal** | Verificación de métricas de monitoreo | Admin |
| **Mensual** | Evaluación de necesidad de reentrenamiento | Equipo Técnico |
| **Trimestral** | Reentrenamiento del modelo (si es necesario) | Data Scientist |

### Reentrenamiento Manual

Si el sistema recomienda reentrenar:

1. **Exportar datos** desde el panel de administración
2.  **Ejecutar el script**:
```bash
python retrain_model.py
```
3. **Verificar** que se creó un backup en `models/backup/`
4. **Reiniciar** la aplicación para cargar el nuevo modelo

### Estructura de Almacenamiento

```
data/feedback/
├── user_feedback.jsonl          # Calificaciones de usuarios
├── prediction_history.jsonl     # Historial de predicciones
└── analyzed_profiles.jsonl      # Perfiles para reentrenamiento
```

Cada archivo `. jsonl` contiene una línea JSON por registro.

---

## 🐛 Solución de Problemas

### Error: "No se pudo cargar el modelo"

**Causa**: Archivo de modelo corrupto o ruta incorrecta

**Solución**:
1.  Verifique que existe `models/best_model.pkl`
2.  Actualice las rutas en `recommender_system.py` (líneas 19-20):
```python
model_path=r"ruta/completa/a/best_model.pkl",
metadata_path=r"ruta/completa/a/model_metadata.pkl"
```

### Error: "ModuleNotFoundError"

**Causa**: Dependencias no instaladas

**Solución**:
```bash
pip install -r requirements. txt --upgrade
```

### La interfaz no se abre

**Causa**: Puerto 8501 ocupado

**Solución**:
```bash
streamlit run app.py --server.port 8502
```

### Predicciones inconsistentes

**Causa**: Modelo necesita reentrenamiento

**Solución**:
1.  Vaya al **Panel de Administración**
2. Verifique el **Estado del Modelo**
3. Si es necesario, ejecute `python retrain_model.py`

---

## 📊 Métricas Clave Explicadas

| Métrica | Descripción | Rango Óptimo |
|---------|-------------|--------------|
| **playtime_hours** | Tiempo total de juego | Varía por estilo |
| **win_rate** | Tasa de victoria | 0. 4 - 0.7 (40-70%) |
| **engagement_score** | Nivel de compromiso calculado | > 2.0 |
| **skill_level** | Habilidad estimada | > 50 |
| **pvp_matches** | Partidas competitivas | Varía por estilo |
| **achievements_unlocked** | Logros completados | > 50 |

### Cálculo de Métricas Derivadas

```python
# Engagement Score
engagement_score = (sessions_per_week * avg_session_length * 10) / 100

# Skill Level  
skill_level = (win_rate * 50) + (difficulty_level * 5)

# KD Ratio
kd_ratio = pvp_matches / max(death_count, 1)
```

---

## 📁 Estructura del Proyecto

```
videogame-recomendation/
│
├── app.py                          # Aplicación principal Streamlit
├── recommender_system.py           # Motor de recomendaciones
├── feedback_system.py              # Sistema de feedback y monitoreo
├── train_model.py                  # Script de entrenamiento
├── preprocessing.py                # Preprocesamiento de datos
├── eda_analysis.py                 # Análisis exploratorio
├── requirements.txt                # Dependencias
│
├── models/
│   ├── best_model.pkl              # Modelo entrenado
│   ├── model_metadata.pkl          # Metadatos
│   └── backup/                     # Versiones anteriores
│
├── data/
│   ├── gaming_behavior_raw.csv     # Dataset original
│   ├── gaming_behavior_processed.csv # Dataset procesado
│   └── feedback/                   # Datos de feedback
│
└── visualizations/                 # Gráficos generados
```

---

## 🔒 Seguridad y Privacidad

### Datos Procesados
- Todos los datos son **pseudonimizados** (solo `player_id`)
- No se almacena información personal identificable (PII)
- Cumplimiento con estándares de privacidad de datos

### Credenciales de Admin
> ⚠️ **IMPORTANTE**: Cambie la contraseña predeterminada en producción

Edite en `app.py` (línea ~708):
```python
if password == "TU_CONTRASEÑA_SEGURA":  # Cambiar "admin123"
```

---

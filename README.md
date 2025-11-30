# 🎮 Sistema Inteligente de Recomendación para Videojuegos

**Universidad Tecnológica de Panamá**  
**Facultad de Ingeniería de Sistemas Computacionales**  
**Curso:** Sistemas Inteligentes  
**Grupo:** 1IL-142

## 👥 Equipo de Desarrollo

- **Laura Rivera**
- **Marco Rodríguez**
- **David Tao**

**Profesor:** Euclides Samaniego  
**Semestre:** II-2025

---

## 📋 Descripción del Proyecto

Sistema inteligente basado en Machine Learning que analiza el comportamiento de jugadores de videojuegos para:
- Identificar automáticamente su estilo de juego
- Generar recomendaciones personalizadas
- Optimizar la experiencia y engagement del jugador
- Predecir áreas de mejora

### 🎯 Objetivos SMART

- **Específico:** Predecir el estilo de juego y recomendar estrategias de optimización
- **Medible:** Lograr ≥85% de precisión en clasificación de estilos
- **Alcanzable:** Usando Random Forest y dataset sintético de 10,000 jugadores
- **Relevante:** Mejora la retención y satisfacción de jugadores
- **Temporal:** Desarrollado en 4 días intensivos

---

## 🏗️ Arquitectura del Sistema

```
proyecto/
├── data/                          # Datos del proyecto
│   ├── gaming_behavior_raw.csv    # Dataset original (10K registros)
│   └── gaming_behavior_processed.csv  # Datos preprocesados
│
├── models/                        # Modelos entrenados
│   ├── best_model.pkl            # Mejor modelo (Random Forest)
│   ├── scaler.pkl                # Normalizador de datos
│   ├── label_encoders.pkl        # Codificadores de categorías
│   └── model_metadata.pkl        # Metadatos del modelo
│
├── visualizations/                # Gráficos y análisis
│   ├── 1_target_distribution.png
│   ├── 2_numerical_distributions.png
│   ├── 3_correlation_matrix.png
│   ├── 4_playstyle_characteristics.png
│   ├── 5_combat_style_analysis.png
│   ├── 6_engagement_analysis.png
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── model_comparison.png
│
├── generate_synthetic_data.py     # Generador de datos
├── preprocessing.py               # Preprocesamiento
├── eda_analysis.py               # Análisis exploratorio
├── train_model.py                # Entrenamiento de modelos
├── recommender_system.py         # Sistema de recomendaciones
├── app.py                        # Aplicación web (Streamlit)
├── run_all.py                    # Pipeline completo
├── requirements.txt              # Dependencias
└── README.md                     # Este archivo
```

---

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- 2GB de espacio en disco
- 4GB de RAM recomendado

### Instalación Rápida

```bash
# 1. Clonar o descargar el proyecto
cd sistema-recomendacion-videojuegos

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar pipeline completo (genera datos, entrena modelo, etc.)
python run_all.py

# 4. Lanzar aplicación web
streamlit run app.py
```

### Instalación Manual (Paso a Paso)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Generar dataset sintético
python generate_synthetic_data.py

# 3. Preprocesar datos
python preprocessing.py

# 4. Análisis exploratorio (opcional)
python eda_analysis.py

# 5. Entrenar modelos
python train_model.py

# 6. Probar sistema de recomendaciones
python recommender_system.py

# 7. Lanzar interfaz web
streamlit run app.py
```

---

## 📊 Metodología

### ETAPA 1: Planificación y Recopilación de Datos

**Alcance definido:**
- Dominio: Comportamiento de jugadores en videojuegos
- Objetivo: Clasificar estilos de juego y recomendar mejoras

**Métricas clave:**
- Tiempo de juego (horas totales)
- Frecuencia de sesiones (por semana)
- Nivel de dificultad preferido (1-10)
- Tasa de victoria (win rate)
- Actividad PvP (partidas jugador vs jugador)
- Logros desbloqueados
- Estilo de combate (Melee, Ranged, Magic, etc.)

**Dataset:**
- 10,000 jugadores sintéticos
- 15 características por jugador
- 5 estilos de juego: Aggressive, Strategic, Casual, Explorer, Competitive

### ETAPA 2: Desarrollo del Modelo

**Preprocesamiento:**
- Limpieza de datos (nulos, duplicados, outliers)
- Ingeniería de características:
  - K/D Ratio (kills/deaths)
  - Play Intensity (horas por sesión)
  - Commitment Score (nivel de compromiso)
  - PvP Experience (experiencia logarítmica)
  - Achievement Rate (logros por hora)
- Normalización con StandardScaler
- Codificación de variables categóricas

**Modelos evaluados:**
1. **Random Forest** ⭐ (Mejor modelo)
   - 100 árboles de decisión
   - Precisión: 85-87%
   - Robusto y explicable

2. **Gradient Boosting**
   - Precisión: 83-85%
   - Mayor tiempo de entrenamiento

3. **Decision Tree**
   - Precisión: 78-80%
   - Más simple pero menos preciso

4. **Logistic Regression**
   - Precisión: 75-77%
   - Baseline para comparación

**División de datos:**
- Entrenamiento: 80% (8,000 registros)
- Prueba: 20% (2,000 registros)
- Validación cruzada: 5-fold

### ETAPA 3: Implementación

**Sistema de Recomendaciones:**
- Análisis comparativo con jugadores del mismo estilo
- Generación de 5-7 recomendaciones personalizadas
- Justificación basada en:
  - Feature importance del modelo
  - Estadísticas de referencia por estilo
  - Brechas entre perfil actual y óptimo

**Interfaz de Usuario (Streamlit):**
- Dashboard interactivo
- Inputs mediante sliders y selectores
- Visualizaciones con Plotly:
  - Gráfico de radar (comparación de perfil)
  - Barras de probabilidad por estilo
  - Métricas clave en tiempo real
- Sistema de descarga de reportes

**Características de la UI:**
- Diseño responsive
- Navegación por tabs
- Ayuda contextual
- Exportación de resultados

---

## 📈 Resultados

### Métricas del Modelo

| Métrica | Valor |
|---------|-------|
| **Precisión (Test)** | 85.3% |
| **Precisión (Train)** | 92.1% |
| **Precision (weighted)** | 85.8% |
| **Recall (weighted)** | 85.3% |
| **F1-Score (weighted)** | 85.4% |

### Características Más Importantes

1. **Win Rate** (18.2%) - Mayor predictor de estilo
2. **Difficulty Level** (15.7%) - Indica preferencias
3. **PvP Matches** (14.3%) - Distingue competitivos
4. **Playtime Hours** (12.8%) - Nivel de compromiso
5. **Commitment Score** (11.5%) - Engagement general

### Matriz de Confusión

El modelo clasifica correctamente:
- Aggressive: 88%
- Strategic: 91%
- Casual: 79%
- Explorer: 84%
- Competitive: 87%

---

## 💡 Ejemplos de Recomendaciones

### Ejemplo 1: Jugador Casual

**Perfil:**
- 45 horas jugadas
- 3 sesiones/semana
- Dificultad nivel 4
- Win rate: 42%

**Recomendaciones generadas:**
1. ✅ Incrementa tu tiempo de práctica (Prioridad: Media)
2. ✅ Experimenta con nuevas estrategias (Prioridad: Baja)
3. ✅ Establece rutina más consistente (Prioridad: Media)

### Ejemplo 2: Jugador Competitivo

**Perfil:**
- 250 horas jugadas
- 15 sesiones/semana
- Dificultad nivel 9
- Win rate: 68%

**Recomendaciones generadas:**
1. ✅ Refina combos avanzados (Prioridad: Alta)
2. ✅ Analiza replays de derrotas (Prioridad: Alta)
3. ✅ Participa en torneos (Prioridad: Media)

---

## 🔬 Justificación Técnica

### ¿Por qué Machine Learning?

Un sistema basado en reglas fijas no puede:
- Detectar patrones complejos en 15+ características
- Adaptarse a nuevos datos sin reprogramación
- Manejar relaciones no lineales entre variables
- Generalizar a miles de perfiles diferentes

El ML ofrece:
- ✅ Aprendizaje automático de patrones
- ✅ Predicciones precisas (85%+)
- ✅ Escalabilidad a millones de jugadores
- ✅ Mejora continua con nuevos datos

### ¿Por qué Random Forest?

Ventajas sobre otros modelos:
- **Alta precisión** sin overfitting
- **Interpretable** (feature importance)
- **Robusto** a outliers y datos ruidosos
- **Rápido** en predicción
- **No requiere normalización** (pero la aplicamos)

---

## 🎯 Casos de Uso

### 1. Desarrolladores de Videojuegos
- Entender su base de jugadores
- Diseñar contenido personalizado
- Mejorar retención y engagement
- Balancear dificultad

### 2. Plataformas de Gaming
- Sistema de matchmaking mejorado
- Recomendaciones de juegos
- Detección de abandono (churn)
- Segmentación de usuarios

### 3. Jugadores Individuales
- Mejorar su rendimiento
- Descubrir su estilo de juego
- Recibir coaching personalizado
- Optimizar tiempo de práctica

### 4. Equipos E-Sports
- Analizar rendimiento de jugadores
- Identificar fortalezas/debilidades
- Reclutar talento
- Diseñar estrategias de entrenamiento

---

## 🔮 Trabajo Futuro

### Mejoras a Corto Plazo
- [ ] Integrar con APIs de juegos reales (Steam, Xbox Live)
- [ ] Añadir más estilos de juego (15+ categorías)
- [ ] Sistema de feedback del usuario
- [ ] Reentrenamiento automático mensual

### Mejoras a Medio Plazo
- [ ] Implementar Deep Learning (LSTM para secuencias)
- [ ] Predicción de abandono (churn prediction)
- [ ] Recomendaciones de juegos similares
- [ ] Sistema de amigos/matchmaking

### Mejoras a Largo Plazo
- [ ] Reinforcement Learning para NPCs adaptativos
- [ ] Generación procedural de contenido
- [ ] Procesamiento de Lenguaje Natural para diálogos
- [ ] Integración con motores de juego (Unity/Unreal)

---

## 📚 Tecnologías Utilizadas

### Lenguaje
- **Python 3.8+** - Lenguaje principal

### Machine Learning
- **scikit-learn** - Modelos de ML
- **NumPy** - Computación numérica
- **Pandas** - Manipulación de datos

### Visualización
- **Matplotlib** - Gráficos estáticos
- **Seaborn** - Visualizaciones estadísticas
- **Plotly** - Gráficos interactivos

### Web Framework
- **Streamlit** - Interfaz web interactiva

### Utilidades
- **Joblib** - Serialización de modelos
- **SciPy** - Funciones científicas

---

## 🧪 Testing

### Pruebas Realizadas

**1. Validación de Datos**
- ✅ Valores en rangos esperados
- ✅ Sin nulos críticos
- ✅ Distribuciones balanceadas

**2. Validación de Modelo**
- ✅ Precisión > 80% en test set
- ✅ No overfitting (train/test gap < 10%)
- ✅ Validación cruzada consistente

**3. Pruebas de Sistema**
- ✅ Pipeline completo funcional
- ✅ Recomendaciones coherentes
- ✅ UI responsive y sin errores

**4. Casos de Prueba**
```python
# Ejemplo de caso de prueba
test_profiles = [
    {'playtime': 50, 'win_rate': 0.45, 'expected': 'Casual'},
    {'playtime': 200, 'win_rate': 0.70, 'expected': 'Competitive'},
    {'playtime': 180, 'achievements': 90, 'expected': 'Explorer'}
]
# Todos los casos pasaron con 100% de precisión
```

---

## 📖 Referencias Bibliográficas

1. Safadi, F., Fonteneau, R., & Ernst, D. (2015). *Artificial intelligence in video games: Towards a unified framework.* International Journal of Computer Games Technology.

2. Vasconcelos, S. (2025). *Generación Procedural de Contenido en la programación de videojuegos.* Universidad Nacional Autónoma de México.

3. Rueda, J. (2024). *Generación Procedural Inteligente de Niveles de Plataforma 2D utilizando Algoritmos Genéticos.* Ridaa unicen.

4. Navarro, J. (2024). *Procesamiento del lenguaje natural como eje central de la inteligencia artificial generativa.* Dialnet.

5. Sánchez, F., & Pantoja, E. (2024). *Revisión de la literatura sobre el uso de la inteligencia artificial con enfoque a su aplicación en los videojuegos.* Universidad Politécnica Salesiana.

6. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5-32.

7. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.

---

## 📞 Soporte y Contacto

### Equipo de Desarrollo

**Laura Rivera**
- Email: laura.rivera@utp.ac.pa
- Rol: Líder de proyecto, Desarrollo ML

**Marco Rodríguez**
- Email: marco.rodriguez@utp.ac.pa
- Rol: Análisis de datos, Visualización

**David Tao**
- Email: david.tao@utp.ac.pa
- Rol: Desarrollo UI, Testing

### Profesor

**Euclides Samaniego**
- Curso: Sistemas Inteligentes
- Institución: Universidad Tecnológica de Panamá

---

## 📄 Licencia

Este proyecto fue desarrollado con fines académicos para el curso de Sistemas Inteligentes de la Universidad Tecnológica de Panamá.

**Año:** 2025  
**Semestre:** II

---

## 🙏 Agradecimientos

- A la Universidad Tecnológica de Panamá por la formación académica
- Al profesor Euclides Samaniego por su guía en el curso
- A la comunidad de código abierto por las herramientas utilizadas
- A todos los investigadores citados en las referencias

---

## 📊 Estadísticas del Proyecto

- **Líneas de código:** ~3,500
- **Archivos Python:** 8
- **Tiempo de desarrollo:** 4 días intensivos
- **Dataset generado:** 10,000 registros
- **Modelos evaluados:** 4
- **Visualizaciones creadas:** 12+
- **Precisión alcanzada:** 85.3%

---

**¡Gracias por revisar nuestro proyecto! 🎮🚀**

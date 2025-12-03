# ============================================
# setup_deployment.py
# Script para configurar el entorno de producción
# ============================================

import os
import json
from pathlib import Path
import shutil

def setup_directory_structure():
    """Crea la estructura de directorios necesaria"""
    directories = [
        'data/feedback',
        'data/backup',
        'models/backup',
        'logs',
        'visualizations',
        'docs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directorio creado: {directory}")

def create_config_file():
    """Crea archivo de configuración"""
    config = {
        "model": {
            "path": "models/best_model.pkl",
            "metadata_path": "models/model_metadata.pkl",
            "backup_dir": "models/backup"
        },
        "data": {
            "feedback_dir": "data/feedback",
            "training_data": "data/gaming_behavior_processed.csv",
            "retraining_data": "data/retraining_data.csv"
        },
        "monitoring": {
            "accuracy_threshold": 0.75,
            "drift_threshold": 0.1,
            "min_feedback_for_retrain": 100,
            "check_interval_days": 7
        },
        "app": {
            "port": 8501,
            "host": "0.0.0.0",
            "admin_password": "CHANGE_ME_IN_PRODUCTION"
        }
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Archivo de configuración creado: config.json")

def create_requirements_file():
    """Crea archivo requirements.txt actualizado"""
    requirements = """
# Core dependencies
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.1

# Visualization
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.14.1

# Web framework
streamlit==1.28.0

# Data handling
imbalanced-learn==0.11.0

# Utilities
python-dateutil==2.8.2
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements.strip())
    
    print("✅ Archivo requirements.txt creado")

def create_dockerfile():
    """Crea Dockerfile para containerización"""
    dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Exponer puerto
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content.strip())
    
    print("✅ Dockerfile creado")

def create_docker_compose():
    """Crea docker-compose.yml"""
    docker_compose_content = """
version: '3.8'

services:
  gaming-recommender:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose_content.strip())
    
    print("✅ docker-compose.yml creado")

def create_gitignore():
    """Crea .gitignore apropiado"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Data
data/feedback/
data/backup/
data/retraining_data.csv
*.csv

# Models (commit only the initial model)
models/backup/

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    
    print("✅ .gitignore creado")

def create_readme():
    """Crea README.md completo"""
    readme_content = """
# 🎮 Sistema Inteligente de Recomendación para Videojuegos

Sistema de machine learning que analiza el comportamiento de jugadores y genera recomendaciones personalizadas.

## 📋 Características

- ✅ Predicción de estilo de juego (5 categorías)
- ✅ Recomendaciones personalizadas basadas en IA
- ✅ Sistema de feedback para mejora continua
- ✅ Monitoreo automático del rendimiento del modelo
- ✅ Reentrenamiento automático cuando es necesario
- ✅ Panel de administración con métricas en tiempo real

## 🚀 Instalación

### Opción 1: Instalación Local

```bash
# Clonar repositorio
git clone <tu-repo>
cd videogame-recommendation

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\\Scripts\\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar estructura
python setup_deployment.py

# Ejecutar aplicación
streamlit run app.py
```

### Opción 2: Docker

```bash
# Construir y ejecutar con docker-compose
docker-compose up -d

# Ver logs
docker-compose logs -f
```

## 📊 Estructura del Proyecto

```
videogame-recommendation/
├── app.py                      # Aplicación principal Streamlit
├── recommender_system.py       # Sistema de recomendaciones
├── feedback_system.py          # Sistema de feedback y monitoreo
├── train_model.py             # Script de entrenamiento
├── retrain_model.py           # Script de reentrenamiento automático
├── setup_deployment.py        # Script de configuración
├── config.json                # Configuración del sistema
├── requirements.txt           # Dependencias Python
├── Dockerfile                 # Configuración Docker
├── docker-compose.yml         # Orquestación Docker
├── data/
│   ├── feedback/              # Datos de feedback de usuarios
│   ├── backup/                # Backups de datos
│   └── gaming_behavior_processed.csv
├── models/
│   ├── best_model.pkl         # Modelo entrenado
│   ├── model_metadata.pkl     # Metadatos del modelo
│   └── backup/                # Backups de modelos
├── logs/                      # Logs del sistema
└── docs/                      # Documentación adicional
```

## 🔧 Configuración

Edita `config.json` para ajustar:

- Umbrales de accuracy y drift
- Intervalos de verificación
- Rutas de archivos
- Configuración del servidor

## 📈 Uso

### Para Usuarios

1. Accede a la aplicación en `http://localhost:8501`
2. Completa tu perfil de jugador
3. Obtén tu análisis y recomendaciones
4. Proporciona feedback para mejorar el sistema

### Para Administradores

1. Ve a la pestaña "⚙️ Admin"
2. Ingresa contraseña de administrador
3. Monitorea métricas del sistema
4. Exporta datos para reentrenamiento
5. Genera reportes

## 🔄 Mantenimiento

### Reentrenamiento Manual

```bash
python retrain_model.py
```

### Reentrenamiento Automático (Cron)

Agrega a crontab para ejecutar mensualmente:

```bash
0 0 1 * * cd /path/to/project && python retrain_model.py >> logs/retrain.log 2>&1
```

### Backup de Datos

```bash
# Backup automático cada semana
0 0 * * 0 tar -czf backup_$(date +\%Y\%m\%d).tar.gz data/ models/
```

## 📊 Métricas de Rendimiento

El sistema actual tiene:
- **Accuracy:** ~85%
- **Precision:** ~83%
- **Recall:** ~82%
- **F1-Score:** ~82%

## 🛠️ Tecnologías

- **ML:** scikit-learn, pandas, numpy
- **Frontend:** Streamlit, Plotly
- **Deployment:** Docker, Docker Compose

## 👥 Autores

- **Laura Rivera** (8-969-1184)
- **Marco Rodríguez** (8-956-932)
- **David Tao** (8-961-1083)

**Curso:** Sistemas Inteligentes - UTP 2025

## 📄 Licencia

Este proyecto fue desarrollado con fines educativos.

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Soporte

Para problemas o preguntas, abre un issue en el repositorio.
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content.strip())
    
    print("✅ README.md creado")

def create_changelog():
    """Crea CHANGELOG.md"""
    changelog_content = """
# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

## [1.0.0] - 2025-01-XX

### Agregado
- Sistema completo de recomendaciones con ML
- Interface web con Streamlit
- Sistema de feedback de usuarios
- Monitoreo automático del modelo
- Sistema de reentrenamiento automático
- Panel de administración
- Documentación técnica completa
- Containerización con Docker
- Scripts de deployment

### Características
- 5 estilos de juego detectables
- Predicción con ~85% accuracy
- Recomendaciones personalizadas
- Visualizaciones interactivas
- Exportación de reportes

## [0.1.0] - 2025-01-XX

### Agregado
- Entrenamiento inicial del modelo
- EDA básico
- Preprocesamiento de datos
"""
    
    with open('CHANGELOG.md', 'w') as f:
        f.write(changelog_content.strip())
    
    print("✅ CHANGELOG.md creado")

def create_startup_script():
    """Crea script de inicio"""
    startup_content = """#!/bin/bash
# startup.sh - Script para iniciar el sistema

echo "🎮 Iniciando Sistema Inteligente de Recomendación..."

# Verificar estructura de directorios
python setup_deployment.py

# Verificar que el modelo existe
if [ ! -f "models/best_model.pkl" ]; then
    echo "❌ Error: Modelo no encontrado. Ejecuta train_model.py primero."
    exit 1
fi

# Iniciar aplicación
echo "✅ Iniciando aplicación Streamlit..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
"""
    
    with open('startup.sh', 'w') as f:
        f.write(startup_content.strip())
    
    # Hacer ejecutable
    os.chmod('startup.sh', 0o755)
    
    print("✅ startup.sh creado")

def main():
    """Ejecuta la configuración completa"""
    print("=" * 60)
    print("CONFIGURACIÓN DE DEPLOYMENT")
    print("=" * 60)
    print()
    
    setup_directory_structure()
    print()
    
    create_config_file()
    create_requirements_file()
    create_dockerfile()
    create_docker_compose()
    create_gitignore()
    create_readme()
    create_changelog()
    create_startup_script()
    
    print()
    print("=" * 60)
    print("✅ CONFIGURACIÓN COMPLETADA")
    print("=" * 60)
    print()
    print("Próximos pasos:")
    print("1. Revisar y ajustar config.json")
    print("2. Cambiar contraseña de admin en config.json")
    print("3. Ejecutar: python train_model.py (si no tienes modelo)")
    print("4. Ejecutar: streamlit run app.py")
    print("5. O usar Docker: docker-compose up -d")

if __name__ == "__main__":
    main()
"""
Entrenamiento y Evaluación de Modelos de Machine Learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

GAMING_COLORS = {
    'primary': '#00FF88',      # Neon Green (Cyber)
    'secondary': '#0088FF',    # Electric Blue
    'accent': '#FF0088',       # Neon Pink
    'dark_bg': '#0A0A12',      # Dark Blue-Black
    'card_bg': '#1A1A2E',      # Dark Card
    'grid': '#2A2A3E',         # Grid Lines
    'text': '#E0E0FF',         # Light Blue Text
    'highlight': '#FFFFFF',    # Blanco para máximo contraste
    'light_text': '#F0F0FF',   # Texto más claro
    'medium_bg': '#252540',    # Fondo intermedio para mejor contraste
}

# Gaming palettes
PLAYSTYLE_PALETTE = ['#00FF88', '#0088FF', '#FF0088', '#FFAA00', '#AA00FF']

class GameplayModelTrainer:
    """Clase para entrenar y evaluar modelos de clasificación"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.results = {}
    
    def _apply_gaming_style(self, fig, title):
        """Apply gaming theme to figure - matches EDA style"""
        fig.patch.set_facecolor(GAMING_COLORS['dark_bg'])
        if hasattr(fig, 'axes'):
            for ax in fig.axes:
                ax.set_facecolor(GAMING_COLORS['card_bg'])
                
                # Mejorar contraste de etiquetas
                ax.xaxis.label.set_color(GAMING_COLORS['light_text'])
                ax.yaxis.label.set_color(GAMING_COLORS['light_text'])
                ax.tick_params(colors=GAMING_COLORS['light_text'], labelsize=10)
                
                # Mejorar título
                if hasattr(ax, 'title'):
                    ax.title.set_color(GAMING_COLORS['primary'])
                    ax.title.set_fontweight('bold')
                    ax.title.set_fontsize(12)
                
                # Style grid
                ax.grid(True, alpha=0.3, color=GAMING_COLORS['grid'])
                
                # Style spines
                for spine in ax.spines.values():
                    spine.set_color(GAMING_COLORS['primary'])
                    spine.set_linewidth(1.5)
        
        # Use text-only title to match EDA style
        fig.suptitle(title, fontsize=16, color=GAMING_COLORS['primary'], 
                    fontweight='bold')
        
    def load_data(self, filepath='data/gaming_behavior_processed.csv'):
        """Carga los datos preprocesados"""
        print("Cargando datos procesados...")
        df = pd.read_csv(filepath)
        print(f"Datos cargados: {df.shape}")
        return df
    
    def prepare_data(self, df, target_col='playstyle', test_size=0.2):
        """Prepara los datos para entrenamiento"""
        print(f"\nPreparando datos para entrenamiento...")

        # Características a usar - UPDATED to match your actual columns
        feature_cols = [
            'playtime_hours', 'sessions_per_week', 'avg_session_length',
            'achievements_unlocked', 'difficulty_level', 'win_rate',
            'pvp_matches', 'death_count', 'engagement_score', 'skill_level',
            'kd_ratio', 'play_intensity', 'commitment_score',
            'pvp_experience', 'achievement_rate', 'last_login_days_ago',
            'combat_style_encoded', 'premium_user'
        ]

        # Verify all feature columns exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"ADVERTENCIA: Columnas faltantes: {missing_cols}")
            # Remove missing columns
            feature_cols = [col for col in feature_cols if col not in missing_cols]
        
        print(f"Usando {len(feature_cols)} características")
        print(f"Características: {feature_cols}")

        X = df[feature_cols]
        y = df[target_col]

        # Store feature names for later use
        self.feature_names = feature_cols

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print(f"Datos divididos:")
        print(f"   - Entrenamiento: {self.X_train.shape}")
        print(f"   - Prueba: {self.X_test.shape}")
        print(f"   - Features: {len(feature_cols)}")

        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_random_forest(self, optimize=False):
        """Entrena un modelo Random Forest"""
        print("\nEntrenando Random Forest...")
        
        if optimize:
            print("   Optimizando hiperparámetros...")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=3, 
                                      scoring='accuracy', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            model = grid_search.best_estimator_
            print(f"   Mejores parámetros: {grid_search.best_params_}")
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = model
        print("Random Forest entrenado")
        
        return model
    
    def train_gradient_boosting(self):
        """Entrena un modelo Gradient Boosting"""
        print("\nEntrenando Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['Gradient Boosting'] = model
        print("Gradient Boosting entrenado")
        
        return model
    
    def train_decision_tree(self):
        """Entrena un modelo de Árbol de Decisión"""
        print("\nEntrenando Decision Tree...")
        
        model = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['Decision Tree'] = model
        print("Decision Tree entrenado")
        
        return model
    
    def train_logistic_regression(self):
        """Entrena una Regresión Logística"""
        print("\nEntrenando Logistic Regression...")
        
        model = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['Logistic Regression'] = model
        print("Logistic Regression entrenada")
        
        return model
    
    def evaluate_model(self, model, model_name):
        """Evalúa un modelo específico"""
        print(f"\nEvaluando {model_name}...")
        
        # Predicciones
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Métricas
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        
        # Calcular métricas adicionales (promedio weighted para multiclase)
        precision = precision_score(self.y_test, y_pred_test, average='weighted')
        recall = recall_score(self.y_test, y_pred_test, average='weighted')
        f1 = f1_score(self.y_test, y_pred_test, average='weighted')
        
        # Guardar resultados
        self.results[model_name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred_test
        }
        
        print(f"   Train Accuracy: {train_accuracy:.4f}")
        print(f"   Test Accuracy:  {test_accuracy:.4f}")
        print(f"   Precision:      {precision:.4f}")
        print(f"   Recall:         {recall:.4f}")
        print(f"   F1-Score:       {f1:.4f}")
        
        return self.results[model_name]
    
    def compare_models(self):
        """Compara todos los modelos entrenados"""
        print("\n" + "=" * 60)
        print("COMPARACIÓN DE MODELOS")
        print("=" * 60)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Train Acc': [self.results[m]['train_accuracy'] for m in self.results],
            'Test Acc': [self.results[m]['test_accuracy'] for m in self.results],
            'Precision': [self.results[m]['precision'] for m in self.results],
            'Recall': [self.results[m]['recall'] for m in self.results],
            'F1-Score': [self.results[m]['f1_score'] for m in self.results]
        })
        
        comparison_df = comparison_df.sort_values('Test Acc', ascending=False)
        
        print("\nResultados:")
        print(comparison_df.to_string(index=False))
        
        # Seleccionar el mejor modelo
        best_idx = comparison_df['Test Acc'].idxmax()
        self.best_model_name = comparison_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\nMejor modelo: {self.best_model_name}")
        print(f"  Test Accuracy: {comparison_df.loc[best_idx, 'Test Acc']:.4f}")
        
        return comparison_df
    
    def plot_feature_importance(self):
        """Visualiza la importancia de características con estilo gaming"""
        if not hasattr(self.best_model, 'feature_importances_'):
            print("El mejor modelo no tiene feature_importances_")
            return
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=True).tail(15)  # Changed to horizontal bars
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Gaming-style horizontal bars
        bars = ax.barh(importance_df['feature'], importance_df['importance'],
                    color=GAMING_COLORS['secondary'], alpha=0.8,
                    edgecolor=GAMING_COLORS['primary'], linewidth=2)
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center',
                    fontweight='bold', color=GAMING_COLORS['highlight'],
                    bbox=dict(boxstyle="round,pad=0.3",
                            facecolor=GAMING_COLORS['dark_bg'],
                            edgecolor=GAMING_COLORS['primary'],
                            alpha=0.8))
        
        ax.set_xlabel('Importancia', color=GAMING_COLORS['light_text'], fontsize=12)
        ax.set_ylabel('Feature', color=GAMING_COLORS['light_text'], fontsize=12)
        
        self._apply_gaming_style(fig, f'TOP 15 FEATURES - {self.best_model_name.upper()}')
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight',
                    facecolor=GAMING_COLORS['dark_bg'])
        print("\nFeature importance guardada: visualizations/feature_importance.png")
        
        return importance_df
    
    def plot_confusion_matrix(self):
        """Visualiza la matriz de confusión con estilo gaming"""
        y_pred = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Gaming-style heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=self.best_model.classes_,
                yticklabels=self.best_model.classes_, 
                ax=ax, 
                annot_kws={
                    'color': 'white', 
                    'weight': 'bold', 
                    'size': 12,
                    'ha': 'center',
                    'va': 'center'
                },
                cbar_kws={
                    "shrink": 0.8,
                    "label": "Cantidad"
                })
        
        ax.set_ylabel('Valor Real', color=GAMING_COLORS['light_text'], fontsize=12, fontweight='bold')
        ax.set_xlabel('Valor Predicho', color=GAMING_COLORS['light_text'], fontsize=12, fontweight='bold')
        
        # Style the heatmap to match gaming theme with better contrast
        ax.tick_params(colors=GAMING_COLORS['light_text'], labelsize=10)
        
        # Style colorbar for better visibility
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color=GAMING_COLORS['light_text'])
        cbar.outline.set_edgecolor(GAMING_COLORS['primary'])
        cbar.ax.set_ylabel('Cantidad', color=GAMING_COLORS['light_text'], fontsize=10)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), 
                color=GAMING_COLORS['light_text'], 
                fontsize=9)
        
        # Apply gaming style but remove spines from the main ax to avoid cell borders
        fig.patch.set_facecolor(GAMING_COLORS['dark_bg'])
        ax.set_facecolor(GAMING_COLORS['card_bg'])
        
        # Style labels and ticks
        ax.xaxis.label.set_color(GAMING_COLORS['light_text'])
        ax.yaxis.label.set_color(GAMING_COLORS['light_text'])
        ax.tick_params(colors=GAMING_COLORS['light_text'], labelsize=10)
        
        # Style title
        ax.title.set_color(GAMING_COLORS['primary'])
        ax.title.set_fontweight('bold')
        ax.title.set_fontsize(12)
        
        # Style grid - remove for heatmap
        ax.grid(False)
        
        # Remove spines from main axes (this fixes the cell border issue)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Use text-only title to match EDA style
        fig.suptitle(f'MATRIZ DE CONFUSIÓN - {self.best_model_name.upper()}', 
                    fontsize=16, color=GAMING_COLORS['primary'], fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight',
                    facecolor=GAMING_COLORS['dark_bg'])
        print("Matriz de confusión guardada: visualizations/confusion_matrix.png")
        
        return cm
    
    def plot_model_comparison(self, comparison_df):
        """Visualiza comparación de modelos con estilo gaming"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Gráfico 1: Comparación de métricas - Gaming bars
        metrics = ['Test Acc', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(comparison_df))
        width = 0.2
        
        colors = [GAMING_COLORS['primary'], GAMING_COLORS['secondary'], 
                GAMING_COLORS['accent'], '#FFAA00']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            bars = axes[0].bar(x + i*width, comparison_df[metric], width, 
                    label=metric, alpha=0.8, color=color,
                    edgecolor='white', linewidth=1.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom',
                            fontweight='bold', color=GAMING_COLORS['highlight'],
                            bbox=dict(boxstyle="round,pad=0.2",
                                    facecolor=GAMING_COLORS['dark_bg'],
                                    edgecolor=GAMING_COLORS['primary'],
                                    alpha=0.8))
        
        axes[0].set_xlabel('Modelo', color=GAMING_COLORS['light_text'], fontsize=12)
        axes[0].set_ylabel('Score', color=GAMING_COLORS['light_text'], fontsize=12)
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(comparison_df['Model'], rotation=45, ha='right',
                            color=GAMING_COLORS['light_text'])
        axes[0].legend(facecolor=GAMING_COLORS['card_bg'], 
                    edgecolor=GAMING_COLORS['primary'],
                    labelcolor=GAMING_COLORS['text'])
        axes[0].grid(axis='y', alpha=0.3, color=GAMING_COLORS['grid'])
        
        # Gráfico 2: Train vs Test Accuracy - Gaming scatter
        scatter = axes[1].scatter(comparison_df['Train Acc'], comparison_df['Test Acc'], 
                    s=200, alpha=0.8, c=range(len(comparison_df)), 
                    cmap='viridis', edgecolors='white', linewidth=2)
        
        for idx, row in comparison_df.iterrows():
            axes[1].annotate(row['Model'], 
                        (row['Train Acc'], row['Test Acc']),
                        xytext=(8, 8), textcoords='offset points',
                        color=GAMING_COLORS['highlight'],
                        bbox=dict(boxstyle="round,pad=0.3",
                                facecolor=GAMING_COLORS['dark_bg'],
                                edgecolor=GAMING_COLORS['primary'],
                                alpha=0.8))
        
        # Línea de referencia (overfitting)
        max_acc = max(comparison_df['Train Acc'].max(), comparison_df['Test Acc'].max())
        axes[1].plot([0, max_acc], [0, max_acc], '--', alpha=0.7, 
                    color=GAMING_COLORS['accent'], linewidth=2, label='Perfect fit')
        
        axes[1].set_xlabel('Train Accuracy', color=GAMING_COLORS['light_text'], fontsize=12)
        axes[1].set_ylabel('Test Accuracy', color=GAMING_COLORS['light_text'], fontsize=12)
        axes[1].legend(facecolor=GAMING_COLORS['card_bg'], 
                    edgecolor=GAMING_COLORS['primary'],
                    labelcolor=GAMING_COLORS['text'])
        axes[1].grid(alpha=0.3, color=GAMING_COLORS['grid'])
        
        self._apply_gaming_style(fig, "COMPARACIÓN DE MODELOS - ANÁLISIS DE RENDIMIENTO")
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight',
                    facecolor=GAMING_COLORS['dark_bg'])
        print("Comparación de modelos guardada: visualizations/model_comparison.png")
    
    def save_best_model(self, path='models/'):
        """Guarda el mejor modelo"""
        os.makedirs(path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar modelo
        model_path = os.path.join(path, 'best_model.pkl')
        joblib.dump(self.best_model, model_path)
        
        # Guardar metadatos
        metadata = {
            'model_name': self.best_model_name,
            'test_accuracy': self.results[self.best_model_name]['test_accuracy'],
            'precision': self.results[self.best_model_name]['precision'],
            'recall': self.results[self.best_model_name]['recall'],
            'f1_score': self.results[self.best_model_name]['f1_score'],
            'feature_names': self.feature_names,
            'classes': list(self.best_model.classes_),
            'timestamp': timestamp
        }
        
        metadata_path = os.path.join(path, 'model_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        print(f"\nMejor modelo guardado:")
        print(f"   - Modelo: {model_path}")
        print(f"   - Metadata: {metadata_path}")
    
    def generate_classification_report(self):
        """Genera reporte detallado de clasificación"""
        y_pred = self.best_model.predict(self.X_test)
        
        print("\n" + "=" * 60)
        print(f"REPORTE DE CLASIFICACIÓN - {self.best_model_name}")
        print("=" * 60)
        print(classification_report(self.y_test, y_pred))
    
    def train_and_evaluate_all(self):
        """Pipeline completo de entrenamiento y evaluación"""
        print("\n" + "=" * 60)
        print("ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
        print("=" * 60)

        # Entrenar modelos
        self.train_random_forest(optimize=False)
        self.train_gradient_boosting()
        self.train_decision_tree()
        self.train_logistic_regression()

        # Evaluar todos los modelos
        for model_name, model in self.models.items():
            self.evaluate_model(model, model_name)

        # Comparar modelos
        comparison_df = self.compare_models()

        # Visualizaciones
        print("\nGenerando visualizaciones...")
        self.plot_feature_importance()
        self.plot_confusion_matrix()
        self.plot_model_comparison(comparison_df)

        # Reporte detallado
        self.generate_classification_report()

        # Guardar mejor modelo
        self.save_best_model()

        # NO guardar scaler
        print("\n⚠ Características usadas en su escala original - sin scaler")

        print("\n" + "=" * 60)
        print("Entrenamiento completado.")
        print("=" * 60)

def cross_validate_model(trainer, model, model_name, cv=5):
    """Perform cross-validation"""
    from sklearn.model_selection import cross_val_score
    
    print(f"\nCross-Validation for {model_name}:")
    cv_scores = cross_val_score(model, trainer.X_train, trainer.y_train, cv=cv, scoring='accuracy')
    
    print(f"  CV Scores: {cv_scores}")
    print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

def analyze_prediction_confidence(trainer, num_samples=10):
    """Analyze prediction confidence on test set"""
    probabilities = trainer.best_model.predict_proba(trainer.X_test)
    predictions = trainer.best_model.predict(trainer.X_test)
    
    print(f"\nPrediction Confidence Analysis (first {num_samples} samples):")
    for i in range(min(num_samples, len(predictions))):
        true_label = trainer.y_test.iloc[i]
        pred_label = predictions[i]
        confidence = probabilities[i].max()
        pred_class = trainer.best_model.classes_[probabilities[i].argmax()]
        
        print(f"Sample {i}: True={true_label}, Pred={pred_label}, Confidence={confidence:.3f}")
        
        # Show top 3 predictions
        top3_idx = np.argsort(probabilities[i])[-3:][::-1]
        top3 = [(trainer.best_model.classes_[idx], probabilities[i][idx]) for idx in top3_idx]
        print(f"  Top 3: {top3}")

def analyze_errors(trainer):
    """Analyze misclassified samples"""
    y_pred = trainer.best_model.predict(trainer.X_test)
    y_true = trainer.y_test
    
    # Confusion matrix analysis
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*50)
    print("DETAILED ERROR ANALYSIS")
    print("="*50)
    
    # Show confusion matrix
    cm_df = pd.DataFrame(cm, 
                        index=trainer.best_model.classes_,
                        columns=trainer.best_model.classes_)
    print("\nConfusion Matrix:")
    print(cm_df)
    
    # Analyze specific error patterns
    error_patterns = {}
    for true_class in trainer.best_model.classes_:
        for pred_class in trainer.best_model.classes_:
            if true_class != pred_class:
                count = cm_df.loc[true_class, pred_class]
                if count > 0:
                    error_patterns[f"{true_class}→{pred_class}"] = count
    
    print("\nMost Common Error Patterns:")
    for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {pattern}: {count} samples")

def test_with_custom_data(self):
    """Prueba con perfiles personalizados"""
    import pandas as pd
    import numpy as np

    print("\n" + "="*60)
    print("PRUEBA DE PERFILES")
    print("="*60)

    # Perfiles de prueba
    test_profiles = [
        {
            'name': 'Casual Profile',
            'data': {
                'playtime_hours': 35.0,
                'sessions_per_week': 4.0,
                'avg_session_length': 2.0,
                'achievements_unlocked': 25.0,
                'difficulty_level': 4.0,
                'win_rate': 0.42,
                'pvp_matches': 25.0,
                'death_count': 107.0,
                'engagement_score': 0.36,
                'skill_level': 34.0,
                'kd_ratio': 0.231,
                'play_intensity': 2.917,
                'commitment_score': 49.5,
                'pvp_experience': 2.773,
                'achievement_rate': 0.514,
                'last_login_days_ago': 5.0,
                'combat_style_encoded': 0,
                'premium_user': 0
            },
            'expected': 'Casual'
        },
        {
            'name': 'Competitive Profile',
            'data': {
                'playtime_hours': 203.0,
                'sessions_per_week': 15.0,
                'avg_session_length': 3.5,
                'achievements_unlocked': 78.0,
                'difficulty_level': 9.0,
                'win_rate': 0.71,
                'pvp_matches': 406.0,
                'death_count': 320.0,
                'engagement_score': 5.88,
                'skill_level': 81.0,
                'kd_ratio': 1.188,
                'play_intensity': 3.929,
                'commitment_score': 252.0,
                'pvp_experience': 5.944,
                'achievement_rate': 0.418,
                'last_login_days_ago': 1.0,
                'combat_style_encoded': 3,
                'premium_user': 1
            },
            'expected': 'Competitive'
        }
    ]

    correct = 0
    total = len(test_profiles)

    for profile in test_profiles:
        print(f"\n{'='*50}")
        print(f"Perfil: {profile['name']}")
        print(f"{'='*50}")

        # Crear DataFrame con valores originales
        profile_df = pd.DataFrame([profile['data']])

        # Mostrar valores (en escala original)
        print("\nValores:")
        print(f"  Playtime: {profile['data']['playtime_hours']}h")
        print(f"  PvP Matches: {profile['data']['pvp_matches']}")
        print(f"  Win Rate: {profile['data']['win_rate']:.2f}")
        print(f"  KD Ratio: {profile['data']['kd_ratio']:.3f}")

        try:
            # PREDECIR DIRECTAMENTE
            prediction = self.best_model.predict(profile_df)[0]
            probabilities = self.best_model.predict_proba(profile_df)[0]
            confidence = probabilities.max()

            is_correct = prediction == profile['expected']
            if is_correct:
                correct += 1
                result_symbol = '✓'
            else:
                result_symbol = '✗'

            print(f"\nResultado:")
            print(f"  Esperado:   {profile['expected']}")
            print(f"  Predicción: {prediction} {result_symbol}")
            print(f"  Confianza:  {confidence:.3f}")

            # Mostrar todas las probabilidades
            print("\nProbabilidades por clase:")
            for i, (class_name, prob) in enumerate(zip(self.best_model.classes_, probabilities)):
                print(f"  {class_name}: {prob:.3f}")

        except Exception as e:
            print(f"Error procesando perfil: {e}")
            print(f"Características disponibles: {list(profile_df.columns)}")
            continue

    print(f"\n{'='*60}")
    print(f"RESUMEN DE PRUEBAS")
    print(f"{'='*60}")
    print(f"Correctas: {correct}/{total} ({(correct/total)*100:.1f}%)")
    print(f"{'='*60}")

def calculate_kd_ratio(pvp_matches, death_count):
    """Calculate KD ratio from PvP matches and death count"""
    if death_count == 0:
        return 0
    return pvp_matches / death_count

def test_boundary_cases(trainer):
    """Test edge cases and boundary conditions"""
    
    boundary_profiles = [
        # Casual-Explorer boundary (medium playtime, medium achievements)
        {
            'name': 'Casual-Explorer Boundary',
            'data': [80.0, 6.0, 3.0, 45.0, 5.0, 0.45, 40.0, 120.0, 2.0, 48.0,
                    0.33, 2.67, 90.0, 2.0, 0.56, 6, 1, 0],
            'description': 'Between Casual and Explorer'
        },
        # Aggressive-Competitive boundary (high PvP, medium-high stats)
        {
            'name': 'Aggressive-Competitive Boundary',
            'data': [160.0, 12.0, 3.8, 70.0, 8.0, 0.63, 300.0, 320.0, 4.5, 70.0,
                    0.94, 4.0, 200.0, 5.0, 0.47, 2, 2, 1],
            'description': 'Between Aggressive and Competitive'
        },
        # Explorer-Strategic boundary (balanced profile)
        {
            'name': 'Explorer-Strategic Boundary',
            'data': [170.0, 8.5, 4.8, 82.0, 7.0, 0.58, 110.0, 190.0, 4.0, 65.0,
                    0.58, 4.25, 170.0, 4.5, 0.49, 3, 3, 1],
            'description': 'Between Explorer and Strategic'
        }
    ]
    
    print("\n" + "="*50)
    print("BOUNDARY CASE TESTING")
    print("="*50)
    
    for profile in boundary_profiles:
        sample_df = pd.DataFrame([profile['data']], columns=trainer.feature_names)
        prediction = trainer.best_model.predict(sample_df)[0]
        probabilities = trainer.best_model.predict_proba(sample_df)[0]
        
        print(f"\n{profile['name']}:")
        print(f"  Description: {profile['description']}")
        print(f"  Predicted: {prediction}")
        
        # Show top 3 probabilities
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3 = [(trainer.best_model.classes_[idx], probabilities[idx]) for idx in top3_idx]
        print(f"  Top 3 Predictions:")
        for playstyle, prob in top3:
            print(f"    {playstyle}: {prob:.3f}")

def analyze_feature_patterns(trainer):
    """Analyze how features differentiate playstyles based on your stats"""
    
    print("\n" + "="*50)
    print("FEATURE PATTERN ANALYSIS")
    print("="*50)
    
    patterns = {
        'Casual': "Low playtime (35h), low PvP (25), low engagement",
        'Aggressive': "Medium playtime (120h), high PvP (204), high deaths (295)",
        'Explorer': "High playtime (186h), high achievements (89), medium PvP",
        'Competitive': "Highest playtime (203h), highest PvP (406), premium user",
        'Strategic': "High playtime (151h), high difficulty (8), balanced stats"
    }
    
    for playstyle, pattern in patterns.items():
        print(f"  {playstyle}: {pattern}")
    
    print(f"\nKey Differentiators:")
    print(f"  • Playtime: Casual (35h) → Competitive (203h)")
    print(f"  • PvP Matches: Casual (25) → Competitive (406)") 
    print(f"  • Sessions/Week: Casual (4) → Competitive (15)")
    print(f"  • Win Rate: Casual (42%) → Competitive (71%)")

# Add these helper functions
def calculate_kd_ratio(pvp_matches, death_count):
    """Calculate KD ratio from PvP matches and death count"""
    if death_count == 0:
        return 0
    return pvp_matches / death_count

def calculate_play_intensity(playtime_hours, sessions_per_week):
    """Calculate play intensity"""
    if sessions_per_week == 0:
        return 0
    return playtime_hours / sessions_per_week

def calculate_commitment_score(playtime_hours, achievements_unlocked):
    """Calculate commitment score"""
    return (playtime_hours * achievements_unlocked) / 100

def calculate_pvp_experience(pvp_matches, win_rate):
    """Calculate PvP experience"""
    return pvp_matches * win_rate

def calculate_achievement_rate(achievements_unlocked, playtime_hours):
    """Calculate achievement rate"""
    if playtime_hours == 0:
        return 0
    return achievements_unlocked / playtime_hours

def main():
    """Función principal"""
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELOS ML")
    print("=" * 60)
    
    # Crear trainer
    trainer = GameplayModelTrainer()
    
    # Cargar datos
    df = trainer.load_data()
    
    # Preparar datos - this now stores X_train, X_test, etc. in the trainer
    trainer.prepare_data(df)
    
    # Entrenar y evaluar todos los modelos
    trainer.train_and_evaluate_all()
    
    # ADD THESE TESTS:
    print("\n" + "="*60)
    print("POST-TRAINING MODEL TESTING")
    print("="*60)
    
    # 1. Manual testing
    test_with_custom_data(trainer)

    # 2. Test boundary cases
    test_boundary_cases(trainer)
    
    # 3. Analyze feature patterns
    analyze_feature_patterns(trainer)
    
    # 2. Cross-validation
    cross_validate_model(trainer, trainer.best_model, trainer.best_model_name)
    
    # 3. Confidence analysis
    analyze_prediction_confidence(trainer)
    
    # 4. Error analysis
    analyze_errors(trainer)
    
    plt.show()

if __name__ == "__main__":
    main()
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
        
        # Características a usar
        feature_cols = [
            'playtime_hours', 'sessions_per_week', 'avg_session_length',
            'achievements_unlocked', 'difficulty_level', 'win_rate',
            'pvp_matches', 'death_count', 'engagement_score', 'skill_level',
            'kd_ratio', 'play_intensity', 'commitment_score',
            'pvp_experience', 'achievement_rate', 'last_login_days_ago',
            'combat_style_encoded', 'premium_user'
        ]
        
        # Filtrar solo columnas que existen
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_names = feature_cols
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Dividir datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=self.best_model.classes_,
                yticklabels=self.best_model.classes_, 
                ax=ax, annot_kws={'color': 'white', 'weight': 'bold', 'size': 12},
                cbar_kws={"shrink": 0.8})
        
        ax.set_ylabel('Valor Real', color=GAMING_COLORS['light_text'], fontsize=12)
        ax.set_xlabel('Valor Predicho', color=GAMING_COLORS['light_text'], fontsize=12)
        
        # Style the heatmap to match gaming theme
        ax.tick_params(colors=GAMING_COLORS['light_text'])
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color=GAMING_COLORS['text'])
        cbar.outline.set_edgecolor(GAMING_COLORS['primary'])
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=GAMING_COLORS['text'])
        
        self._apply_gaming_style(fig, f'MATRIZ DE CONFUSIÓN - {self.best_model_name.upper()}')
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
        
        print("\n" + "=" * 60)
        print("Entrenamiento completado.")
        print("=" * 60)

def main():
    """Función principal"""
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELOS ML")
    print("=" * 60)
    
    # Crear trainer
    trainer = GameplayModelTrainer()
    
    # Cargar datos
    df = trainer.load_data()
    
    # Preparar datos
    trainer.prepare_data(df)
    
    # Entrenar y evaluar todos los modelos
    trainer.train_and_evaluate_all()
    
    plt.show()

if __name__ == "__main__":
    main()
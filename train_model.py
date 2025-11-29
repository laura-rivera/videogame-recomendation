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
        """Visualiza la importancia de características"""
        if not hasattr(self.best_model, 'feature_importances_'):
            print("El mejor modelo no tiene feature_importances_")
            return
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.barplot(data=importance_df.head(15), y='feature', x='importance', 
                   palette='viridis', ax=ax)
        ax.set_title(f'Top 15 Features - {self.best_model_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Importancia')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance guardada: visualizations/feature_importance.png")
        
        return importance_df
    
    def plot_confusion_matrix(self):
        """Visualiza la matriz de confusión"""
        y_pred = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.best_model.classes_,
                   yticklabels=self.best_model.classes_, ax=ax)
        ax.set_title(f'Matriz de Confusión - {self.best_model_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Valor Real')
        ax.set_xlabel('Valor Predicho')
        
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Matriz de confusión guardada: visualizations/confusion_matrix.png")
        
        return cm
    
    def plot_model_comparison(self, comparison_df):
        """Visualiza comparación de modelos"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico 1: Comparación de métricas
        metrics = ['Test Acc', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(comparison_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            axes[0].bar(x + i*width, comparison_df[metric], width, 
                       label=metric, alpha=0.8)
        
        axes[0].set_xlabel('Modelo')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Comparación de Métricas por Modelo', fontweight='bold')
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Gráfico 2: Train vs Test Accuracy
        axes[1].scatter(comparison_df['Train Acc'], comparison_df['Test Acc'], 
                       s=200, alpha=0.6, c=range(len(comparison_df)), cmap='viridis')
        
        for idx, row in comparison_df.iterrows():
            axes[1].annotate(row['Model'], 
                           (row['Train Acc'], row['Test Acc']),
                           xytext=(5, 5), textcoords='offset points')
        
        # Línea de referencia (overfitting)
        max_acc = max(comparison_df['Train Acc'].max(), comparison_df['Test Acc'].max())
        axes[1].plot([0, max_acc], [0, max_acc], 'r--', alpha=0.5, label='Perfect fit')
        
        axes[1].set_xlabel('Train Accuracy')
        axes[1].set_ylabel('Test Accuracy')
        axes[1].set_title('Train vs Test Accuracy', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
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
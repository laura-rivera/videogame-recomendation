"""
Análisis Exploratorio de Datos (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class EDAAnalyzer:
    """Clase para realizar análisis exploratorio de datos"""
    
    def __init__(self, df):
        self.df = df
        self.figures = []
        
    def basic_info(self):
        """Muestra información básica del dataset"""
        print("=" * 60)
        print("INFORMACIÓN BÁSICA DEL DATASET")
        print("=" * 60)
        print(f"\nDimensiones: {self.df.shape[0]} filas x {self.df.shape[1]} columnas")
        print(f"\nColumnas:")
        print(self.df.dtypes)
        print(f"\nValores nulos:")
        print(self.df.isnull().sum())
        print(f"\nEstadísticas descriptivas:")
        print(self.df.describe())
        
    def plot_target_distribution(self):
        """Visualiza la distribución de la variable objetivo"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico de barras
        playstyle_counts = self.df['playstyle'].value_counts()
        axes[0].bar(playstyle_counts.index, playstyle_counts.values, 
                    color=sns.color_palette("husl", len(playstyle_counts)))
        axes[0].set_title('Distribución de Estilos de Juego', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Estilo de Juego')
        axes[0].set_ylabel('Cantidad de Jugadores')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Agregar valores en las barras
        for i, v in enumerate(playstyle_counts.values):
            axes[0].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Gráfico de pastel
        axes[1].pie(playstyle_counts.values, labels=playstyle_counts.index, 
                    autopct='%1.1f%%', startangle=90,
                    colors=sns.color_palette("husl", len(playstyle_counts)))
        axes[1].set_title('Proporción de Estilos de Juego', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_numerical_distributions(self):
        """Visualiza distribuciones de variables numéricas clave"""
        numerical_vars = ['playtime_hours', 'sessions_per_week', 
                         'avg_session_length', 'achievements_unlocked',
                         'difficulty_level', 'win_rate']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, var in enumerate(numerical_vars):
            axes[idx].hist(self.df[var].dropna(), bins=30, 
                          color='steelblue', edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribución: {var}', fontweight='bold')
            axes[idx].set_xlabel(var.replace('_', ' ').title())
            axes[idx].set_ylabel('Frecuencia')
            
            # Añadir línea de media
            mean_val = self.df[var].mean()
            axes[idx].axvline(mean_val, color='red', linestyle='--', 
                            linewidth=2, label=f'Media: {mean_val:.2f}')
            axes[idx].legend()
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_correlation_matrix(self):
        """Visualiza matriz de correlación"""
        # Seleccionar solo columnas numéricas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Crear heatmap
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title('Matriz de Correlación - Variables Numéricas', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_playstyle_characteristics(self):
        """Analiza características por estilo de juego"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Tiempo de juego por estilo
        self.df.boxplot(column='playtime_hours', by='playstyle', ax=axes[0, 0])
        axes[0, 0].set_title('Tiempo de Juego por Estilo', fontweight='bold')
        axes[0, 0].set_xlabel('Estilo de Juego')
        axes[0, 0].set_ylabel('Horas de Juego')
        plt.sca(axes[0, 0])
        plt.xticks(rotation=45)
        
        # 2. Win rate por estilo
        self.df.boxplot(column='win_rate', by='playstyle', ax=axes[0, 1])
        axes[0, 1].set_title('Tasa de Victoria por Estilo', fontweight='bold')
        axes[0, 1].set_xlabel('Estilo de Juego')
        axes[0, 1].set_ylabel('Win Rate')
        plt.sca(axes[0, 1])
        plt.xticks(rotation=45)
        
        # 3. Dificultad por estilo
        playstyle_difficulty = self.df.groupby('playstyle')['difficulty_level'].mean().sort_values()
        axes[1, 0].barh(playstyle_difficulty.index, playstyle_difficulty.values,
                       color=sns.color_palette("viridis", len(playstyle_difficulty)))
        axes[1, 0].set_title('Nivel de Dificultad Promedio por Estilo', fontweight='bold')
        axes[1, 0].set_xlabel('Dificultad Promedio')
        axes[1, 0].set_ylabel('Estilo de Juego')
        
        # Añadir valores
        for i, v in enumerate(playstyle_difficulty.values):
            axes[1, 0].text(v + 0.1, i, f'{v:.2f}', va='center')
        
        # 4. Logros por estilo
        playstyle_achievements = self.df.groupby('playstyle')['achievements_unlocked'].mean().sort_values()
        axes[1, 1].barh(playstyle_achievements.index, playstyle_achievements.values,
                       color=sns.color_palette("rocket", len(playstyle_achievements)))
        axes[1, 1].set_title('Logros Promedio por Estilo', fontweight='bold')
        axes[1, 1].set_xlabel('Logros Desbloqueados')
        axes[1, 1].set_ylabel('Estilo de Juego')
        
        # Añadir valores
        for i, v in enumerate(playstyle_achievements.values):
            axes[1, 1].text(v + 1, i, f'{v:.1f}', va='center')
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_combat_style_analysis(self):
        """Analiza estilos de combate"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Distribución de estilos de combate
        combat_counts = self.df['combat_style'].value_counts()
        axes[0].bar(combat_counts.index, combat_counts.values,
                   color=sns.color_palette("Set2", len(combat_counts)))
        axes[0].set_title('Distribución de Estilos de Combate', fontweight='bold')
        axes[0].set_xlabel('Estilo de Combate')
        axes[0].set_ylabel('Cantidad de Jugadores')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. Relación entre estilo de combate y estilo de juego
        combat_playstyle = pd.crosstab(self.df['combat_style'], self.df['playstyle'])
        combat_playstyle.plot(kind='bar', stacked=True, ax=axes[1],
                             colormap='tab10')
        axes[1].set_title('Estilos de Combate por Estilo de Juego', fontweight='bold')
        axes[1].set_xlabel('Estilo de Combate')
        axes[1].set_ylabel('Cantidad de Jugadores')
        axes[1].legend(title='Estilo de Juego', bbox_to_anchor=(1.05, 1))
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_engagement_analysis(self):
        """Analiza el engagement de jugadores"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scatter: Tiempo vs Engagement
        scatter = axes[0, 0].scatter(self.df['playtime_hours'], 
                                    self.df['engagement_score'],
                                    c=self.df['skill_level'], 
                                    cmap='viridis', alpha=0.6)
        axes[0, 0].set_title('Tiempo de Juego vs Engagement Score', fontweight='bold')
        axes[0, 0].set_xlabel('Horas de Juego')
        axes[0, 0].set_ylabel('Engagement Score')
        plt.colorbar(scatter, ax=axes[0, 0], label='Nivel de Habilidad')
        
        # 2. Sesiones vs Logros
        axes[0, 1].scatter(self.df['sessions_per_week'], 
                          self.df['achievements_unlocked'],
                          alpha=0.5, color='coral')
        axes[0, 1].set_title('Sesiones por Semana vs Logros', fontweight='bold')
        axes[0, 1].set_xlabel('Sesiones por Semana')
        axes[0, 1].set_ylabel('Logros Desbloqueados')
        
        # 3. Riesgo de abandono
        churn_counts = self.df['churn_risk'].value_counts()
        axes[1, 0].pie(churn_counts.values, labels=churn_counts.index,
                      autopct='%1.1f%%', startangle=90,
                      colors=['lightgreen', 'gold', 'salmon'])
        axes[1, 0].set_title('Distribución de Riesgo de Abandono', fontweight='bold')
        
        # 4. PvP Matches por estilo
        self.df.boxplot(column='pvp_matches', by='playstyle', ax=axes[1, 1])
        axes[1, 1].set_title('Partidas PvP por Estilo de Juego', fontweight='bold')
        axes[1, 1].set_xlabel('Estilo de Juego')
        axes[1, 1].set_ylabel('Partidas PvP')
        plt.sca(axes[1, 1])
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def save_all_figures(self, output_dir='visualizations/'):
        """Guarda todas las figuras generadas"""
        os.makedirs(output_dir, exist_ok=True)
        
        figure_names = [
            'target_distribution',
            'numerical_distributions',
            'correlation_matrix',
            'playstyle_characteristics',
            'combat_style_analysis',
            'engagement_analysis'
        ]
        
        for idx, (fig, name) in enumerate(zip(self.figures, figure_names)):
            filepath = os.path.join(output_dir, f'{idx+1}_{name}.png')
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Guardado: {filepath}")
    
    def generate_report(self):
        """Genera reporte completo del EDA"""
        print("\n" + "=" * 60)
        print("GENERANDO REPORTE DE ANÁLISIS EXPLORATORIO")
        print("=" * 60)
        
        self.basic_info()
        
        print("\nGenerando visualizaciones...")
        self.plot_target_distribution()
        print("- Distribución del target")
        
        self.plot_numerical_distributions()
        print("- Distribuciones numéricas")
        
        self.plot_correlation_matrix()
        print("- Matriz de correlación")
        
        self.plot_playstyle_characteristics()
        print("- Características por estilo")
        
        self.plot_combat_style_analysis()
        print("- Análisis de estilos de combate")
        
        self.plot_engagement_analysis()
        print("- Análisis de engagement")
        
        self.save_all_figures()
        
        print("\n" + "=" * 60)
        print("Reporte completado.")
        print("=" * 60)

def main():
    """Función principal"""
    print("=" * 60)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("=" * 60)
    
    # Cargar datos
    print("\nCargando datos...")
    df = pd.read_csv('data/gaming_behavior_raw.csv')
    print(f"Datos cargados: {df.shape}")
    
    # Crear analizador
    analyzer = EDAAnalyzer(df)
    
    # Generar reporte completo
    analyzer.generate_report()
    
    plt.show()

if __name__ == "__main__":
    main()
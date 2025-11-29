"""
Análisis Exploratorio de Datos (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

GAMING_COLORS = {
    'primary': '#00FF88',      # Neon Green (Cyber)
    'secondary': '#0088FF',    # Electric Blue
    'accent': '#FF0088',       # Neon Pink
    'dark_bg': '#0A0A12',      # Dark Blue-Black
    'card_bg': '#1A1A2E',      # Dark Card
    'grid': '#2A2A3E',         # Grid Lines
    'text': '#E0E0FF',         # Light Blue Text
    'highlight': '#FFFFFF',      # Blanco para máximo contraste
    'light_text': '#F0F0FF',     # Texto más claro
    'medium_bg': '#252540',      # Fondo intermedio para mejor contraste
}

# Gaming palettes for different categories
PLAYSTYLE_PALETTE = ['#00FF88', '#0088FF', '#FF0088', '#FFAA00', '#AA00FF']
COMBAT_PALETTE = ['#FF5555', '#55FF55', '#5555FF', '#FFFF55', '#FF55FF']
RISK_PALETTE = ['#FF4444', '#FFAA44', '#44FF44']  # High, Medium, Low

# Configure style to avoid emoji issues
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use standard font that supports basic symbols
sns.set_style("whitegrid")

class EDAAnalyzer:
    """Clase para realizar análisis exploratorio con tema gaming - FIXED"""
    
    def __init__(self, df):
        self.df = df
        self.figures = []
        
    def _apply_gaming_style(self, fig, title):
        """Apply theme to figure"""
        fig.patch.set_facecolor(GAMING_COLORS['dark_bg'])
        if hasattr(fig, 'axes'):
            for ax in fig.axes:
                ax.set_facecolor(GAMING_COLORS['card_bg'])
                
                # Mejorar contraste de etiquetas
                ax.xaxis.label.set_color(GAMING_COLORS['light_text'])  # Más contraste
                ax.yaxis.label.set_color(GAMING_COLORS['light_text'])  # Más contraste
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
        
        # Use text-only title to avoid emoji issues
        fig.suptitle(title, fontsize=16, color=GAMING_COLORS['primary'], 
                    fontweight='bold')
        
    def basic_info(self):
        """Muestra información básica del dataset"""
        print("=" * 60)
        print("INFORMACIÓN BÁSICA DEL DATASET")
        print("=" * 60)
        print(f"\nDimensiones: {self.df.shape[0]} jugadores x {self.df.shape[1]} atributos")
        print(f"\nColumnas:")
        print(self.df.dtypes)
        print(f"\nValores nulos:")
        print(self.df.isnull().sum())
        print(f"\nEstadísticas descriptivas:")
        print(self.df.describe())
        
    def plot_target_distribution(self):
        """Visualiza la distribución de la variable objetivo"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(GAMING_COLORS['dark_bg'])
        
        # Gráfico de barras con estilo gaming
        playstyle_counts = self.df['playstyle'].value_counts()
        bars = axes[0].bar(playstyle_counts.index, playstyle_counts.values, 
                          color=PLAYSTYLE_PALETTE, edgecolor='white', linewidth=2,
                          alpha=0.8)
        axes[0].set_title('DISTRIBUCIÓN DE ESTILOS DE JUEGO', 
                         fontsize=14, fontweight='bold', color=GAMING_COLORS['primary'])
        axes[0].set_xlabel('Estilo de Juego', color=GAMING_COLORS['light_text'])
        axes[0].set_ylabel('Jugadores', color=GAMING_COLORS['light_text'])
        axes[0].tick_params(axis='x', rotation=15, colors=GAMING_COLORS['text'])
        axes[0].tick_params(axis='y', colors=GAMING_COLORS['text'])
        
        # Add values on bars without emojis
        for i, v in enumerate(playstyle_counts.values):
            axes[0].text(i, v + 30, f'{v}', ha='center', va='bottom',
                        fontweight='bold', 
                        color=GAMING_COLORS['highlight'],
                        bbox=dict(boxstyle="round,pad=0.3",
                                facecolor=GAMING_COLORS['dark_bg'],
                                edgecolor=GAMING_COLORS['primary'],
                                alpha=0.8))
        
        # Gráfico de pastel gaming
        wedges, texts, autotexts = axes[1].pie(playstyle_counts.values, 
                                              labels=playstyle_counts.index, 
                                              autopct='%1.1f%%', startangle=90,
                                              colors=PLAYSTYLE_PALETTE,
                                              textprops={'color': GAMING_COLORS['light_text']})
        
        # Style the pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
            
        axes[1].set_title('PROPORCIÓN DE ESTILOS', 
                         fontsize=14, fontweight='bold', color=GAMING_COLORS['primary'])
        
        self._apply_gaming_style(fig, "ANÁLISIS DE ESTILOS DE JUEGO")
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_numerical_distributions(self):
        """Visualiza distribuciones con tema cyber - FIXED"""
        numerical_vars = ['playtime_hours', 'sessions_per_week', 
                         'avg_session_length', 'achievements_unlocked',
                         'difficulty_level', 'win_rate']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.patch.set_facecolor(GAMING_COLORS['dark_bg'])
        
        for idx, var in enumerate(numerical_vars):
            ax = axes[idx//3, idx%3]
            n, bins, patches = ax.hist(self.df[var].dropna(), bins=30, 
                                     color=GAMING_COLORS['secondary'], 
                                     edgecolor=GAMING_COLORS['primary'],
                                     alpha=0.7, linewidth=1)
            
            # Color gradient for histogram
            for i, patch in enumerate(patches):
                patch.set_facecolor(plt.cm.viridis(i/len(patches)))
            
            ax.set_title(f'{var.replace("_", " ").title()}', 
                        fontweight='bold', color=GAMING_COLORS['primary'])
            ax.set_xlabel(var.replace('_', ' ').title(), color=GAMING_COLORS['light_text'])
            ax.set_ylabel('Frecuencia', color=GAMING_COLORS['light_text'])
            
            # Gaming-style mean line
            mean_val = self.df[var].mean()
            ax.axvline(mean_val, color=GAMING_COLORS['accent'], linestyle='--', 
                      linewidth=3, label=f'AVG: {mean_val:.1f}')
            ax.legend(facecolor=GAMING_COLORS['card_bg'], 
                     edgecolor=GAMING_COLORS['primary'])
        
        self._apply_gaming_style(fig, "DISTRIBUCIÓN DE MÉTRICAS PRINCIPALES")
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_correlation_matrix(self):
        """Matriz de correlación estilo gaming HUD - FIXED"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(16, 14))
        fig.patch.set_facecolor(GAMING_COLORS['dark_bg'])
        
        # FIXED: Remove the mask to show full correlation matrix
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
           cmap='coolwarm', center=0, square=True,  # Cambiado a coolwarm
           linewidths=1, linecolor=GAMING_COLORS['medium_bg'],  # Mejor contraste
           annot_kws={'color': GAMING_COLORS['highlight'], 'weight': 'bold'},  # AÑADIDO
           cbar_kws={"shrink": 0.8, "label": "Correlación"},
           ax=ax) # Removed mask parameter
        
        ax.set_title('MATRIZ DE CORRELACIÓN - RADAR DE MÉTRICAS', 
                    fontsize=18, fontweight='bold', color=GAMING_COLORS['primary'], pad=20)
        
        # Style the heatmap
        ax.tick_params(colors=GAMING_COLORS['light_text'], rotation=45)
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color=GAMING_COLORS['text'])
        cbar.outline.set_edgecolor(GAMING_COLORS['primary'])
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=GAMING_COLORS['text'])
        
        self._apply_gaming_style(fig, "ANÁLISIS DE CORRELACIONES")
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_playstyle_characteristics(self):
        """Características por estilo con diseño gaming - FIXED"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.patch.set_facecolor(GAMING_COLORS['dark_bg'])
        
        # 1. Tiempo de juego - Boxplot gaming
        playstyles_sorted = self.df['playstyle'].unique()
        box_data = [self.df[self.df['playstyle'] == style]['playtime_hours'] 
                   for style in playstyles_sorted]
        
        bp = axes[0, 0].boxplot(box_data, labels=playstyles_sorted,
                       patch_artist=True,
                       boxprops=dict(linewidth=2),
                       medianprops=dict(linewidth=2, color='white'), 
                       whiskerprops=dict(linewidth=2),
                       capprops=dict(linewidth=2),
                       flierprops=dict(marker='o', 
                                     markerfacecolor=GAMING_COLORS['accent'],  
                                     markersize=4,
                                     markeredgecolor=GAMING_COLORS['text'],  
                                     alpha=0.6))
        
        # Style boxplot
        for patch, color in zip(bp['boxes'], PLAYSTYLE_PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color=GAMING_COLORS['text'], linewidth=2)
        
        axes[0, 0].set_title('HORAS DE JUEGO POR ESTILO', 
                           fontweight='bold', color=GAMING_COLORS['primary'])
        axes[0, 0].set_xlabel('Estilo de Juego', color=GAMING_COLORS['light_text'])
        axes[0, 0].set_ylabel('Horas Totales', color=GAMING_COLORS['light_text'])
        axes[0, 0].tick_params(axis='x', rotation=15, colors=GAMING_COLORS['light_text'])
        axes[0, 0].tick_params(axis='y', colors=GAMING_COLORS['light_text'])
        
        # 2. Win rate - Horizontal bars
        playstyle_winrate = self.df.groupby('playstyle')['win_rate'].mean().sort_values()
        bars = axes[0, 1].barh(list(playstyle_winrate.index), playstyle_winrate.values,
                      color=PLAYSTYLE_PALETTE, edgecolor='white', alpha=0.8)

        # Añadir etiquetas mejoradas
        for bar, value in zip(bars, playstyle_winrate.values):
            axes[0, 1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{value:.2f}', ha='left', va='center',
                        fontweight='bold',
                        color=GAMING_COLORS['highlight'],
                        bbox=dict(boxstyle="round,pad=0.3",
                                facecolor=GAMING_COLORS['dark_bg'],
                                edgecolor=GAMING_COLORS['primary'],
                                alpha=0.8))
        
        axes[0, 1].set_title('WIN RATE PROMEDIO', 
                           fontweight='bold', color=GAMING_COLORS['primary'])
        axes[0, 1].set_xlabel('Tasa de Victoria', color=GAMING_COLORS['light_text'])
        axes[0, 1].set_xlim(0, 1)
        
        # 3. Dificultad 
        playstyle_difficulty = self.df.groupby('playstyle')['difficulty_level'].mean()
        # Reindex to match winrate order for consistency
        playstyle_difficulty = playstyle_difficulty.reindex(playstyle_winrate.index)
        
        bars = axes[1, 0].bar(playstyle_difficulty.index, playstyle_difficulty.values,
                     color=PLAYSTYLE_PALETTE, alpha=0.8, edgecolor='white')
        
        for bar, value in zip(bars, playstyle_difficulty.values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, value + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom',
                        fontweight='bold',
                        color=GAMING_COLORS['highlight'],
                        bbox=dict(boxstyle="round,pad=0.3",
                                facecolor=GAMING_COLORS['dark_bg'],
                                edgecolor=GAMING_COLORS['primary'],
                                alpha=0.8))
        axes[1, 0].set_title('NIVEL DE DIFICULTAD', 
                           fontweight='bold', color=GAMING_COLORS['primary'])
        axes[1, 0].set_ylabel('Dificultad Promedio', color=GAMING_COLORS['light_text'])
        axes[1, 0].tick_params(axis='x', rotation=15, colors=GAMING_COLORS['light_text'])
        axes[1, 0].tick_params(axis='y', colors=GAMING_COLORS['text'])
        
        # 4. Logros 
        playstyle_achievements = self.df.groupby('playstyle')['achievements_unlocked'].mean()
        playstyle_achievements = playstyle_achievements.reindex(playstyle_winrate.index)
        
        bars = axes[1, 1].bar(playstyle_achievements.index, playstyle_achievements.values,
                             color=PLAYSTYLE_PALETTE, alpha=0.8, edgecolor='white')
        
        for bar, value in zip(bars, playstyle_achievements.values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, value + 2, 
                        f'{value:.0f}', ha='center', va='bottom',
                        fontweight='bold', 
                        color=GAMING_COLORS['highlight'],  # Color de alto contraste
                        bbox=dict(boxstyle="round,pad=0.3",  # AÑADIDO: fondo
                                facecolor=GAMING_COLORS['dark_bg'],
                                edgecolor=GAMING_COLORS['primary'],
                                alpha=0.8))
        
        axes[1, 1].set_title('LOGROS DESBLOQUEADOS', 
                           fontweight='bold', color=GAMING_COLORS['primary'])
        axes[1, 1].set_ylabel('Logros Promedio', color=GAMING_COLORS['light_text'])
        axes[1, 1].tick_params(axis='x', rotation=15, colors=GAMING_COLORS['light_text'])
        axes[1, 1].tick_params(axis='y', colors=GAMING_COLORS['text'])
        
        self._apply_gaming_style(fig, "PERFIL DE ESTILOS DE JUEGO")
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_combat_style_analysis(self):
        """Análisis de combate con estilo gaming - FIXED no emojis"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(GAMING_COLORS['dark_bg'])
        
        # 1. Distribución de estilos de combate - Gaming bars
        combat_counts = self.df['combat_style'].value_counts()
        bars = axes[0].bar(combat_counts.index, combat_counts.values,
                          color=COMBAT_PALETTE, edgecolor='white', 
                          alpha=0.8, linewidth=2)
        
        axes[0].set_title('ESTILOS DE COMBATE', 
                         fontweight='bold', color=GAMING_COLORS['primary'])
        axes[0].set_xlabel('Estilo de Combate', color=GAMING_COLORS['light_text'])
        axes[0].set_ylabel('Jugadores', color=GAMING_COLORS['light_text'])
        axes[0].tick_params(axis='x', rotation=15, colors=GAMING_COLORS['light_text'])
        axes[0].tick_params(axis='y', colors=GAMING_COLORS['light_text'])
        
        # Add values without emojis
        for i, (style, count) in enumerate(combat_counts.items()):
            axes[0].text(i, count + 50, f'{count}', 
                        ha='center', va='bottom', fontweight='bold',
                        color=GAMING_COLORS['highlight'],
                        bbox=dict(boxstyle="round,pad=0.3",
                                facecolor=GAMING_COLORS['dark_bg'],
                                edgecolor=GAMING_COLORS['primary'],
                                alpha=0.8))
        
        # 2. Heatmap de relación estilos de juego vs combate - FIXED
        combat_playstyle = pd.crosstab(self.df['combat_style'], self.df['playstyle'])
        sns.heatmap(combat_playstyle, annot=True, fmt='d', cmap='YlOrRd',
                   ax=axes[1], cbar_kws={'label': 'Jugadores'})
        
        axes[1].set_title('DISTRIBUCIÓN ESTILO/COMBATE', 
                         fontweight='bold', color=GAMING_COLORS['primary'])
        axes[1].set_xlabel('Estilo de Juego', color=GAMING_COLORS['light_text'])
        axes[1].set_ylabel('Estilo de Combate', color=GAMING_COLORS['light_text'])
        axes[1].tick_params(colors=GAMING_COLORS['text'])
        
        self._apply_gaming_style(fig, "ANÁLISIS DE ESTILOS DE COMBATE")
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_engagement_analysis(self):
        """Análisis de engagement con visualizaciones gaming - FIXED"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.patch.set_facecolor(GAMING_COLORS['dark_bg'])
        
        # 1. Scatter: Tiempo vs Engagement con color por skill - FIXED
        valid_data = self.df[['playtime_hours', 'engagement_score', 'skill_level']].dropna()
        if len(valid_data) > 0:
            scatter = axes[0, 0].scatter(valid_data['playtime_hours'], 
                                        valid_data['engagement_score'],
                                        c=valid_data['skill_level'], 
                                        cmap='plasma', alpha=0.7, s=50,
                                        edgecolors='white', linewidth=0.5)
            axes[0, 0].set_title('ENGAGEMENT VS TIEMPO', 
                               fontweight='bold', color=GAMING_COLORS['primary'])
            axes[0, 0].set_xlabel('Horas de Juego', color=GAMING_COLORS['light_text'])
            axes[0, 0].set_ylabel('Puntaje de Engagement', color=GAMING_COLORS['light_text'])
            cbar = plt.colorbar(scatter, ax=axes[0, 0])
            cbar.set_label('Nivel de Habilidad', color=GAMING_COLORS['light_text'])
            cbar.ax.tick_params(colors=GAMING_COLORS['text'])
        
        # 2. Sesiones vs Logros con tendencia - FIXED
        valid_data = self.df[['sessions_per_week', 'achievements_unlocked']].dropna()
        if len(valid_data) > 0:
            axes[0, 1].scatter(valid_data['sessions_per_week'], 
                              valid_data['achievements_unlocked'],
                              alpha=0.6, color=GAMING_COLORS['accent'], s=40)
            
            # Add trend line only if we have enough data
            if len(valid_data) > 1:
                z = np.polyfit(valid_data['sessions_per_week'], 
                              valid_data['achievements_unlocked'], 1)
                p = np.poly1d(z)
                axes[0, 1].plot(valid_data['sessions_per_week'], 
                               p(valid_data['sessions_per_week']), 
                               color=GAMING_COLORS['primary'], linewidth=3,
                               label='Tendencia')
            
            axes[0, 1].set_title('SESIONES VS LOGROS', 
                               fontweight='bold', color=GAMING_COLORS['primary'])
            axes[0, 1].set_xlabel('Sesiones por Semana', color=GAMING_COLORS['light_text'])
            axes[0, 1].set_ylabel('Logros Desbloqueados', color=GAMING_COLORS['light_text'])
            axes[0, 1].legend(facecolor=GAMING_COLORS['card_bg'])
        
        # 3. Riesgo de abandono - Donut chart FIXED with labels
        churn_counts = self.df['churn_risk'].value_counts()
        wedges, texts, autotexts = axes[1, 0].pie(churn_counts.values, 
                                                 labels=churn_counts.index,  # This adds labels
                                                 autopct='%1.1f%%', startangle=90,
                                                 colors=RISK_PALETTE,
                                                 wedgeprops={'edgecolor': 'white', 
                                                           'linewidth': 2},
                                                 textprops={'color': GAMING_COLORS['light_text']})  # FIXED: Add text color
        
        # Style the labels and percentages
        for text in texts:
            text.set_color(GAMING_COLORS['text'])
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            
        # Make it a donut
        centre_circle = plt.Circle((0,0), 0.70, fc=GAMING_COLORS['dark_bg'])
        axes[1, 0].add_artist(centre_circle)
        axes[1, 0].set_title('RIESGO DE ABANDONO', 
                           fontweight='bold', color=GAMING_COLORS['primary'])
        
        # 4. PvP por estilo - Boxplot instead of violin to avoid issues
        playstyles_sorted = self.df['playstyle'].unique()
        pvp_data = [self.df[self.df['playstyle'] == style]['pvp_matches'].dropna() 
                   for style in playstyles_sorted]
        
        # Only plot if we have data
        if any(len(data) > 0 for data in pvp_data):
            bp = axes[1, 1].boxplot(pvp_data, labels=playstyles_sorted, 
                       patch_artist=True,
                       boxprops=dict(linewidth=2),
                       medianprops=dict(linewidth=2, color='white'),  # FIJADO
                       whiskerprops=dict(linewidth=2),
                       capprops=dict(linewidth=2),
                       flierprops=dict(marker='o', 
                                     markerfacecolor=GAMING_COLORS['accent'],  # FIJADO
                                     markersize=4,
                                     markeredgecolor=GAMING_COLORS['text'],  # FIJADO
                                     alpha=0.6))
            
            # Style boxplot
            for patch, color in zip(bp['boxes'], PLAYSTYLE_PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for element in ['whiskers', 'caps', 'medians']:
                plt.setp(bp[element], color=GAMING_COLORS['text'], linewidth=2)
            
            axes[1, 1].set_title('PARTIDAS PvP POR ESTILO', 
                               fontweight='bold', color=GAMING_COLORS['primary'])
            axes[1, 1].set_xlabel('Estilo de Juego', color=GAMING_COLORS['light_text'])
            axes[1, 1].set_ylabel('Partidas PvP', color=GAMING_COLORS['light_text'])
            axes[1, 1].tick_params(axis='x', rotation=15, colors=GAMING_COLORS['light_text'])
            axes[1, 1].tick_params(axis='y', colors=GAMING_COLORS['text'])
        
        self._apply_gaming_style(fig, "ANÁLISIS DE ENGAGEMENT Y RENDIMIENTO")
        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def save_all_figures(self, output_dir='visualizations/'):
        """Guarda todas las figuras con tema gaming"""
        os.makedirs(output_dir, exist_ok=True)
        
        figure_names = [
            'estilos_juego',
            'distribuciones_metricas',
            'matriz_correlacion',
            'caracteristicas_estilos',
            'analisis_combate',
            'engagement_rendimiento'
        ]
        
        for idx, (fig, name) in enumerate(zip(self.figures, figure_names)):
            filepath = os.path.join(output_dir, f'gaming_{idx+1}_{name}.png')
            fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor=GAMING_COLORS['dark_bg'])
            print(f"Guardado: {filepath}")
    
    def generate_report(self):
        """Genera reporte completo"""
        print("\n" + "=" * 60)
        print("GENERANDO REPORTE GAMING EDA")
        print("=" * 60)
        
        self.basic_info()
        
        print("\nGenerando visualizaciones gaming...")
        self.plot_target_distribution()
        print("- Distribución de estilos")
        
        self.plot_numerical_distributions()
        print("- Distribuciones numéricas")
        
        self.plot_correlation_matrix()
        print("- Matriz de correlación")
        
        self.plot_playstyle_characteristics()
        print("- Características por estilo")
        
        self.plot_combat_style_analysis()
        print("- Análisis de combate")
        
        self.plot_engagement_analysis()
        print("- Análisis de engagement")
        
        self.save_all_figures()
        
        print("\n" + "=" * 60)
        print("Reporte completado.")
        print("=" * 60)

def main():
    """Función principal"""
    print("=" * 60)
    print("ANÁLISIS EXPLORATORIO")
    print("=" * 60)
    
    # Cargar datos
    print("\nCargando datos...")
    df = pd.read_csv('data/gaming_behavior_raw.csv')
    print(f"Datos cargados: {df.shape[0]} jugadores, {df.shape[1]} atributos")
    
    # Crear analizador gaming
    analyzer = EDAAnalyzer(df)
    
    # Generar reporte completo
    analyzer.generate_report()
    
    print("\nVisualizaciones en la carpeta 'visualizations/'")
    plt.show()

if __name__ == "__main__":
    main()
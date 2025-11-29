"""
Preprocesamiento y Limpieza de Datos
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class DataPreprocessor:
    """Clase para preprocesar datos de jugadores"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath='data/gaming_behavior_raw.csv'):
        """Carga los datos desde un archivo CSV"""
        print(f"Cargando datos desde {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Cargados {len(df)} registros con {len(df.columns)} columnas")
        return df
    
    def clean_data(self, df):
        """Limpia el dataset"""
        print("\nLimpiando datos...")
        
        initial_rows = len(df)
        
        # 1. Eliminar duplicados
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        print(f"  - Duplicados eliminados: {duplicates_removed}")
        
        # 2. Manejar valores nulos
        # Para columnas numéricas, rellenar con la mediana
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  - {col}: {df[col].isnull().sum()} nulos rellenados con mediana ({median_val:.2f})")
        
        # 3. Validar rangos de valores
        df['difficulty_level'] = df['difficulty_level'].clip(1, 10)
        df['win_rate'] = df['win_rate'].clip(0, 1)
        df['sessions_per_week'] = df['sessions_per_week'].clip(0, None)
        df['playtime_hours'] = df['playtime_hours'].clip(0, None)
        
        print(f"Limpieza completada. Registros finales: {len(df)}")
        
        return df
    
    def engineer_features(self, df):
        """Crea nuevas características derivadas"""
        print("\nCreando features derivadas...")
        
        # Ratio de kills/deaths (K/D)
        df['kd_ratio'] = df.apply(
            lambda x: x['pvp_matches'] / max(x['death_count'], 1), axis=1
        )
        
        # Intensidad de juego (horas por sesión)
        df['play_intensity'] = df['playtime_hours'] / (df['sessions_per_week'] * 4 + 1)
        
        # Nivel de compromiso
        df['commitment_score'] = (
            df['playtime_hours'] * 0.3 + 
            df['sessions_per_week'] * 10 + 
            df['achievements_unlocked'] * 0.5
        )
        
        # Experiencia en PvP
        df['pvp_experience'] = np.log1p(df['pvp_matches'])
        
        # Eficiencia en logros
        df['achievement_rate'] = df['achievements_unlocked'] / (df['playtime_hours'] + 1)
        
        print(f"Features creadas: kd_ratio, play_intensity, commitment_score, pvp_experience, achievement_rate")
        
        return df
    
    def encode_categorical(self, df, fit=True):
        """Codifica variables categóricas"""
        print("\nCodificando variables categóricas...")
        
        categorical_cols = ['combat_style', 'churn_risk']
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                print(f"  - {col}: {len(le.classes_)} categorías -> valores numéricos")
            else:
                if col in self.label_encoders:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def scale_features(self, df, fit=True):
        """Normaliza las características numéricas"""
        print("\nNormalizando características...")
        
        # Características a normalizar
        features_to_scale = [
            'playtime_hours', 'sessions_per_week', 'avg_session_length',
            'achievements_unlocked', 'difficulty_level', 'win_rate',
            'pvp_matches', 'death_count', 'engagement_score', 'skill_level',
            'kd_ratio', 'play_intensity', 'commitment_score', 
            'pvp_experience', 'achievement_rate', 'last_login_days_ago'
        ]
        
        # Filtrar solo las que existen en el df
        features_to_scale = [f for f in features_to_scale if f in df.columns]
        
        if fit:
            df[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])
            self.feature_names = features_to_scale
            print(f"Normalizadas {len(features_to_scale)} características")
        else:
            df[features_to_scale] = self.scaler.transform(df[features_to_scale])
        
        return df, features_to_scale
    
    def prepare_features(self, df, target_col='playstyle'):
        """Prepara X (features) e y (target)"""
        
        # Características a usar para el modelo
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
        
        X = df[feature_cols]
        y = df[target_col]
        
        return X, y, feature_cols
    
    def save_preprocessor(self, path='models/'):
        """Guarda el preprocesador"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(path, 'label_encoders.pkl'))
        joblib.dump(self.feature_names, os.path.join(path, 'feature_names.pkl'))
        print(f"\nPreprocesador guardado en {path}")
    
    def load_preprocessor(self, path='models/'):
        """Carga el preprocesador guardado"""
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(path, 'label_encoders.pkl'))
        self.feature_names = joblib.load(os.path.join(path, 'feature_names.pkl'))
        print(f"Preprocesador cargado desde {path}")
    
    def process_pipeline(self, df, fit=True):
        """Pipeline completo de preprocesamiento"""
        df = self.clean_data(df)
        df = self.engineer_features(df)
        df = self.encode_categorical(df, fit=fit)
        df, scaled_features = self.scale_features(df, fit=fit)
        
        return df

def main():
    """Función principal"""
    print("=" * 60)
    print("PREPROCESAMIENTO DE DATOS")
    print("=" * 60)
    
    # Crear preprocesador
    preprocessor = DataPreprocessor()
    
    # Cargar datos
    df = preprocessor.load_data('data/gaming_behavior_raw.csv')
    
    # Procesar datos
    df_processed = preprocessor.process_pipeline(df, fit=True)
    
    # Preparar X e y
    print("\nPreparando features y target...")
    X, y, feature_cols = preprocessor.prepare_features(df_processed)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features: {len(feature_cols)}")
    
    # Guardar datos procesados
    print("\nGuardando datos procesados...")
    df_processed.to_csv('data/gaming_behavior_processed.csv', index=False)
    print("Guardado: data/gaming_behavior_processed.csv")
    
    # Guardar preprocesador
    preprocessor.save_preprocessor()
    
    # Mostrar estadísticas finales
    print("\nEstadísticas finales:")
    print(f"  - Registros totales: {len(df_processed)}")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Distribución de clases:")
    print(y.value_counts())
    
    print("\n" + "=" * 60)
    print("Preprocesamiento completado.")
    print("=" * 60)

if __name__ == "__main__":
    main()
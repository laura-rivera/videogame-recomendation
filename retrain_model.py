"""
Script para reentrenamiento automático del modelo
Ejecutar periódicamente (ej: cron job mensual)
"""

def retrain_model_if_needed():
    from feedback_system import FeedbackSystem, ModelMonitoring
    from train_model import GameplayModelTrainer
    
    feedback_sys = FeedbackSystem()
    monitor = ModelMonitoring(feedback_sys)
    
    # Verificar si se necesita reentrenamiento
    needs_retrain, reason, metrics = monitor.check_retraining_needed()
    
    if needs_retrain:
        print(f"🔄 Iniciando reentrenamiento: {reason}")
        
        # Exportar nuevos datos
        if not feedback_sys.export_training_data('data/retraining_data.csv'):
            print("❌ No hay suficientes datos para reentrenar")
            return False
        
        # Combinar con datos originales
        original_data = pd.read_csv('data/gaming_behavior_processed.csv')
        new_data = pd.read_csv('data/retraining_data.csv')
        
        combined_data = pd.concat([original_data, new_data], ignore_index=True)
        combined_data.to_csv('data/combined_training_data.csv', index=False)
        
        # Reentrenar modelo
        trainer = GameplayModelTrainer()
        df = trainer.load_data('data/combined_training_data.csv')
        trainer.prepare_data(df)
        trainer.train_and_evaluate_all()
        
        print("✅ Reentrenamiento completado")
        
        # Crear backup del modelo anterior
        import shutil
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy(
            'models/best_model.pkl',
            f'models/backup/best_model_{timestamp}.pkl'
        )
        
        return True
    else:
        print(f"✅ Reentrenamiento no necesario: {reason}")
        return False

if __name__ == "__main__":
    retrain_model_if_needed()
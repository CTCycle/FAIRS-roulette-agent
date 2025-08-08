import os
import pandas as pd
import numpy as np
from keras import Model

from FAIRS.app.utils.learning.callbacks import LearningInterruptCallback
from FAIRS.app.utils.data.serializer import DataSerializer, ModelSerializer
from FAIRS.app.interface.workers import check_thread_status, update_progress_callback
from FAIRS.app.constants import CHECKPOINT_PATH
from FAIRS.app.logger import logger



# [LOAD MODEL]
################################################################################
class ModelEvaluationSummary:

    def __init__(self, configuration : dict):         
        self.configuration = configuration

    #---------------------------------------------------------------------------
    def scan_checkpoint_folder(self):
        model_paths = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():                
                pretrained_model_path = os.path.join(entry.path, 'saved_model.keras')                
                if os.path.isfile(pretrained_model_path):
                    model_paths.append(entry.path)                

        return model_paths  

    #---------------------------------------------------------------------------
    def get_checkpoints_summary(self, **kwargs):
        modser = ModelSerializer() 
        serializer = DataSerializer(self.configuration)           
        model_paths = self.scan_checkpoint_folder()
        model_parameters = []                       
        for i, model_path in enumerate(model_paths):
            configuration, history = modser.load_training_configuration(model_path)
            model_name = os.path.basename(model_path)                   
            precision = 16 if configuration.get("use_mixed_precision", np.nan) else 32 
            
            scores = history.get('history', {})
            chkp_config = {
                    'checkpoint': model_name,
                    'sample_size': configuration.get('sample_size', np.nan),
                    'seed': configuration.get('train_seed', np.nan),
                    'precision': precision,
                    'episodes': history.get('episodes', np.nan),      
                    'max_steps_episode': history.get('max_steps_episode', np.nan),                
                    'batch_size': configuration.get('batch_size', np.nan),
                    'jit_compile': configuration.get('jit_compile', np.nan),
                    'has_tensorboard_logs': configuration.get('use_tensorboard', np.nan),
                    'learning_rate': configuration.get('learning_rate', np.nan),  
                    'neurons': configuration.get('QNet_neurons', np.nan),  
                    'embedding_dimensions': configuration.get('embedding_dimensions', np.nan),
                    'perceptive_field_size': configuration.get('perceptive_field_size', np.nan),      
                    'exploration_rate': configuration.get('exploration_rate', np.nan),   
                    'exploration_rate_decay': configuration.get('exploration_rate_decay', np.nan),    
                    'discount_rate': configuration.get('discount_rate', np.nan),              
                    'perceptive_field_size': configuration.get('perceptive_field_size', np.nan), 
                    'model_update_frequency': configuration.get('model_uptade_frequency', np.nan),   
                    'loss': scores.get('loss', [np.nan])[-1],                     
                    'accuracy': scores.get('cosine_similarity', [np.nan])[-1], 
                    }
                    
            
            model_parameters.append(chkp_config)

            # check for thread status and progress bar update   
            check_thread_status(kwargs.get('worker', None))         
            update_progress_callback(
                i+1, len(model_paths), kwargs.get('progress_callback', None)) 

        dataframe = pd.DataFrame(model_parameters)
        serializer.save_checkpoints_summary(dataframe)    
            
        return dataframe
    
    #--------------------------------------------------------------------------
    def get_evaluation_report(self, model : Model, validation_dataset, **kwargs):
        callbacks_list = [LearningInterruptCallback(kwargs.get('worker', None))]
        validation = model.evaluate(validation_dataset, verbose=1, callbacks=callbacks_list) 
        logger.info(f'Evaluation of pretrained model has been completed')   
        logger.info(f'RMSE loss {validation[0]:.3f}')
        logger.info(f'Cosine similarity {validation[1]:.3f}') 




# [VALIDATION OF PRETRAINED MODELS]
###############################################################################
class BetsAccuracy:

    def __init__(self, model : Model):        
        self.file_type = 'jpeg'        
        self.model = model  

    # comparison of data distribution using statistical methods 
    #--------------------------------------------------------------------------     
    def plot_timeseries_prediction(self, values, name, path, dpi=400):
        train_data = values['train']
        test_data = values['test']
        plt.figure(figsize=(12, 10))        
        plt.scatter(train_data[0], train_data[1], label='True train', color='blue')
        plt.scatter(test_data[0], test_data[1], label='True test', color='cyan')        
        plt.scatter(train_data[0], train_data[2], label='Predicted train', color='orange')
        plt.scatter(test_data[0], test_data[2], label='Predicted test', color='magenta')
        plt.xlabel('Extraction N.', fontsize=14)
        plt.ylabel('Class', fontsize=14)
        plt.title('FAIRS Extractions', fontsize=14)
        plt.legend()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plot_loc = os.path.join(path, f'{name}.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi=dpi)
        plt.close()
    
    # comparison of data distribution using statistical methods 
    #--------------------------------------------------------------------------     
    def plot_confusion_matrix(self, Y_real, predictions, name, path, dpi=400): 
        class_names = ['green', 'black', 'red']        
        cm = confusion_matrix(Y_real, predictions)    
        plt.figure(figsize=(14, 14))        
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)        
        plt.xlabel('Predicted labels', fontsize=14)
        plt.ylabel('True labels', fontsize=14)
        plt.title('Confusion Matrix', fontsize=14)
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45, fontsize=12, ha="right")
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0, fontsize=12, va="center")          
        plt.tight_layout()
        plot_loc = os.path.join(path, f'{name}.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi=self.DPI)
        plt.close()

    
        

              
        
        
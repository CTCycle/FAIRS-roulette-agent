import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import layers 
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


# [CALLBACK FOR REAL TIME TRAINING MONITORING]
#==============================================================================
#==============================================================================
#==============================================================================
class RealTimeHistory(keras.callbacks.Callback):
    
    """ 
    A class including the callback to show a real time plot of the training history. 
      
    Methods:
        
    __init__(plot_path): initializes the class with the plot savepath       
    
    """   
    def __init__(self, plot_path, validation=True):        
        super().__init__()
        self.plot_path = plot_path
        self.epochs = []
        self.loss_hist = []
        self.metric_hist = []
        self.loss_val_hist = []        
        self.metric_val_hist = []
        self.validation = validation            
    #--------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs = {}):
        if epoch % 5 == 0:                    
            self.epochs.append(epoch)
            self.loss_hist.append(logs[list(logs.keys())[0]])
            self.metric_hist.append(logs[list(logs.keys())[1]])
            if self.validation==True:
                self.loss_val_hist.append(logs[list(logs.keys())[2]])            
                self.metric_val_hist.append(logs[list(logs.keys())[3]])
        if epoch % 20 == 0:            
            #------------------------------------------------------------------
            fig_path = os.path.join(self.plot_path, 'training_history.jpeg')
            plt.subplot(2, 1, 1)
            plt.plot(self.epochs, self.loss_hist, label = 'training loss')
            if self.validation==True:
                plt.plot(self.epochs, self.loss_val_hist, label = 'validation loss')
                plt.legend(loc = 'best', fontsize = 8)
            plt.title('Loss plot')
            plt.ylabel('Categorical Crossentropy')
            plt.xlabel('epoch')
            plt.subplot(2, 1, 2)
            plt.plot(self.epochs, self.metric_hist, label = 'train metrics') 
            if self.validation==True: 
                plt.plot(self.epochs, self.metric_val_hist, label = 'validation metrics') 
                plt.legend(loc = 'best', fontsize = 8)
            plt.title('metrics plot')
            plt.ylabel('Categorical accuracy')
            plt.xlabel('epoch')       
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches = 'tight', format = 'jpeg', dpi = 300)
            plt.close()
                  

# Class for preprocessing tabular data prior to GAN training 
#==============================================================================
#==============================================================================
#==============================================================================
class GroupedClassModel:

    def __init__(self, learning_rate, window_size, output_size):

        self.learning_rate = learning_rate
        self.window_size = window_size 
        self.output_size = output_size
        self.name = 'FAIRS_GCM'

    def build(self):                
        
        sequence_input = layers.Input(shape=(self.window_size, 1))       
        #----------------------------------------------------------------------
        lstm1 = layers.LSTM(512, use_bias=True, return_sequences=True, activation='tanh', dropout=0.2)(sequence_input) 
        #----------------------------------------------------------------------        
        lstm2 = layers.LSTM(256, use_bias=True, return_sequences=True, activation='tanh', dropout=0.2)(lstm1)
        #----------------------------------------------------------------------   
        lstm3 = layers.LSTM(128, use_bias=True, return_sequences=True, activation='tanh', dropout=0.2)(lstm2) 
        #----------------------------------------------------------------------
        flatten1 = layers.Flatten()(lstm3)       
        
               
        conv1 = layers.Conv1D(512, kernel_size=6, activation='relu')(sequence_input)
        #----------------------------------------------------------------------               
        conv2 = layers.Conv1D(256, kernel_size=6, activation='relu')(conv1)
        #----------------------------------------------------------------------        
        conv3 = layers.Conv1D(128, kernel_size=6, activation='relu')(conv2)
        #----------------------------------------------------------------------       
        flatten2 = layers.Flatten()(conv3)


        concat = layers.Concatenate()([flatten1, flatten2])
        #----------------------------------------------------------------------                
        dense1 = layers.Dense(1024, activation='relu')(concat)
        #---------------------------------------------------------------------- 
        drop1 = layers.Dropout(rate=0.2)(dense1)   
        #----------------------------------------------------------------------      
        dense2 = layers.Dense(512, activation='relu')(drop1)
        #----------------------------------------------------------------------
        drop2 = layers.Dropout(rate=0.2)(dense2)   
        #----------------------------------------------------------------------
        dense3 = layers.Dense(256, activation='relu')(drop2)
        #----------------------------------------------------------------------
        drop3 = layers.Dropout(rate=0.2)(dense3)   
        #----------------------------------------------------------------------
        dense4 = layers.Dense(128, activation='relu')(drop3)
        #----------------------------------------------------------------------
        dense5 = layers.Dense(64, activation='relu')(dense4)
        #----------------------------------------------------------------------
        output = layers.Dense(self.output_size, activation='softmax', dtype='float32')(dense5)        
        
        model = Model(inputs = sequence_input, outputs = output, name = 'FAIRS_model')      
    
    
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = keras.losses.CategoricalCrossentropy()
        metrics = keras.metrics.CategoricalAccuracy()
        model.compile(loss = loss, optimizer = opt, metrics = metrics)       
        
        return model
    

              


# define model class
#==============================================================================
#==============================================================================
#==============================================================================
class ModelTraining:
    
    def __init__(self, device = 'default'):
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)              
        np.random.seed(42)
        tf.random.set_seed(42)         
        self.available_devices = tf.config.list_physical_devices()
        print('----------------------------------------------------------------')
        print('The current devices are available: ')
        for dev in self.available_devices:
            print()
            print(dev)
        print()
        print('----------------------------------------------------------------')
        if device == 'GPU':
            self.physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.set_visible_devices(self.physical_devices[0], 'GPU')
            print('GPU is set as active device')
            print('----------------------------------------------------------------')
            print()        
        elif device == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            print('CPU is set as active device')
            print('----------------------------------------------------------------')
            print()
          
    # sequential model as generator with Keras module
    #========================================================================== 
    def load_pretrained_model(self, path):
        
        model_folders = []
        for entry in os.scandir(path):
            if entry.is_dir():
                model_folders.append(entry.name)
        model_folders.sort()
        index_list = [idx + 1 for idx, item in enumerate(model_folders)]     
        print('Please select a pretrained model:') 
        print()
        for i, directory in enumerate(model_folders):
            print('{0} - {1}'.format(i + 1, directory))
        
        print()               
        while True:
           try:
              dir_index = int(input('Type the model index to select it: '))
              print()
           except:
              continue
           break
                         
        while dir_index not in index_list:
           try:
               dir_index = int(input('Input is not valid! Try again: '))
               print()
           except:
               continue  
           
        model_path = os.path.join(path, model_folders[dir_index - 1])
        model = keras.models.load_model(model_path)        
        
        return model   


# define class for trained model validation and data comparison
#============================================================================== 
#==============================================================================
#==============================================================================
class ModelValidation:    
    
    
    
    

    # comparison of data distribution using statistical methods 
    #==========================================================================     
    def FAIRS_confusion(self, Y_real, predictions, classes, name, path, dpi):         
        cm = confusion_matrix(Y_real, predictions)    
        fig, ax = plt.subplots()        
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
                cbar=False)
        
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(classes)
        ax.yaxis.set_ticklabels(classes)
        plt.tight_layout()
        plot_loc = os.path.join(path, f'confusion_matrix_{name}.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi = dpi)        

    # comparison of data distribution using statistical methods 
    #==========================================================================
    def plot_multi_ROC(Y_real, predictions, class_dict, path, dpi):
    
        Y_real_bin = label_binarize(Y_real, classes=list(class_dict.values()))
        n_classes = Y_real_bin.shape[1]        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_real_bin[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])    
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(list(class_dict.keys())[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")       
        plot_loc = os.path.join(path, 'multi_ROC.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi=dpi)           
    


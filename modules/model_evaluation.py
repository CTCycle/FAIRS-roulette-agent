import os
import sys
import pickle
import numpy as np

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add modules path if this file is launched as __main__
#------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import modules and components
#------------------------------------------------------------------------------
from modules.components.model_assets import Inference, ModelValidation
import modules.global_variables as GlobVar

# [LOAD DATASETS AND MODEL]
#==============================================================================
# Load patient dataset and dictionaries from .csv files in the dataset folder.
#==============================================================================

# Load model
#------------------------------------------------------------------------------
inference = Inference() 
model, parameters = inference.load_pretrained_model(GlobVar.models_path)
model_folder = inference.folder_path
model.summary(expand_nested=True)

# Load normalizer and encoders
#------------------------------------------------------------------------------
if parameters['Model_name']=='CCM':    
    encoder_path = os.path.join(model_folder, 'preprocessing', 'categorical_encoder.pkl')
    with open(encoder_path, 'rb') as file:
        encoder = pickle.load(file)    

# load npy files
#------------------------------------------------------------------------------
if parameters['Model_name']=='CCM':
    load_path = os.path.join(model_folder, 'preprocessing')
    X_train = np.load(os.path.join(load_path, 'train_data.npy'))
    Y_train_OHE = np.load(os.path.join(load_path, 'train_labels.npy'))
    X_test = np.load(os.path.join(load_path, 'test_data.npy'))
    Y_test_OHE = np.load(os.path.join(load_path, 'test_labels.npy'))
    train_inputs, train_outputs = X_train, Y_train_OHE
    test_inputs, test_outputs = X_test, Y_test_OHE
elif parameters['Model_name']=='NMM':
    load_path = os.path.join(model_folder, 'preprocessing')
    X_train_ext = np.load(os.path.join(load_path, 'train_extractions.npy'))
    X_train_pos = np.load(os.path.join(load_path, 'train_positions.npy'))
    Y_train_OHE = np.load(os.path.join(load_path, 'train_labels.npy'))
    X_test_ext = np.load(os.path.join(load_path, 'test_extractions.npy'))
    X_test_pos = np.load(os.path.join(load_path, 'test_positions.npy'))
    Y_test_OHE = np.load(os.path.join(load_path, 'test_labels.npy'))
    train_inputs, train_outputs = [X_train_ext, X_train_pos], Y_train_OHE
    test_inputs, test_outputs = [X_test_ext, X_test_pos], Y_test_OHE

# [MODEL VALIDATION]
#==============================================================================
# ...
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
FAIRS model evaluation
-------------------------------------------------------------------------------
...
''')

validator = ModelValidation(model)

# create subfolder for evaluation data
#------------------------------------------------------------------------------
eval_path = os.path.join(model_folder, 'evaluation') 
if not os.path.exists(eval_path):
    os.mkdir(eval_path)

# predict lables from train set
#------------------------------------------------------------------------------
predicted_train = model.predict(train_inputs, verbose=0)
y_pred = np.argmax(predicted_train, axis=-1)
y_true = np.argmax(train_outputs, axis=-1)


# show predicted classes (train dataset)
#------------------------------------------------------------------------------
class_pred, class_true = np.unique(y_pred), np.unique(y_true)
print(f'''
Number of classes observed in train (true labels): {len(class_true)}
Number of classes observed in train (predicted labels): {len(class_pred)}
-------------------------------------------------------------------------------
Classes observed in predicted train labels:
-------------------------------------------------------------------------------
{class_pred}
''')

# generate confusion matrix from train set (if class num is equal)
#------------------------------------------------------------------------------
try:
    validator.FAIRS_confusion(y_true, y_pred, 'confusion_matrix_train', eval_path)    
except Exception as e:
    print('Could not generate confusion matrix for train dataset')
    print('Error:', str(e))

# predict labels from test set
#------------------------------------------------------------------------------
predicted_test = model.predict(test_inputs, verbose=0)
y_pred_labels = np.argmax(predicted_test, axis=-1)
y_true_labels = np.argmax(test_outputs, axis=-1)

# show predicted classes (testdataset)
#------------------------------------------------------------------------------
class_pred, class_true = np.unique(y_pred), np.unique(y_true)
print(f'''
-------------------------------------------------------------------------------
Number of classes observed in test (true labels): {len(class_true)}
Number of classes observed in test (predicted labels): {len(class_pred)}
-------------------------------------------------------------------------------
Classes observed in predicted test labels:
-------------------------------------------------------------------------------
{class_pred}
-------------------------------------------------------------------------------
''')    

# generate confusion matrix from test set (if class num is equal)
#------------------------------------------------------------------------------
try:
    validator.FAIRS_confusion(y_true, y_pred, 'confusion_matrix_test', eval_path)        
except Exception as e:
    print('Could not generate confusion matrix for test dataset')
    print('Error:', str(e))


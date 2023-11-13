import os
import sys
import numpy as np
import pandas as pd
import pickle

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
from modules.components.data_classes import PreProcessing
from modules.components.training_classes import ModelTraining
import modules.global_variables as GlobVar
import modules.configurations as cnf



# [LOAD MODEL AND DATA]
#==============================================================================
# ....
#==============================================================================        
print(f'''
-------------------------------------------------------------------------------
FAIRS predictions
-------------------------------------------------------------------------------
...
''')

# Load dataset
#------------------------------------------------------------------------------
filepath = os.path.join(GlobVar.CCM_data_path, 'predictions_inputs.csv')                
df_predictions = pd.read_csv(filepath, sep= ';', encoding='utf-8')

# Load model
#------------------------------------------------------------------------------
trainworker = ModelTraining(device = cnf.training_device) 
model = trainworker.load_pretrained_model(GlobVar.CCM_model_path)
parameters = trainworker.model_configuration
model.summary(expand_nested=True)

# Load normalizer and encoders
#------------------------------------------------------------------------------
if parameters['Class encoding'] == True:
    encoder_path = os.path.join(GlobVar.CCM_data_path, 'categorical_encoder.pkl')
    with open(encoder_path, 'rb') as file:
        encoder = pickle.load(file)

# [PREPROCESS DATA]
#==============================================================================
# ...
#==============================================================================

# map numbers to roulette color and reshape array
#------------------------------------------------------------------------------
PP = PreProcessing()
if parameters['Class encoding'] == True:
    df_predictions = PP.roulette_colormapping(df_predictions, no_mapping=False)
    CCM_timeseries = df_predictions['encoding']
    CCM_timeseries = CCM_timeseries.values.reshape(-1, 1)
    categories = [['green', 'black', 'red']]
    CCM_timeseries = encoder.fit_transform(CCM_timeseries)
    CCM_timeseries = pd.DataFrame(CCM_timeseries, columns=['encoding'])
else:
    df_FAIRS = PP.roulette_colormapping(df_predictions, no_mapping=True)
    categories = [[x for x in df_FAIRS['encoding'].unique()]]
    FAIRS_categorical = df_FAIRS['encoding']


# generate windowed dataset
#------------------------------------------------------------------------------
CCM_inputs = PP.timeseries_labeling(CCM_timeseries, parameters['Window size'], 
                                              parameters['Output seq length'])
predictions_inputs = CCM_inputs[0]

# [PERFORM PREDICTIONS]
#==============================================================================
# ....
#==============================================================================
print('''Perform prediction using the loaded model
''')

# predict using pretrained model
#------------------------------------------------------------------------------ 
probability_vectors = model.predict(predictions_inputs)
expected_class = np.argmax(probability_vectors, axis=-1)

last_window = CCM_timeseries['color encoding'].to_list()[-parameters['Window size']:]
last_window = np.reshape(last_window, (1, parameters['Window size'], 1))
next_prob_vector = model.predict(last_window)
next_exp_class = np.argmax(next_prob_vector, axis=-1)

# inverse encoding of the classes
#------------------------------------------------------------------------------ 
expected_class = np.array(expected_class).reshape(-1, 1)
expected_color = encoder.inverse_transform(expected_class)
expected_color = expected_color.flatten().tolist()

next_exp_class = np.array(next_exp_class).reshape(-1, 1)
next_exp_color = encoder.inverse_transform(next_exp_class)
next_exp_color = next_exp_color.flatten().tolist()[0]

original_class = np.array(CCM_timeseries['color encoding'].to_list()).reshape(-1, 1)
original_names = encoder.inverse_transform(original_class)
original_names = np.append(original_names.flatten().tolist(), '?')

# synchronize the window of timesteps with the predictions
#------------------------------------------------------------------------------ 
sync_expected_vector = {'Green' : [], 'Black' : [], 'Red' : []}
sync_expected_color = []
for ts in range(cnf.window_size):
    sync_expected_vector['Green'].append('')
    sync_expected_vector['Black'].append('')
    sync_expected_vector['Red'].append('')
    sync_expected_color.append('')
for x, z in zip(probability_vectors, expected_color):    
    sync_expected_vector['Green'].append(x[0,0])
    sync_expected_vector['Black'].append(x[0,1])
    sync_expected_vector['Red'].append(x[0,2])
    sync_expected_color.append(z)

for i in range(next_prob_vector.shape[1]):
    sync_expected_vector['Green'].append(next_prob_vector[0,i,0])
    sync_expected_vector['Black'].append(next_prob_vector[0,i,1])
    sync_expected_vector['Red'].append(next_prob_vector[0,i,2])
    sync_expected_color.append(next_exp_color)

# add column with prediction to dataset
#------------------------------------------------------------------------------
CCM_timeseries.loc[len(CCM_timeseries.index)] = None
CCM_timeseries['expected color'] = sync_expected_color
CCM_timeseries['color encoding'] = original_names
df_probability = pd.DataFrame(sync_expected_vector)
df_merged = pd.concat([CCM_timeseries, df_probability], axis=1)

# print console report
#------------------------------------------------------------------------------ 
print(f'''
-------------------------------------------------------------------------------
Next predicted color: {next_exp_color}
-------------------------------------------------------------------------------
Probability vector from softmax (%):
Green: {next_prob_vector[0,0,0] * 100}
Black: {next_prob_vector[0,0,1] * 100}
Red: {next_prob_vector[0,0,2] * 100}
-------------------------------------------------------------------------------
''')

# [SAVE FILES]
#==============================================================================
# Save the trained preprocessing systems (normalizer and encoders) for further use 
#==============================================================================
print('''Saving CCM_predictions file (as CSV)
''')
file_loc = os.path.join(GlobVar.CCM_data_path, 'CCM_predictions.csv')         
df_merged.to_csv(file_loc, index=False, sep = ';', encoding = 'utf-8')






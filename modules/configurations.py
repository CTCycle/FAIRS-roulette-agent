# Define general variables
#------------------------------------------------------------------------------
generate_model_graph = True
use_mixed_precision = True
use_tensorboard = False
XLA_acceleration = False

# Define variables for the training
#------------------------------------------------------------------------------
seed = 51
training_device = 'GPU'
neuron_baseline = 64
epochs = 500
learning_rate = 0.0001
batch_size = 512

# embedding and convolutions
#------------------------------------------------------------------------------
embedding_size = 600
kernel_size = 20

# Define variables for preprocessing
#------------------------------------------------------------------------------
use_test_data = True
invert_test = False
data_size = 1.0
test_size = 0.2
window_size = 90
output_size = 1

# mapping data
#------------------------------------------------------------------------------
color_encoding = False
categories_mapping = {0 : 'green', 1 : 'black', 2 : 'red'}




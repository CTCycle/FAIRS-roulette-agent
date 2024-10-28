import keras
from keras import losses, metrics, layers, Model
import torch

from FAIRS.commons.utils.learning.embeddings import PositionalEmbedding
from FAIRS.commons.utils.learning.transformers import RouletteTransformerEncoder
from FAIRS.commons.utils.learning.logits import QScoreNet
from FAIRS.commons.utils.learning.metrics import RouletteCategoricalCrossentropy, RouletteAccuracy
from FAIRS.commons.constants import CONFIG, STATES
from FAIRS.commons.logger import logger


# [FAIRS MODEL]
###############################################################################
class FAIRSnet: 

    def __init__(self):  
       
        self.perceptive_size = CONFIG["dataset"]["PERCEPTIVE_SIZE"] 
        self.embedding_dims = CONFIG["model"]["EMBEDDING_DIMS"]        
        self.num_heads = CONFIG["model"]["NUM_HEADS"]  
        self.num_encoders = CONFIG["model"]["NUM_ENCODERS"]         
        self.jit_compile = CONFIG["model"]["JIT_COMPILE"]
        self.jit_backend = CONFIG["model"]["JIT_BACKEND"]
        self.learning_rate = CONFIG["training"]["LEARNING_RATE"]
        self.seed = CONFIG["SEED"]
       
        self.action_size = STATES
        self.timeseries = layers.Input(shape=(self.perceptive_size,), name='timeseries')                 
        self.encoders = [RouletteTransformerEncoder(self.embedding_dims, self.num_heads, self.seed) 
                                                    for _ in range(self.num_encoders)]
        self.embedding = PositionalEmbedding(self.embedding_dims, self.perceptive_size, mask_negative=False)
        self.QNet = QScoreNet(512, self.action_size, self.seed)   
        
        
    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, model_summary=True):    

        # initialize the image encoder and the transformers encoders and decoders      
        timeseries = layers.Input(shape=(self.perceptive_size,), name='timeseries') 

        # encode images using the convolutional encoder
        embeddings = self.embedding(timeseries)         
        
        # handle the connections between transformers blocks        
        encoder_output = embeddings           
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output, training=False)       

        # apply the softmax classifier layer
        output = self.QNet(encoder_output)   
      
        
        # define the model from inputs and outputs
        model = Model(inputs=timeseries, outputs=output)                

        # define model compilation parameters such as learning rate, loss, metrics and optimizer
        loss = losses.MeanSquaredError() 
        metric = [metrics.SparseCategoricalAccuracy()]
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)          
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False)

        if self.jit_compile:
            model = torch.compile(model, backend=self.jit_backend, mode='default')

        if model_summary:
            model.summary(expand_nested=True)

        return model           
       
       
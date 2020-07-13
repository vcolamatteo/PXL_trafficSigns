# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:20:27 2020

@author: ValerioC
"""


# libs
import pandas as pd
import numpy as np
import timeit

from sklearn.utils import class_weight
from keras import backend as K
from keras.applications.resnet50 import ResNet50

from keras_preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
import matplotlib.pyplot as plt

import myUtils
import myResnet


"""
PARAMS
"""

# flag to train feature extractor
#OPTIMIZER           = 'ADAM'
OPTIMIZER           = 'SGD'
FROM_IMAGENET       = True
# data and model
NUM_CLASSES         = 21
CLASSES             = ["0", "1", "2", "3", "4","5","6", "7", "8", "9", "10","11","12", "13", "14", "15", "16","17", "18", "19", "20"]
MODEL_FC_NEURONS    = 512
MODEL_DROPOUT_RATIO = .5
BATCH_SIZE          = 16
EPOCHS              = 1
file_size           = 64 

SPLIT               = 3
CUSTOM_MODEL        = 0

# input data
if CUSTOM_MODEL==0:

    DATASET_PATH        = ["PNG\\Processed_64\\germany\\", 
                           "PNG\\Processed_64\\germany\\"]
    
    directory=          "\\Train_models_AAA_Color_per_video_split_"+str(SPLIT)+"\\"
    
    INPUT_TENSOR_SHAPE  = (file_size, file_size, 3) 
    
    color_mode="rgb"
    INITIAL_LR          = .0001
    
else:
    
    DATASET_PATH        = ["PNG\\Processed_2_Gray_64\\germany\\", 
                           "PNG\\Processed_2_Gray_64\\germany\\"] 
    
    directory=          "\\Train_models_Gray_per_video_split_"+str(SPLIT)+"\\"
    
    INPUT_TENSOR_SHAPE  = (file_size, file_size, 1) 
    
    color_mode="grayscale"
    INITIAL_LR          = .001





DATASET_INPUTS=["splits_per_video\\train_"+str(SPLIT)+".txt","splits_per_video\\validation_"+str(SPLIT)+".txt"]

# output data
MODELS_ROOT          ="Models"

TRAIN_DIR            = MODELS_ROOT+directory
myUtils.checkPath(TRAIN_DIR)



"""
LOAD TRAINING AND VALIDATION DATA
"""

print('Loading ' + DATASET_INPUTS[0])
train_data              = pd.read_csv(DATASET_INPUTS[0], delimiter='  ')
print('Loading ' + DATASET_INPUTS[1])
val_data                = pd.read_csv(DATASET_INPUTS[1], delimiter='  ')






val_truth               = val_data['label']
train_truth             = train_data['label']
class_mode              = 'categorical'

metrics_mode            = 'accuracy'


val_data['label']    = val_data['label'].astype(str)
train_data['label']  = train_data['label'].astype(str)
y_col_name                  = 'label'



#val_truth               = val_data['label']
#train_truth             = train_data['label']


class_mode              = 'categorical'
loss_mode               = 'categorical_crossentropy'

metrics_mode            = 'accuracy'
    

# convert to categorical type
val_truth    = to_categorical(val_data['label'], num_classes = NUM_CLASSES, dtype='float32')
train_truth  = to_categorical(train_data['label'], num_classes = NUM_CLASSES, dtype='float32')

    


#BAlancing
y_train = [y.argmax() for y in train_truth]
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

class_weight_dict = dict(enumerate(class_weights))


# load weights from imagenet --- fine tuning
if FROM_IMAGENET:
    WEIGHTS             = 'imagenet'

else:
    WEIGHTS             = None




    
input_train_gen     = ImageDataGenerator(samplewise_center                  = True,
                                         samplewise_std_normalization       = True,
                                         rescale               = 1./255,          
                                         rotation_range=20,
                                         zoom_range=0.35,
                                         height_shift_range=0.2,
                                         width_shift_range=0.2,
                                         shear_range=0.35,
                                         horizontal_flip=False,
                                         vertical_flip=False,
                                         fill_mode="nearest"    )
    
    
input_val_gen       = ImageDataGenerator(samplewise_center                  = True,
                                         samplewise_std_normalization       = True,#)
                                         #zca_whitening                      = True)
                                         rescale               = 1./255)                                         
                                         
input_train_data    = input_train_gen.flow_from_dataframe(train_data, 
                                                          DATASET_PATH[0], 
                                                          x_col         = 'filename', 
                                                          y_col         = y_col_name, 
                                                          has_ext       = True,
                                                          target_size   = (INPUT_TENSOR_SHAPE[0], 
                                                                           INPUT_TENSOR_SHAPE[1]), 
                                                          classes       = CLASSES, 
                                                          batch_size    = BATCH_SIZE, 
                                                          color_mode=color_mode,
                                                          shuffle       = True
                                                          )
input_val_data      = input_val_gen.flow_from_dataframe(val_data, 
                                                        DATASET_PATH[1], 
                                                        x_col         = 'filename', 
                                                        y_col         = y_col_name, 
                                                        has_ext       = True,
                                                        target_size   = (INPUT_TENSOR_SHAPE[0], 
                                                                         INPUT_TENSOR_SHAPE[1]), 
                                                        classes       = CLASSES, 
                                                        batch_size    = BATCH_SIZE, 
                                                        color_mode=color_mode,
                                                        shuffle       = False)
                                                        
# reset batch indices
input_train_data.reset()
input_val_data.reset()

"""
INSTANCING THE FEATURE EXTRACTOR
"""
    

if CUSTOM_MODEL==1:

    filters=16
    feat_extractor = myResnet.ResNet50(input_shape=file_size,starting_filters=filters, num_classes=NUM_CLASSES, max_Pooling=False)
    feat_extractor= Model(inputs  = feat_extractor.input, outputs =feat_extractor.layers[-2].get_output_at(0))
        
    
    x = Dropout(0.5)(feat_extractor.output)
    
    
    y=Dense(NUM_CLASSES,activation='softmax')(x) #final layer with softmax activation    
        # assembling the model 
    model           = Model(inputs  = feat_extractor.input, outputs = y)


else:    
    
    
    
    feat_extractor  = ResNet50(input_shape    = INPUT_TENSOR_SHAPE, 
                                      include_top    = False, 
                                      weights        = WEIGHTS,             # weights initialization
                                      input_tensor   = None, 
                                      pooling        = 'avg',                 # average pooling will be enforced    
                                      classes        = NUM_CLASSES)
    x=Dense(1024,activation='relu')(feat_extractor.output) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dropout(0.5)(x)
    x=Dense(512,activation='relu')(x) #dense layer 3
    x = Dropout(0.5)(x)
    y=Dense(NUM_CLASSES,activation='softmax')(x) #final layer with softmax activation    
    model           = Model(inputs  = feat_extractor.input, outputs = y)
    
    
    
   
"""
FINALIZING THE MODEL
"""


# save periodically the checkpoint
filepath        = TRAIN_DIR + "RESNET" +"_batch_size_"+ str(BATCH_SIZE)+"_"+'_epoch_{epoch:03d}_'+'_tr_acc_'+'{accuracy:.4f}' + '_tr_loss_'+'{loss:.4f}'+'_val_loss_'+'{val_loss:.4f}'+'_val_acc_' +'{val_accuracy:.4f}' +'.hdf5'
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor           ='val_accuracy', 
                                  verbose           = 2, 
                                  save_best_only    = True, 
                                  mode              ='auto', 
                                  period            = 1)


# complete the callbacks list
callbacks_list  = [checkpoint]
 
                

if OPTIMIZER == 'Adam':
    print("ADAM")
    model_optimizer         = Adam(lr=INITIAL_LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
else:
    model_optimizer         = SGD(lr=INITIAL_LR, momentum=0.99, decay = INITIAL_LR/EPOCHS, nesterov = False)


model.compile(optimizer     = model_optimizer, loss          = loss_mode,    metrics       = [metrics_mode])

# print the summary
model.summary()


print("lr: ",K.eval(model.optimizer.lr))

"""
START THE TRAINING
"""

# finally, train the model
print('Model Training Started ...')
tStart          = timeit.default_timer()
# dimensioning the proper step size per epoch

STEP_SIZE_TRAIN = (train_data['label'].shape[0]//BATCH_SIZE)
STEP_SIZE_VAL   = (val_data['label'].shape[0]//BATCH_SIZE)




# fit the generator
history=model.fit_generator(generator   = input_train_data,
                    steps_per_epoch     = STEP_SIZE_TRAIN,
                    epochs              = EPOCHS,
                    validation_data     = input_val_data,
                    validation_steps    = STEP_SIZE_VAL,
                    verbose             = 1, 
                    callbacks           = callbacks_list,
                    class_weight        = class_weights)





tElapsed        = timeit.default_timer() - tStart
print('... done in ' + str(tElapsed) + ' [s].')




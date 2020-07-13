# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 20:47:00 2020

@author: ValerioC
"""



import pandas as pd
import numpy as np

from sklearn.metrics import classification_report

from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model

from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys
import myUtils

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



"""
PARAMS
"""




# data and model
NUM_CLASSES         = 21
CLASSES             = ["0", "1", "2", "3", "4","5","6", "7", "8", "9", "10","11","12", "13", "14", "15", "16","17", "18", "19", "20"]
EPOCHS              = 200
file_size           = 64
SPLIT               = 3
CUSTOM_MODEL        = 0


# input data
if CUSTOM_MODEL==0:

    DATASET_PATH        = ["PNG\\Processed_64\\germany\\", 
                           "PNG\\Processed_64\\belgium\\"]
    
    directory=          "\\Train_models_Color_per_video_split_"+str(SPLIT)+"\\"
    
    INPUT_TENSOR_SHAPE  = (file_size, file_size, 3) 
    
    color_mode="rgb"
    
else:
    
    DATASET_PATH        = ["PNG\\Processed_2_Gray_64\\germany\\", 
                           "PNG\\Processed_2_Gray_64\\belgium\\"] 
    
    directory=          "\\Train_models_Gray_per_video_split_"+str(SPLIT)+"\\"
    
    INPUT_TENSOR_SHAPE  = (file_size, file_size, 1) 
    
    color_mode="grayscale"


DATASET_INPUTS=["splits_per_video\\validation_"+str(SPLIT)+".txt","splits_per_video\\belgium.txt"]


# output data
MODELS_ROOT          ="Models"

TRAIN_DIR            = MODELS_ROOT+directory




"""
LOAD TRAINING AND VALIDATION DATA
"""


print('Loading ' + DATASET_INPUTS[0] )
val_data              = pd.read_csv(DATASET_INPUTS[0], delimiter='  ')

print('Loading ' + DATASET_INPUTS[1] )
test_data                = pd.read_csv(DATASET_INPUTS[1], delimiter='  ')




test_truth               = test_data['label']
val_truth                = val_data['label']
loss_mode               = 'categorical_crossentropy'
metrics_mode            = 'accuracy'


test_data['label']    = test_data['label'].astype(str)
val_data['label']     =  val_data['label'].astype(str)

y_col_name                  = 'label'



   

val_truth               = val_data['label']
test_truth             = test_data['label']
class_mode              = 'categorical'
loss_mode               = 'categorical_crossentropy'
metrics_mode            = 'accuracy'
    
test_truth=np.asarray(test_truth,'int')
test_truth=np.reshape(test_truth,(len(test_truth),1))


# set the input data generators
input_val_gen       = ImageDataGenerator(samplewise_center                 = True,
                                         samplewise_std_normalization       = True,                                         
                                         rescale                            = 1./255)    


input_test_gen       = ImageDataGenerator(samplewise_center                 = True,
                                         samplewise_std_normalization       = True,
                                         rescale                            = 1./255)                                         
                                  
input_val_data    = input_val_gen.flow_from_dataframe(val_data, 
                                                        DATASET_PATH[0], 
                                                        x_col         = 'filename', 
                                                        y_col         = y_col_name, 
                                                        has_ext       = True,
                                                        target_size   = (INPUT_TENSOR_SHAPE[0], 
                                                                         INPUT_TENSOR_SHAPE[1]), 
                                                        classes       = CLASSES, 
                                                        shuffle       = False,  
                                                        color_mode    = color_mode,
                                                        class_mode    = class_mode)
                                                        
input_test_data      = input_test_gen.flow_from_dataframe(test_data, 
                                                        DATASET_PATH[1], 
                                                        x_col         = 'filename', 
                                                        y_col         = y_col_name, 
                                                        has_ext       = True,
                                                        target_size   = (INPUT_TENSOR_SHAPE[0], 
                                                                         INPUT_TENSOR_SHAPE[1]), 
                                                        classes       = CLASSES, 
                                                        shuffle       = False,  
                                                        color_mode    = color_mode,
                                                        class_mode    = class_mode)
# reset batch indices
input_val_data.reset()
input_test_data.reset()


#label_map = (input_test_data.class_indices)
#print(label_map)
#from glob import glob
#class_names = glob("C:\\Users\\ValerioC\\Downloads\\PXL\\traffic_signs_train\\traffic_signs_train\\PNG\\Processed_64\\belgium\\*") # Reads all the folders in which images are present
#class_names = sorted(class_names) # Sorting them
#name_id_map = dict(zip(class_names, range(len(class_names))))
test_truth = np.array(input_test_data.classes)
#print(name_id_map)

val_truth = np.array(input_val_data.classes)

   
# split 2
#file_to_load=TRAIN_DIR+"RESNET50_batch_size_16__epoch_072__tr_acc_0.9983_tr_loss_0.0060_val_loss_0.0000_val_acc_0.9953.hdf5"

#split 1
#file_to_load=TRAIN_DIR+"RESNET50_batch_size_16__epoch_050__tr_acc_0.9979_tr_loss_0.0098_val_loss_0.0000_val_acc_0.9973.hdf5"

#split 3
#file_to_load=TRAIN_DIR+"RESNET50_batch_size_16__epoch_123__tr_acc_0.9966_tr_loss_0.0130_val_loss_0.0000_val_acc_0.9905.hdf5"        

#split 3 GRAY
#file_to_load=TRAIN_DIR+"RESNET50_batch_size_16__epoch_070__tr_acc_0.9974_tr_loss_0.0087_val_loss_0.0000_val_acc_0.9975.hdf5"        



# Resnet50 split 1
#file_to_load=TRAIN_DIR+"RESNET50_batch_size_16__epoch_045__tr_acc_0.9951_tr_loss_0.0168_val_loss_0.0000_val_acc_0.9960.hdf5"


# Resnet50 split 2
#file_to_load=TRAIN_DIR+"RESNET50_batch_size_16__epoch_033__tr_acc_0.9987_tr_loss_0.0055_val_loss_0.0000_val_acc_0.9928.hdf5"


# Resnet50 split 3
file_to_load=TRAIN_DIR+"RESNET50_batch_size_16__epoch_037__tr_acc_0.9981_tr_loss_0.0065_val_loss_0.0000_val_acc_0.9948.hdf5"



print("FILE TO L OAD: ",file_to_load)


model=load_model(file_to_load)


 
# print the summary
model.summary()




test_truth_pred = model.predict_generator(input_test_data, #or you can place here some test videos as x=np.array(exp_8_data) ,
                                       verbose=1)

val_truth_pred = model.predict_generator(input_val_data, #or you can place here some test videos as x=np.array(exp_8_data) ,
                                       verbose=1)



classes=["Speed limit 20","Speed limit 50","Speed limit 70","no Overtaking","Roundabout",
         "Priority road","Give way","Stop","Road closed","no Heavy goods vehicles",
         "no Entry","Obstacles" ,"Left hand curve","Right hand curve","Kepp straight ahead",
         "Slippery road","Keep straight","Construction ahead","Rough road","Traffic lights","School ahead"]




img_path=TRAIN_DIR+"results\\"
dst=img_path
myUtils.checkPath(dst)
plot_model(model, show_shapes = True, to_file=dst + 'model.png')
np.savetxt(dst+"test_pred_file_out.txt",np.reshape(test_truth_pred,(test_truth_pred.shape[0],NUM_CLASSES)),fmt='%f')
np.savetxt(dst+"val_pred_file_out.txt",np.reshape(val_truth_pred,(val_truth_pred.shape[0],NUM_CLASSES)),fmt='%f')

# Thresholding values
th=["0.00","0.50","0.75","0.90","0.95","0.99"]

test_truth=test_truth.transpose()


dst_exclusives=dst+"exclusives\\"
myUtils.checkPath(dst_exclusives)
test_truth=np.reshape(test_truth,(1,test_truth.shape[0]))
val_truth=np.reshape(val_truth,(1,val_truth.shape[0]))

test_truth_pred_th_exclusive_1,test_truth_exc1=myUtils.threshold_pred(np.reshape(test_truth_pred,(test_truth_pred.shape[0],NUM_CLASSES)), float(th[1]), label_to_return=-1, val_truth=test_truth)
np.savetxt(dst_exclusives+"test_pred_file_out_th_exclusive_at_th_"+th[1]+".txt",test_truth_pred_th_exclusive_1.transpose(),fmt='%d')
np.savetxt(dst_exclusives+"test_truth_exclusive_at_th_"+th[1]+".txt",test_truth_exc1.transpose(),fmt='%d')

test_truth_pred_th_exclusive_2,test_truth_exc2=myUtils.threshold_pred(np.reshape(test_truth_pred,(test_truth_pred.shape[0],NUM_CLASSES)), float(th[2]), label_to_return=-1, val_truth=test_truth)
np.savetxt(dst_exclusives+"test_pred_file_out_th_exclusive_at_th_"+th[2]+".txt",test_truth_pred_th_exclusive_2.transpose(),fmt='%d')
np.savetxt(dst_exclusives+"test_truth_exclusive_at_th_"+th[2]+".txt",test_truth_exc2.transpose(),fmt='%d')

test_truth_pred_th_exclusive_3,test_truth_exc3=myUtils.threshold_pred(np.reshape(test_truth_pred,(test_truth_pred.shape[0],NUM_CLASSES)), float(th[3]), label_to_return=-1, val_truth=test_truth)
np.savetxt(dst_exclusives+"test_pred_file_out_th_exclusive_at_th_"+th[3]+".txt",test_truth_pred_th_exclusive_3.transpose(),fmt='%d')
np.savetxt(dst_exclusives+"test_truth_exclusive_at_th_"+th[3]+".txt",test_truth_exc3.transpose(),fmt='%d')

test_truth_pred_th_exclusive_4,test_truth_exc4=myUtils.threshold_pred(np.reshape(test_truth_pred,(test_truth_pred.shape[0],NUM_CLASSES)), float(th[4]), label_to_return=-1, val_truth=test_truth)
np.savetxt(dst_exclusives+"test_pred_file_out_th_exclusive_at_th_"+th[4]+".txt",test_truth_pred_th_exclusive_4.transpose(),fmt='%d')
np.savetxt(dst_exclusives+"test_truth_exclusive_at_th_"+th[4]+".txt",test_truth_exc4.transpose(),fmt='%d')

test_truth_pred_th_exclusive_5,test_truth_exc5=myUtils.threshold_pred(np.reshape(test_truth_pred,(test_truth_pred.shape[0],NUM_CLASSES)), float(th[5]), label_to_return=-1, val_truth=test_truth)
np.savetxt(dst_exclusives+"test_pred_file_out_th_exclusive_at_th_"+th[5]+".txt",test_truth_pred_th_exclusive_5.transpose(),fmt='%d')
np.savetxt(dst_exclusives+"test_truth_exclusive_at_th_"+th[5]+".txt",test_truth_exc5.transpose(),fmt='%d')




f2 = open(DATASET_INPUTS[1], 'r')
lines = f2.readlines()[1:]
f2.close()    


test_truth_pred = np.argmax(test_truth_pred, axis=1)
test_truth_pred=np.reshape(test_truth_pred,(1,test_truth_pred.shape[0]))


val_truth_pred = np.argmax(val_truth_pred, axis=1)
val_truth_pred=np.reshape(val_truth_pred,(1,val_truth_pred.shape[0]))



np.savetxt(dst+"test_true_file.txt",test_truth.transpose(),fmt='%d')
np.savetxt(dst+"test_pred_file.txt",test_truth_pred.transpose(),fmt='%d')


np.savetxt(dst+"val_true_file.txt",val_truth.transpose(),fmt='%d')
np.savetxt(dst+"val_pred_file.txt",val_truth_pred.transpose(),fmt='%d')


confMatrixVal = confusion_matrix(test_truth[0,:], test_truth_pred[0,:],np.arange(0,NUM_CLASSES))

confMatrixValidation_set = confusion_matrix(val_truth[0,:], val_truth_pred[0,:],np.arange(0,NUM_CLASSES))



# Exclisives
confMatrixtest_exclusive_1 = confusion_matrix(test_truth_exc1, test_truth_pred_th_exclusive_1,np.arange(0,NUM_CLASSES))
confMatrixtest_exclusive_2 = confusion_matrix(test_truth_exc2, test_truth_pred_th_exclusive_2,np.arange(0,NUM_CLASSES))
confMatrixtest_exclusive_3 = confusion_matrix(test_truth_exc3, test_truth_pred_th_exclusive_3,np.arange(0,NUM_CLASSES))
confMatrixtest_exclusive_4 = confusion_matrix(test_truth_exc4, test_truth_pred_th_exclusive_4,np.arange(0,NUM_CLASSES))
confMatrixtest_exclusive_5 = confusion_matrix(test_truth_exc5, test_truth_pred_th_exclusive_5,np.arange(0,NUM_CLASSES))




original_stdout = sys.stdout # Save a reference to the original standard output
with open(dst+'report_test.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(classification_report(test_truth[0,:],test_truth_pred[0,:],target_names=classes))    
    sys.stdout = original_stdout # Reset the standard output to its original value
    
    
original_stdout = sys.stdout # Save a reference to the original standard output
with open(dst+'report_valdation.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(classification_report(test_truth[0,:],test_truth_pred[0,:],target_names=classes))    
    sys.stdout = original_stdout # Reset the standard output to its original value



labels_distribution=np.zeros((NUM_CLASSES),np.int)
for n_class in range(0,NUM_CLASSES):
    print(n_class,len(test_truth[test_truth==n_class]))
    labels_distribution[n_class]=len(test_truth[test_truth==n_class])


# Save Normal Confusion Matrix
fontsize=15
plt.figure()
plt.figure(figsize=(35,20))
myUtils.plot_confusion_matrix_merged(confMatrixVal, classes=classes, normalize=False,title='Confusion Matrix Test Set\n',fontsize=fontsize)
plt.savefig(dst+"Confusion_Matrix_test_set.png",bbox_inches="tight")



fontsize=15
plt.figure()
plt.figure(figsize=(35,20))
myUtils.plot_confusion_matrix_merged(confMatrixValidation_set, classes=classes, normalize=False,title='Confusion Matrix Validation Set\n',fontsize=fontsize)
plt.savefig(dst+"Confusion_Matrix_validation_set.png",bbox_inches="tight")




#save accuracy array
acc=np.round((np.diagonal(confMatrixVal)/labels_distribution)*100,2)
print(acc)
np.save(dst+"accuracies.npy",acc)





# Exclusive
dst_plot_excl=dst_exclusives+"plots\\"
myUtils.checkPath(dst_plot_excl)


accuracy_vec_excl=np.zeros((len(th)),np.int)

fontsize=12

plt.figure()
plt.figure(figsize=(35,20))
accuracy_vec_excl[0]=myUtils.plot_confusion_matrix_merged_excl(confMatrixVal, classes=classes, normalize=False,title='Confusion Matrix No Thresholded\n',fontsize=fontsize, labels_distribution=labels_distribution)
plt.savefig(dst_plot_excl+"Confusion Matrix Exclusive at th_ "+th[0]+".png",bbox_inches="tight")

plt.figure()
plt.figure(figsize=(35,20))
accuracy_vec_excl[1]=myUtils.plot_confusion_matrix_merged_excl(confMatrixtest_exclusive_1, classes=classes, normalize=False,title='Confusion Matrix Exclusive at Threshold '+th[1] +'\n',fontsize=fontsize, labels_distribution=labels_distribution)
plt.savefig(dst_plot_excl+"Confusion Matrix Exclusive at th_ "+th[1]+".png",bbox_inches="tight")


plt.figure()
plt.figure(figsize=(35,20))
accuracy_vec_excl[2]=myUtils.plot_confusion_matrix_merged_excl(confMatrixtest_exclusive_2, classes=classes, normalize=False,title='Confusion Matrix Exclusive at Threshold '+th[2]+'\n',fontsize=fontsize, labels_distribution=labels_distribution)
plt.savefig(dst_plot_excl+"Confusion Matrix Exclusive at th_ "+th[2]+".png",bbox_inches="tight")

plt.figure()
plt.figure(figsize=(35,20))
accuracy_vec_excl[3]=myUtils.plot_confusion_matrix_merged_excl(confMatrixtest_exclusive_3, classes=classes, normalize=False,title='Confusion Matrix Exclusive at Threshold '+th[3]+'\n',fontsize=fontsize, labels_distribution=labels_distribution)
plt.savefig(dst_plot_excl+"Confusion Matrix Exclusive at th_ "+th[3]+".png",bbox_inches="tight")

plt.figure()
plt.figure(figsize=(35,20))
accuracy_vec_excl[4]=myUtils.plot_confusion_matrix_merged_excl(confMatrixtest_exclusive_4, classes=classes, normalize=False,title='Confusion Matrix Exclusive at Threshold '+th[4]+'\n',fontsize=fontsize, labels_distribution=labels_distribution)
plt.savefig(dst_plot_excl+"Confusion Matrix Exclusive at th_ "+th[4]+".png",bbox_inches="tight")

plt.figure()
plt.figure(figsize=(35,20))
accuracy_vec_excl[5]=myUtils.plot_confusion_matrix_merged_excl(confMatrixtest_exclusive_5, classes=classes, normalize=False,title='Confusion Matrix Exclusive at Threshold '+th[5]+'\n',fontsize=fontsize, labels_distribution=labels_distribution)
plt.savefig(dst_plot_excl+"Confusion Matrix Exclusive at th_ "+th[5]+".png",bbox_inches="tight")


print(confMatrixtest_exclusive_1)
print("--------")
    
predictions_distribution=np.zeros((len(th)),np.float)

predictions_distribution[0]=1.00
predictions_distribution[1]=np.round(((np.sum(confMatrixtest_exclusive_1)/np.sum(confMatrixVal))),2)
predictions_distribution[2]=np.round(((np.sum(confMatrixtest_exclusive_2)/np.sum(confMatrixVal))),2)
predictions_distribution[3]=np.round(((np.sum(confMatrixtest_exclusive_3)/np.sum(confMatrixVal))),2)
predictions_distribution[4]=np.round(((np.sum(confMatrixtest_exclusive_4)/np.sum(confMatrixVal))),2)
predictions_distribution[5]=np.round(((np.sum(confMatrixtest_exclusive_5)/np.sum(confMatrixVal))),2)

print(predictions_distribution)


plt.close("all")


myUtils.barPolt(accuracy_vec_excl,th,dst_plot_excl)


myUtils.barPolt_pred(predictions_distribution*100,th,dst_plot_excl)








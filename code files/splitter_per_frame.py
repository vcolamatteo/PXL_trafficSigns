# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:59:32 2020

@author: ValerioC


# generate random training/validation splits

"""

import myUtils
import numpy as np

import random


percentage=0.2  # validation set = 20% of training set 
splits=3  # number of splits to generate

dst_folder="splits\\"
myUtils.checkPath(dst_folder)
out_1=open(dst_folder+"validation_1.txt","w")
out_1.write("filename  label\n")
out_2=open(dst_folder+"validation_2.txt","w")
out_2.write("filename  label\n")
out_3=open(dst_folder+"validation_3.txt","w")
out_3.write("filename  label\n")
    
train_1=open(dst_folder+"train_1.txt","w")
train_1.write("filename  label\n")
train_2=open(dst_folder+"train_2.txt","w")
train_2.write("filename  label\n")
train_3=open(dst_folder+"train_3.txt","w")
train_3.write("filename  label\n")

init_path="PNG\\Processed_64\\germany\\"

folders_1=myUtils.getFolderList(init_path)[0]
print(folders_1)

for i in range(0,len(folders_1)):
    
   vec=myUtils.loadFile([".png"],folders_1[i]) # read all image files
   #print(vec)
   
   # takes a population and a sample size k and returns k random members of the population...
   # generate 3 times the number of samples, 60% of training set and then split them in 3 parts
   nums=random.sample(range(0, len(vec)), splits*int(percentage*len(vec))) 

   if len(np.unique(nums))!=len(nums):
       print("ERROR\n")
       exit()
   
    
   # split the random array in 3 parts, with sorting
   # each element of the 3 vectors is now a unique number of lines in input file (germany.txt)
   s1=np.sort(nums[0:int(len(nums)/splits)])
   s2=np.sort(nums[int(len(nums)/splits):2*int(len(nums)/splits)])
   s3=np.sort(nums[2*int(len(nums)/splits):])
   
   
   # write splits on file for validation and traning #############
   for ff in range(0,len(s1)):
       out_1.write(vec[s1[ff]][myUtils.find_nth_Occ(vec[s1[ff]],"\\",3)+1:] + "  "+ vec[s1[ff]][myUtils.find_nth_Occ(vec[s1[ff]],"\\",3)+1:vec[s1[ff]].rfind("\\")] +"\n")
       
   for ff in range(0,len(s2)):       
       out_2.write(vec[s2[ff]][myUtils.find_nth_Occ(vec[s2[ff]],"\\",3)+1:] + "  "+ vec[s2[ff]][myUtils.find_nth_Occ(vec[s2[ff]],"\\",3)+1:vec[s2[ff]].rfind("\\")] +"\n")
       
   for ff in range(0,len(s3)):       
       out_3.write(vec[s3[ff]][myUtils.find_nth_Occ(vec[s3[ff]],"\\",3)+1:] + "  "+ vec[s3[ff]][myUtils.find_nth_Occ(vec[s3[ff]],"\\",3)+1:vec[s3[ff]].rfind("\\")] +"\n")
       
       
   for ff in range(0,len(vec)):       
       label=vec[ff][myUtils.find_nth_Occ(vec[ff],"\\",3)+1:vec[ff].rfind("\\")]

       if ff not in s1:
           train_1.write(vec[ff][myUtils.find_nth_Occ(vec[ff],"\\",3)+1:]+"  "+ label + "\n")
       if ff not in s2:
           train_2.write(vec[ff][myUtils.find_nth_Occ(vec[ff],"\\",3)+1:]+"  "+ label + "\n")
       if ff not in s3:
           train_3.write(vec[ff][myUtils.find_nth_Occ(vec[ff],"\\",3)+1:]+"  "+ label + "\n")
   #############################################################        
           
       
   
out_1.close()
out_2.close()
out_3.close()

train_1.close()
train_2.close()
train_3.close()
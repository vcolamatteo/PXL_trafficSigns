# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:59:32 2020

@author: ValerioC


# generate random training/validation splits

"""

import myUtils
import numpy as np

import random
import copy


def getVideo(s,length=5):

    s1=""
    for i in range(length-len(str(s))):
        s1=s1+"0"
    s1=s1+str(s)+"_"
    
    return s1


percentage=0.2  # validation set = 20% of training set 
splits=3  # number of splits to generate

dst_folder="splits_per_video\\"
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
   
   s1_list=[]
   s2_list=[]
   s3_list=[]
    
   vec=myUtils.loadFile([".png"],folders_1[i])
   vec_copy=copy.deepcopy(vec)
    
   for j in range(0,len(vec)):
        
        vec_copy[j]=vec[j][vec[j].rfind("\\")+1:vec[j].rfind("_")]

    
   # takes a population and a sample size k and returns k random members of the population...
   # generate 3 times the number of samples, 60% of training set and then split them in 3 parts
   #print(splits*np.round(percentage*np.count_nonzero(np.unique(vec_copy))))
   nums=random.sample(range(0, np.count_nonzero(np.unique(vec_copy))), splits*int(np.round(percentage*np.count_nonzero(np.unique(vec_copy))))) 
   print(nums)
   if len(np.unique(nums))!=len(nums):
       print("ERROR\n")
       exit()
   
#   print(len(nums[0:int(len(nums)/3)]))
#   print(len(nums[int(len(nums)/3):2*int(len(nums)/3)]))
#   print(len(nums[2*int(len(nums)/3):]))
   
    
   # split the random array in 3 parts, with sorting
   # each element of the 3 vectors is now a unique number of lines in input file (germany.txt)
   s1=np.sort(nums[0:int(len(nums)/splits)])
   s2=np.sort(nums[int(len(nums)/splits):2*int(len(nums)/splits)])
   s3=np.sort(nums[2*int(len(nums)/splits):])
   print(s1)
   print(s2)
   print(s3)
   
   
   # write splits on file for validation and traning #############
   for ff in range(0,len(s1)):       
       s1_list.append(getVideo(s1[ff]))
   for ff in range(0,len(s2)):       
       s2_list.append(getVideo(s2[ff]))
   for ff in range(0,len(s3)):       
       s3_list.append(getVideo(s3[ff]))

   
    
   for ff in range(0,len(vec)):
       
       
       label=vec[ff][myUtils.find_nth_Occ(vec[ff],"\\",3)+1:vec[ff].rfind("\\")]       

       if vec_copy[ff]+"_" in s1_list:
               out_1.write(vec[ff][myUtils.find_nth_Occ(vec[ff],"\\",3)+1:] + "  "+ label+"\n")
       else:
               train_1.write(vec[ff][myUtils.find_nth_Occ(vec[ff],"\\",3)+1:]+"  "+ label + "\n")
       

       if vec_copy[ff]+"_" in s2_list:
               out_2.write(vec[ff][myUtils.find_nth_Occ(vec[ff],"\\",3)+1:] + "  "+ label+"\n")
       else:
               train_2.write(vec[ff][myUtils.find_nth_Occ(vec[ff],"\\",3)+1:]+"  "+ label + "\n")
               
               
       if vec_copy[ff]+"_" in s3_list:
               out_3.write(vec[ff][myUtils.find_nth_Occ(vec[ff],"\\",3)+1:] + "  "+ label+"\n")
       else:
               train_3.write(vec[ff][myUtils.find_nth_Occ(vec[ff],"\\",3)+1:]+"  "+ label + "\n")
   
   
       
           
       
   
out_1.close()
out_2.close()
out_3.close()

train_1.close()
train_2.close()
train_3.close()
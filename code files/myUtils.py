# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 09:37:49 2019

@author: Akronos
"""
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import sys
import glob
import itertools
from datetime import datetime

#from win32api import GetSystemMetrics   # having to be installed pywin32 (pip install pywin32)

import platform


def printAll_mode():
    
    np.set_printoptions(threshold=sys.maxsize)

def getDevice_path():
    
    return os.getcwd()[0:os.getcwd().find("\\")]+"\\"

def getPythonVersion(): 

    s=(platform.python_version())
    
    return s[2] 

def getOpenCV_version(output=0, complete_output=0):

    if output==1:
        # Print version string
        print ("OpenCV version :  {0}".format(cv2.__version__))
     
    
    # Extract major, minor, and subminor version numbers 
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    
    '''
    print ("Major version :  {0}".format(major_ver))
    print ("Minor version :  {0}".format(minor_ver))
    print ("Submior version :  {0}".format(subminor_ver))
     
    if int(major_ver) < 3 :
        print("YES")
            else :
        print("NO")

    '''
    if complete_output==1:
        return (major_ver, minor_ver, subminor_ver)
    else:
        return int(major_ver)
    


import math
def distance_between_two_points(a,b):

# as alternative:
#    from scipy.spatial import distance
#    a=[272, 498]
#    b=[268, 502]
#    print(distance.euclidean(a,b))
    
    return math.hypot((a[0]-b[0],a[1]-b[1]))

    
def angleBetweenTwoPoints(p2,p1, correction=1):

    #exampe: 
    # p2 is the arriving point, p1 the starting one.
    # p1=[490,70]
    #p2=[436,136]
    #print(math.degrees(math.atan2(p2[1]-p1[1],p2[0]-p1[0])))   -->>> 129.28940686250036
    # must add -1 sign for the same result of imageJ
    if correction==1:
        return (-1)*math.degrees(math.atan2(p2[1]-p1[1],p2[0]-p1[0]))
    else:
        return math.degrees(math.atan2(p2[1]-p1[1],p2[0]-p1[0]))



def safe_crop(img, bbox):
    
   # bbox cordinates to crop: [y1, y2, x1, x2]
   y1, y2, x1, x2 = bbox
   
   if x2 > img.shape[1] :
       x2=img.shape[1]
   if x1 <0:
       x1=0
   if y2 > img.shape[0] :
       y2=img.shape[0]
   if y1 <0:
       y1=0    
     
   return img[y1:y2, x1:x2, :]


import os
import subprocess

'''
screen_height=1080 
screen_width=1920
'''

def getScreenResolution(verbose=0):

    import ctypes
    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    
    if verbose!=0:
        print(screensize)
    
    return screensize


def writeArray (name, array, intValues=1,precision=3):
    
    if intValues==1:
        np.savetxt(name, array, fmt='% 4d')
    else:
        c=str("\'%1."+str(precision)+"f\'")
        print(c)
        np.savetxt(name, array, fmt="%1."+str(precision)+"f")
        


def writeFile(nameFile, stringToWrite="lkghdlòhlòdhf\nsdfsfòljgslfò\n", mode="w"):


        file = open(nameFile,mode) 
        file.write(stringToWrite) 
         
        file.close() 
        
        return file



def printArray(array, writeOnFile=1, ext=".txt"):

    if(writeOnFile==0):

          for i in range(len(array/1000)):
            for j in range(len(array[i])):
                print(array[i][j], end=' ')

          print() 
    else:
        writeFile(array+ext,array)
#os.system("pause");



###############



def ifExists(file):
    # return 1 if file exists, else return 0
    
    if os.path.exists(file):
        return 1
    else:
        return 0
    

def check(nameFile):
    
    if ifExists(nameFile)!=1:
        print("ERROR: file ", nameFile, " not exists...")
        os.system("pause")
        sys.exit()
        


def myRead_plt(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.waitforbuttonpress(0) # this will wait for indefinite time
    plt.close()
    

def imread(img, mode=-1):
    
    '''
    cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
    cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
    cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
    Note Instead of these three flags, you can simply pass integers 1, 0 or -1 respectively.
    '''
    
    file=img
    if (mode==0):
       img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    elif(mode==1):
       img = cv2.imread(img, cv2.IMREAD_COLOR)
    else:
       img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    
    if img is None:
       print("ERROR!\nImage ", file ," not found!")
       os.system("pause")
       sys.exit()
    else:
       return img

def imshow(img, title="Img",getResolution=False, time=0,destroy=True,scale=0.6):
    
    if getResolution==False:
 
        screen_height=1080 
        screen_width=1920
        
    else:
        screen_width,screen_height=getScreenResolution()
    
    #print(screen_width,screen_height)
    
    height, width = img.shape[:2]
    if (height>=screen_height*scale and width>=screen_width*scale):
        #print("INFO  1 ",height,width,screen_height*scale,screen_width*scale)
        cv2.namedWindow(title)
        a=int(height/(height/(screen_height*scale)))
        b=int(width/(width/(screen_width*scale)))
        #print(a,b)
        img=cv2.resize(img,(b,a))
        #cv2.resizeWindow(title,int(b),int(a))
    elif (height>=screen_height*scale and width<screen_width*scale):
        #print("INFO  2 ",height,width,screen_height*scale,screen_width*scale)
        cv2.namedWindow(title)
        a=int(height/(height/(screen_height*scale)))
        b=int(width/(height/(screen_height*scale)))
        #print(a,b)
        img=cv2.resize(img,(b,a))
    elif (height<screen_height*scale and width>=screen_width*scale):
        #print("INFO 3  ",height,width,screen_height*scale,screen_width*scale)
        cv2.namedWindow(title)
        b=int(width/(width/(screen_width*scale)))
        a=int(height/(width/(screen_width*scale)))
        #print(a,b)
        img=cv2.resize(img,(b,a))
        
        
        
    cv2.imshow(title,img)
    cv2.waitKey(time)
    if (destroy==True):
      cv2.destroyWindow(title)
    
    
def fileSplitting(file):
    
 filename=(file[file.find('\\')+1:len(file)-4])
 
 ext=file[len(file)-4:len(file)]

 return filename, ext




def scale(x, out_range=(0, 1), axis=None):
    domain = np.min(x, axis), np.max(x, axis)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2



def loadFile(ext,path=os.getcwd()):
    
#    print("ddddddd  ",path[len(path)-2:len(path)])
#    if (path[len(path)-2:len(path)]) == '//':
#       path=path[0:len(path)-1]
#       print("path ",path)
    
    vec=[]
    #ext=[".jpg",".png"]
    for i in ext:
      #print(path+'//*'+ i)
      vec.extend(glob.glob(path+'//*'+ i))
    
    
    #print(len(vec))
    #print(vec[0],vec[10])
    #os.system("pause")
    #sys.exit()
    
    return vec


def loadFile_recursive(ext,path=os.getcwd()):
    

    cfiles = []
    for root, dirs, files in os.walk(path):
      for file in files:
        #print(file)
        for i in ext:
            if file.endswith(i):
                cfiles.append(os.path.join(root, file))
    #print(cfiles)
    
    #for i in range(0, len(cfiles)):
    #    print(cfiles[i])
    
    return cfiles
    


def checkPath(path):
    #path=os.getcwd()+'\\'+'background images'
    if not os.path.exists(path):
        os.makedirs(path)
    return 


FILEBROWSER_PATH = os.path.join(os.getenv('WINDIR'), 'explorer.exe')

def explore(path):
    # explorer would choke on forward slashes
    path = os.path.normpath(path)

    if os.path.isdir(path):
        subprocess.run([FILEBROWSER_PATH, path])
    elif os.path.isfile(path):
        subprocess.run([FILEBROWSER_PATH, '/select,', os.path.normpath(path)])
        

def findKeyboardResponce():
    print("--------------------------------------")
    print("Press any key on your keyboard:")
    print("--------------------------------------\n\n\n")
    # Create a blank 300x300 black image
    image = np.zeros((300, 300, 3), np.uint8)
    # Fill image with red color(set each pixel to red)
    image[:] = (0, 0, 255)
 
    while(1):
     cv2.imshow('img',image)
     k = cv2.waitKey(33)
     if k==27:    # Esc key to stop
        break
     elif k==-1:  # normally -1 returned,so don't print it
        continue
     else:
        print ("numeric key pressed: ",k) # else print its value
        

def createBlackOrWhiteImage(width, height, channels=1, black=1):
    
    if black==1:
        color=0
    else:
        color=255
    
    image = np.zeros((width, height, channels), np.uint8)
    # Fill image with red color(set each pixel to red)
    if channels==1:
        image[:] = color
    else:
        image[:] = (color, color, color)
    
    return image



# STRING
#example using: print(line[find_nth_Occ(line,',',2)+1:find_nth_Occ(line,',',3)])

def find_nth_Occ(string, substringToSearch, n):
    start = string.find(substringToSearch)
    while start >= 0 and n > 1:
        start = string.find(substringToSearch, start+len(substringToSearch))
        n -= 1
    return start


def find_nearest(array, value):
    
    '''
    EXAMPLE USAGE
    array = np.random.random(10)
    print(array)
    # [ 0.21069679  0.61290182  0.63425412  0.84635244  0.91599191  0.00213826
    #   0.17104965  0.56874386  0.57319379  0.28719469]
    
    value = 0.5
    
    print(find_nearest(array, value))
    # 0.568743859261
    '''
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    '''
    print("----------------")
    print(array.shape)
    print(array)
    print(idx)
    '''
    
    return array[idx]



####### xml functions #################
        
        
import xml.etree.ElementTree as ET

def readXML_file(nameFile):
    
    check(nameFile)
        
    tree = ET.parse(nameFile)
    #className=np.array([],np.str)
    names= tree.findall('object')
    
    boundBox=np.array([],np.int)
    for c in names:
        name=c.find('name').text
        if name!="HANDLE":
            #print(name)
            xmin=c.find('bndbox/xmin')
            xmax=c.find('bndbox/ymin')
            ymin=c.find('bndbox/xmax')
            ymax=c.find('bndbox/ymax')
            boundBox=np.append(boundBox,(int(xmin.text), int(xmax.text), int(ymin.text), int(ymax.text)))
    
    #print(boundBox)
    
    boundBox=np.reshape(boundBox,(int(boundBox.shape[0]/4),4))
    
    return boundBox

def modifyXML_file(nameFile, dst=""):
    
    
    if dst=="":
       dst=nameFile
    
    tree = ET.parse(nameFile+".xml")
    
    filename=tree.find("filename").text
    ext=filename[len(filename)-4:len(filename)]
    filename=filename[0:len(filename)-4]+"_AUG"+ext
    tree.find("filename").text=filename
    
    file = tree.find("path").text
    file=file[0:file.rfind("\\")+1]+filename
    tree.find("path").text=file
    
    #print("AAAA  " ,nameFile+"_AUG.xml")
    tree.write(dst+"_AUG.xml")

def Read_and_ModifyXML_file(nameFile, dst=""):
    
    if dst=="":
       dst=nameFile
    
    tree = ET.parse(nameFile+".xml")
    
    #className=np.array([],np.str)
    names= tree.findall('object')
    
    boundBox=np.array([],np.int)
    for c in names:
        name=c.find('name').text
        if name!="HANDLE":
            #print(name)
            xmin=c.find('bndbox/xmin')
            xmax=c.find('bndbox/ymin')
            ymin=c.find('bndbox/xmax')
            ymax=c.find('bndbox/ymax')
            boundBox=np.append(boundBox,(int(xmin.text), int(xmax.text), int(ymin.text), int(ymax.text)))
    
    #print(boundBox)
    
    boundBox=np.reshape(boundBox,(int(boundBox.shape[0]/4),4))
    
    
    filename=tree.find("filename").text
    ext=filename[len(filename)-4:len(filename)]
    filename=filename[0:len(filename)-4]+"_AUG"+ext
    tree.find("filename").text=filename
    
    file = tree.find("path").text
    file=file[0:file.rfind("\\")+1]+filename
    tree.find("path").text=file
    
    #print("AAAA  " ,nameFile+"_AUG.xml")
    tree.write(dst+"_AUG.xml")
    
    #print(boundBox)
    return boundBox  # ritorna i valori letti di tutti i bb tranne handle


def addElementToXML(file, values, dst=""):
    
    #print(values.shape)
    #os.system("pause")
    filename=file[file.rfind("\\")+1:len(file)-4]
    
    if dst=="":
       #dst=filename
       dst=file[0:file.rfind("\\")]

    #l=open(dst+"_AUG.xml","r")
    l=open(file,"r")
    text=l.read()
    
    #print(text[text.rfind("</object>")+9:len(text)])
    #os.system("pause")
    
    pos=text[text.rfind("</object>")+9:len(text)]
    before=text[0:text.rfind("</object>")+9]
    
    
    name="UNKNOWN"
    #values=np.array([100,200,300,400],np.int)
    #print(text)
    l.close()
    #file=open(dst+"_AUG.xml","w")
    
    #print(dst+filename+".xml")
    #os.system("pause")
    file=open(dst+filename+".xml","w")
    file.write(before+"\n")
    for i in range(0, values.shape[0]):
        
        add="	<object>\n"+"		<name>"+name+"</name>\n"+"		<pose>Unspecified</pose>"+"\n		<truncated>0</truncated>\n"+"		<difficult>0</difficult>\n"+"		<bndbox>\n"+"			<xmin>"+str(values[i,0])+"</xmin>\n"+"			<ymin>"+str(values[i,1])+"</ymin>\n"+"			<xmax>"+str(values[i,2])+"</xmax>\n"+"			<ymax>"+str(values[i,3])+"</ymax>\n"+"		</bndbox>\n"+"	</object>"
        if (i<values.shape[0]-1):
          file.write(add+"\n")
        else:
          file.write(add)
        
    
    file.write(pos)
    file.close()



###############################################################################

#EXAMPLE USAGE:

'''
import time
date_hour_init=myUtils.dateTime()
start=time.time()
time.sleep(2)
date_hour_end=myUtils.dateTime()
end=time.time()
myUtils.timePassed(start, end, date_hour_init,date_hour_end)
'''

def dateTime():

    date_hour = datetime.now()
    date_hour=date_hour.strftime("%m/%d/%Y - %H:%M:%S")
    
    return date_hour

def timePassed(start,end, startTime, endTime):
    
    print("\n\n\nEXECUTION STARTED AT:  ", startTime, " -- FINISHED AT: ", endTime)
    
    if (end - start)>3600:
        print("EXECUTION TIME:  ", int((end-start)/3600),",",(end-start)%3600, " hours")
    
    elif (end - start)>60:
        print("\n\n\nEXECUTION TIME:  ", int((end-start)/60),",",(end-start)%60, " minutes")
    
    else:
        print("\n\n\nEXECUTION TIME:  ", int(end-start), " seconds")
        
###############################################################################



# camera opening
def openCam(maxNumOfCams=2,res='m'):

    for i in range(maxNumOfCams,-1,-1):
        cam = cv2.VideoCapture(i)
        #print("cam number ", i,cam.isOpened())
        if cam.isOpened()==True:
          cam_num=i
          break;
        else:
            cam.release()
        
    if cam.isOpened():
        #You can set the resolution-
        if res == "h":
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        elif res == "m":
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        elif res == "s":
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            print ("ERROR: Value Not Correct")
            exit()
    
    print("cam opened  ", cam_num) 
    
    return cam,cam_num


# FILE HANDLING
    

'''
    For the given path, get the List of all files in the directory tree 
'''
def getFolderList(dirPath, relativePath=0):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirPath)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirPath, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            #print(fullPath)
            allFiles.append(fullPath)
        
    
    allFiles=np.array(allFiles,np.str)
    allFiles=np.reshape(allFiles,(1,allFiles.shape[0]))
    
    
    if relativePath==1:
        for i in range(0,allFiles.shape[1]):
            allFiles[0,i]=allFiles[0,i][allFiles[0,i].rfind("\\")+1:len(allFiles[0,i])]
        #print(listOfFiles)
    
    return allFiles


# meerging txt files
import shutil
def merging_txt(path,dst_file,ext=".txt",verbose=True):

    vec=loadFile([ext],path)
    #print(len(vec))
    
    with open(dst_file,'wb') as wfd:
        for i in range(0,len(vec)):
            with open(vec[i],'rb') as fd:
                if verbose==True:
                    print("File: ",vec[i][vec[i].rfind("\\")+1:len(vec[i])])
                shutil.copyfileobj(fd, wfd)
            
            
    if verbose==True:
        print("\nmerging completed.")
        

'''
def orderFiles(vec,toFind='_',ext='.png'):
    
    fname=np.array([],np.int)
    path_init=vec[0][0:vec[0].rfind(toFind)+1]
    for i in range(0,len(vec)):
        
        fname=np.append(fname,int(vec[i][vec[i].rfind(toFind)+1:len(vec[i])-4]))
        
        
    
    fname=np.reshape(fname,(1,len(fname)))
    fname=np.sort(fname)
    
    print(fname.shape)
    
    #fname = fname.astype(str)
    fname_=np.array([],np.str)
    
    #print(fname.dtype)
    #ext=np.str(".jpg")
    #print(path_init.dtype)
    for i in range(0, fname.shape[1]):
        
        fname_=np.append(fname_,path_init+str(fname[0,i])+ext)
        #print(fname_[i])
    #fname_=np.reshape(fname_,(1,len(fname)))
    #print(fname_)
    #exit()
    return fname_
'''


def orderFiles(vec,toFind='_',ext='.png'):
    
    #fname=np.array([],np.int)
    fname=np.zeros((len(vec)),np.int)
    path_init=vec[0][0:vec[0].rfind(toFind)+1]
    #print(path_init)

    for i in range(0,len(vec)):
        
        #fname=np.append(fname,int(vec[i][vec[i].rfind(toFind)+1:len(vec[i])-4]))
        fname[i]=int(vec[i][vec[i].rfind(toFind)+1:len(vec[i])-4])
        
        
        
    
    #fname=np.reshape(fname,(1,len(fname)))
    fname=np.sort(fname)
    
    #print(fname.shape)
    #print(len(vec))
    
    
    
    #fname = fname.astype(str)
    #fname_=np.array([],np.str)
    #fname_=np.zeros((fname.shape[1]),np.str)
    
    #print(fname.dtype)
    #ext=np.str(".jpg")
    #print(path_init.dtype)
    for i in range(0, fname.shape[0]):
        
        #fname_=np.append(fname_,path_init+str(fname[0,i])+ext)
        vec[i]=path_init+str(fname[i])+ext
        #print(fname[i])
        #print(vec[i])
        #print(fname_[i])
    #fname_=np.reshape(fname_,(1,len(fname)))
    #print(fname_)
    #exit()
    return np.reshape(vec,(len(vec)))




def recordVideo(filename,fps,dim_x,dim_y,ext=".avi"):
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename+ext,fourcc, fps, (dim_x,dim_y))
    
    return out

def getContours(image, drawing=0, cv2Version=-1):
    
    if cv2Version==-1:
      cv2Version=getPythonVersion()

    if cv2Version==4:
        (contours,_) = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        (_,contours,_) = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours,_ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if drawing==1:
        img=createBlackOrWhiteImage(image.shape[0],image.shape[1],1,1)
        cv2.drawContours(img,contours,-1,255,1)
        return img
    
    else:
         return contours
     


# -------  BIGGER SPOT -------  #
def undesired_objects (image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1

    if (sizes.shape[0]>1):
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]
    
        img2 = np.zeros(output.shape)
        img2[output == max_label] = 255
        #cv2.imshow("Biggest component", np.uint8(img2))
        #cv2.waitKey()
    else:
        return None
        
        
    return np.uint8(img2)


def segmetation(bw):
    
    ret, labels = cv2.connectedComponents(bw)
    #print(ret,"  ",labels)
    #print(labels[0])



    bw=undesired_objects(bw)
    if bw is not None:
    
        #cv2.imshow('Mask Image', bw)
        #cv2.waitKey(0)
        
        return bw
    
    
# -------  BIGGER SPOT -------  #
        
    
    
    
# NN
   
import imgaug as ia
import imgaug.augmenters as iaa


ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
     
def augmenation(mode="individual", grayscale=False):

    if mode=="individual":
        seq_augmented= iaa.Sequential(
               [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.2), # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.Affine(
                scale={"x": (0.5, 1.2), "y": (0.5, 1.2),"fit_output":False}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 1),
                    [
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                       
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
        				
        				iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
        				
        				iaa.LinearContrast((0.2, 1.0), per_channel=0.5), # improve or worsen the contrast
        				iaa.Grayscale(alpha=(0.0, 1.0)),
        				sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
        				#sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
        				sometimes(iaa.PerspectiveTransform(scale=(0.05, 0.1)))
                    ],
                    random_order=True
                )
            ],
            random_order=True)
                
    elif mode=="strong":
            seq_augmented = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.2), # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45), # rotate by -45 to +45 degrees
                    shear=(-16, 16), # shear by -16 to +16 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((3, 3),
                    [
                        sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                        iaa.SimplexNoiseAlpha(iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0.5, 1.0)),
                            iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                        ])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                        # either change the brightness of the whole image (sometimes
                        # per channel) or change the brightness of subareas
                        iaa.OneOf([
                            iaa.Multiply((0.5, 1.5), per_channel=0.5),
                            iaa.FrequencyNoiseAlpha(
                                exponent=(-4, 0),
                                first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                second=iaa.LinearContrast((0.5, 2.0))
                            )
                        ]),
                        iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                        sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

    elif mode=="normal" and grayscale==False:
        seq_augmented = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.3))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        ),
        
        # ADDED LATER NO NORMAL MODE....
        iaa.SomeOf((0, 2),
                    [
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                       
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
        				
        				iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
        				
        				iaa.LinearContrast((0.2, 1.0), per_channel=0.5), # improve or worsen the contrast
        				iaa.Grayscale(alpha=(0.0, 1.0)),
        				sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
        				sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
        				sometimes(iaa.PerspectiveTransform(scale=(0.05, 0.1)))
                    ],
                    random_order=True
                )
    ], random_order=True) # apply augmenters in random order
        
    elif mode=="normal" and grayscale==True:
        
        # se lavoro con immagin sclaa di grigio non posos usare nessuno di questi augmentrs presenti in 
        # C:\Users\ValerioC\AppData\Local\Programs\Python\Python37\Lib\site-packages\imgaug\augmenters\color.py:
        
#        List of augmenters:
#
#        * InColorspace (deprecated)
#        * WithColorspace
#        * WithBrightnessChannels
#        * MultiplyAndAddToBrightness
#        * MultiplyBrightness
#        * AddToBrightness
#        * WithHueAndSaturation
#        * MultiplyHueAndSaturation
#        * MultiplyHue
#        * MultiplySaturation
#        * RemoveSaturation
#        * AddToHueAndSaturation
#        * AddToHue
#        * AddToSaturation
#        * ChangeColorspace
#        * Grayscale
#        * GrayscaleColorwise
#        * KMeansColorQuantization
#        * UniformColorQuantization
        
        #https://github.com/aleju/imgaug-doc/issues/16
        
        
        
        
        seq_augmented = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.3))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        ),
        
        # ADDED LATER NO NORMAL MODE....
        iaa.SomeOf((0, 2),
                    [
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                       
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
        				
        				#iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
        				
        				iaa.LinearContrast((0.2, 1.0), per_channel=0.5), # improve or worsen the contrast
        				#iaa.Grayscale(alpha=(0.0, 1.0)),
        				sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
        				sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
        				sometimes(iaa.PerspectiveTransform(scale=(0.05, 0.1)))
                    ],
                    random_order=True
                )
    ], random_order=True) # apply augmenters in random order
            

    return seq_augmented

    
def build_augmented_video_sequence(frames, window=16,mode="Individual", show=True):
        
    
    
            aug=augmenation(mode)
            
            augDet = aug.to_deterministic()            
            frames_shape=np.shape(frames)
            #images_aug=np.zeros((window,frames_shape[1],frames_shape[1],3),np.int)
            
            for i in range(frames_shape[0]):
                #results.append(augDet.augment_image(frames[i]))
                frames[i,:,:,:]=augDet.augment_image(frames[i])
                
            
            if show==True:
            
                G=np.zeros((112,112,3),np.int)
                for i in range(0,window):
                            
                            G=np.concatenate((G, frames[i]),axis=1)
                            if i==0:
                                G=G[:,112:,:]
                            
                            #cv2.imshow(str(i),np.uint8(G))
                            #print(G.shape)
                            
                            #cv2.imshow("",np.uint8(batch[i,:,:,:]))
                            #cv2.imshow("",images_aug[i])
                            #cv2.waitKey(0)

                return np.uint8(G)
            
            else:
                
                return frames
                
        
def build_augmented_frame(frame,mode="Individual", grayscale=False, show=True):
        
    
        
            aug=augmenation(mode,grayscale)
            
            augDet = aug.to_deterministic()            
            #frames_shape=1
            #images_aug=np.zeros((window,frames_shape[1],frames_shape[1],3),np.int)
            #print(frame.shape)

            #for i in range(frames_shape[0]):
                #results.append(augDet.augment_image(frames[i]))
                #frames[i,:,:,:]=augDet.augment_image(frames[i])
                
            
            frame=augDet.augment_image(frame)
            
            if show==True:
            
                #G=np.zeros((128,128,3),np.int)
                for i in range(0,1):
                            
                            #G=np.concatenate((G, frame),axis=1)
                            #if i==0:
                                #G=G[128:,:]
                            
                            #cv2.imshow(str(i),np.uint8(G))
                            #print(G.shape)
                            
                            #cv2.imshow("",np.uint8(batch[i,:,:,:]))
                            #cv2.imshow("",images_aug[i])
                            cv2.imshow("frame",frame)
                            cv2.waitKey(20)

                return frame
            
            else:
                
                return frame
                


# CONFUSION MATRIX
                
            
def threshold_pred(val_truth_pred_out, th=0.90, label_to_return=1,val_truth=-1):
    
    val_truth_pred_th=np.zeros((val_truth_pred_out.shape[0]),np.int)
    if label_to_return==-1:
        val_truth_excl=val_truth.copy()
        
        
    for i in range(0,val_truth_pred_out.shape[0]):
        
        if val_truth_pred_out[i,:][np.argmax(val_truth_pred_out[i,:])]<=th:
            val_truth_pred_th[i]=label_to_return
            if label_to_return==-1:
               val_truth_excl[0,i]=label_to_return
        
        else:
            val_truth_pred_th[i]=np.argmax(val_truth_pred_out[i,:])
    
    
    # se li passo label_to_return==-1, escludo al calcolo tutti i sample con score <= th
    # quindi rimuovo tutti i valori = a -1 dal return
    # se non ho label_to_return==-1, non ho valori da cancellare ovviamente
    # se considerlo label_to_return==-1 devo aggiornare anche label_truth
    
    
    if label_to_return==-1:
      val_truth_pred_th=val_truth_pred_th[val_truth_pred_th!=-1]
      val_truth_excl=val_truth_excl[val_truth_excl!=-1]
      return val_truth_pred_th, val_truth_excl
    else:
    
        return val_truth_pred_th
  

  
def thereshold_single(values,th, label_to_return=1):
    
    if values[np.argmax(values)]<th:        
      return 1
    else:
     return np.argmax(values)
        
            
              
              
def plot_confusion_matrix(cm, classes,
                              normalize=True,
                              title='normalized validation confusion matrix',
                              cmap=plt.cm.Blues):
       
        # This function prints and plots the confusion matrix.
        # Normalization can be applied by setting `normalize=True'
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print(" Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
            
        # replace eventually NAN values (nan is returned when a class is not present in array...)
        print(cm)
        where_are_NaNs = np.isnan(cm)
        if len(where_are_NaNs[where_are_NaNs==True])>0:
            cm[where_are_NaNs] = 0
            print(cm)
        
        
        acc=np.trace(cm)/np.sum(cm)
        #acc=int(np.round(acc,2)*100)
        acc=np.round(acc*100,2)
        print("accuracy  ",acc)
                
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title+": "+str(acc)+"%")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",fontsize=25,
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')  
        
    
    
    
    
    
def plot_confusion_matrix_merged(cm, classes,
                              normalize=True,
                              title='normalized validation confusion matrix',
                              cmap=plt.cm.Blues, fontsize=18):
       
        # This function prints and plots the confusion matrix.
        # Normalization can be applied by setting `normalize=True'
        
        #if normalize:
        cm=cm.copy()
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(" Normalized confusion matrix")
        #else:
        print('Confusion matrix, without normalization')
            
        # replace eventually NAN values (nan is returned when a class is not present in array...)
        
        where_are_NaNs = np.isnan(cm)
        if len(where_are_NaNs[where_are_NaNs==True])>0:
            cm[where_are_NaNs] = 0
            #print(cm)
        where_are_NaNs = np.isnan(cm_norm)
        if len(where_are_NaNs[where_are_NaNs==True])>0:
            cm_norm[where_are_NaNs] = 0
            #print(cm_norm)
    
        #print(cm)
        #print(cm_norm)
        acc=np.trace(cm)/np.sum(cm)        
        #print("accuracy  ",acc)
        #acc=int(np.round(acc,2)*100)
        acc=np.round(acc*100,2)
        print("accuracy  ",acc)
        
        #for i in range
        #rows=[]
        plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
        plt.title(title+"\nOverall Accuracy:  "+str(acc)+"%" + "  ["+ str(np.trace(cm))+"/" + str(np.sum(cm))+"]")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
    
        fmt_norm = 'd' 
        fmt= 'd'
        #thresh = cm.max() / 2.
        thresh = cm_norm.max()*0.75
        print("thresh Noemal",thresh)
        
        #count=0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            #count=count+1
            #print(count,j,i, cm[i,j])
            plt.text(j, i, format(int(np.round(cm_norm[i, j],2)*100), fmt_norm)+"%" + "\n" + "["+format(cm[i, j], fmt) + "-"+format(np.sum(cm[i,:]),fmt)+"]",
                     verticalalignment="center",horizontalalignment="center",fontsize=fontsize,
                     color="white" if np.round(cm_norm[i, j],2) > thresh else "black")
   
    

                    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')  
        
        return acc
        
        
        
#add a third line for survival cm analysis
def plot_confusion_matrix_merged_excl(cm, classes,labels_distribution,
                              normalize=True,
                              title='normalized validation confusion matrix',
                              cmap=plt.cm.Blues, fontsize=18):
       
        # This function prints and plots the confusion matrix.
        # Normalization can be applied by setting `normalize=True'
        
        #if normalize:
        cm=cm.copy()
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(" Normalized confusion matrix")
        #else:
        print('Confusion matrix, without normalization')
            
        # replace eventually NAN values (nan is returned when a class is not present in array...)
        
        where_are_NaNs = np.isnan(cm)
        if len(where_are_NaNs[where_are_NaNs==True])>0:
            cm[where_are_NaNs] = 0
            #print(cm)
        where_are_NaNs = np.isnan(cm_norm)
        if len(where_are_NaNs[where_are_NaNs==True])>0:
            cm_norm[where_are_NaNs] = 0
            #print(cm_norm)
    
        #print(cm)
        #print(cm_norm)
        acc=np.trace(cm)/np.sum(cm)        
        #print("accuracy  ",acc)
        #acc=int(np.round(acc,2)*100)
        acc=np.round(acc*100,2)
        print("accuracy  ",acc)
        
        #for i in range
        #rows=[]
        plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
        plt.title(title+"Overall Accuracy:  "+str(acc)+"%" + "  ["+ str(np.trace(cm))+"/" + str(np.sum(cm))+"]")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
    
        fmt_norm = 'd' 
        fmt= 'd'
        thresh = cm_norm.max()*0.75
        print("trhesh ",thresh)
        values=np.zeros((cm.shape[0]),np.int)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            values[i]=values[i]+cm[i,j]

        print(values)
        
        #count=0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            #count=count+1
            #print(count,j,i, cm[i,j])
            plt.text(j, i, format(int(np.round(cm_norm[i, j],2)*100), fmt_norm)+"%" + "\n" + "["+format(cm[i, j], fmt) + "-"+format(np.sum(cm[i,:]),fmt)+"]" +
                     "\n" + "("+format(int(np.round((values[i]/labels_distribution[i])*100,2)), fmt) +"%)",
                     verticalalignment="center",horizontalalignment="center",fontsize=fontsize,
                     color="white" if cm[i, j]/(values[i]) > thresh else "black")
   
    

                    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')  
        
        return acc
        
        
        
def barPolt(acc_vec, th, dst):
    
    plt.close("all")
    plt.figure()

    
    plt.style.use(['seaborn-darkgrid'])
    
    
    #plt.bar(th,acc_vec,align='center', alpha=0.5, color=['b','r','g','y'])
    plt.bar(th,acc_vec,align='center', alpha=0.5, color='b')
    plt.ylim([int(np.min(acc_vec)-np.min(acc_vec)*0.2),int(np.max(acc_vec)+np.max(acc_vec)*0.1)])
    
    if "0.00" in th:
        #plt.xticks(th,['NO-Threshold', 'Threshold-90%', 'Threshold-95%', 'Threshold-99%'], rotation=90, color='navy')
        plt.xticks(th,['NO Threshold', 'Threshold '+th[1], 'Threshold '+th[2], 'Threshold '+th[3], 'Threshold '+th[4], 'Threshold '+th[5]], rotation=90, color='navy')
    else:
        plt.xticks(th,['Threshold '+th[0], 'Threshold '+th[1], 'Threshold '+th[2], 'Threshold '+th[3], 'Threshold '+th[4]], rotation=90, color='navy')
    plt.ylabel("Accuracy", color='navy')
    plt.title("Accuracy on Thresholding Network Predictions", color="darkblue")
    # Adjust the margins
    #plt.subplots_adjust(bottom= 0.25, top = 0.98)
    plt.subplots_adjust(bottom= 0.25)
    
    labels=np.array([],np.str)
    for i in range(0,len(th)):
        # Create labels
        #label = [str(acc_vec[0])+"%", str(acc_vec[1])+"%", str(acc_vec[2])+"%", str(acc_vec[3])+"%"]
        labels = np.append(labels,str(acc_vec[i])+"%")
    
    # Text on the top of each barplot
    for i in range(len(th)):
        plt.text(x = th[i] , y = acc_vec[i]+0.1, s = labels[i], size = 12, style='italic',color='black')
    
    plt.savefig(dst+"BarPlot.png",bbox_inches="tight")
    #plt.show()
    
def barPolt_pred(acc_vec, th, dst):
    
    plt.close("all")
    plt.figure()

    
    plt.style.use(['seaborn-darkgrid'])
    
    
    #plt.bar(th,acc_vec,align='center', alpha=0.5, color=['b','r','g','y'])
    plt.bar(th,acc_vec,align='center', alpha=0.5, color='r')
    plt.ylim([int(np.min(acc_vec)-np.min(acc_vec)*0.2),int(np.max(acc_vec)+np.max(acc_vec)*0.1)])
    
    if "0.00" in th:
        #plt.xticks(th,['NO-Threshold', 'Threshold-90%', 'Threshold-95%', 'Threshold-99%'], rotation=90, color='navy')
        plt.xticks(th,['NO Threshold', 'Threshold '+th[1], 'Threshold '+th[2], 'Threshold '+th[3], 'Threshold '+th[4], 'Threshold '+th[5]], rotation=90, color='firebrick')
    else:
        plt.xticks(th,['Threshold '+th[0], 'Threshold '+th[1], 'Threshold '+th[2], 'Threshold '+th[3], 'Threshold '+th[4]], rotation=90, color='firebrick')
    plt.ylabel("Thresholded predictions", color='firebrick')
    plt.title("Thresholded predicitions on total testing samples", color="firebrick")
    # Adjust the margins
    #plt.subplots_adjust(bottom= 0.25, top = 0.98)
    plt.subplots_adjust(bottom= 0.25)
    
    labels=np.array([],np.str)
    for i in range(0,len(th)):
        # Create labels
        #label = [str(acc_vec[0])+"%", str(acc_vec[1])+"%", str(acc_vec[2])+"%", str(acc_vec[3])+"%"]
        labels = np.append(labels,str(acc_vec[i])+"%")
    
    # Text on the top of each barplot
    for i in range(len(th)):
        plt.text(x = th[i], y = acc_vec[i]+0.1, s = labels[i], size = 12, style='italic',color='black')
    
    plt.savefig(dst+"BarPlot_predictions.png",bbox_inches="tight")
    #plt.show()
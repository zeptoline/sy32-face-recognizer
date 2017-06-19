# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:27:29 2017

@author: syresp008
"""

from skimage import io
from skimage import util
from skimage import color
from skimage import feature
from skimage import transform
from sklearn import svm
import numpy as np
import random


#Resolution des carres pour les visages
res = 128
      
def loadimg(size, path):
    tab = np.zeros((24,24,size))
    for i in range(1, size):
        im = io.imread(path + "%05d"%(i) + ".png")
        tab[:,:,i-1] = util.img_as_float(im)
    return tab

def CrossValidation(xtrain, ytrain, Nvc):
    clf = svm.SVC(kernel='linear')
    r = np.zeros(Nvc)
    for i in range(Nvc):
        print("boucle : %d"%(i))
        mask = np.zeros(xtrain.shape[0], dtype = bool)
        mask[np.arange(i, mask.size, Nvc)] = True
        clf.fit(xtrain[~mask], ytrain[~mask])
        r[i] = np.mean(clf.predict(xtrain[mask]) != ytrain[mask])
    print("Taux d'erreur : %d"%(np.mean(r)))
    return clf
def tabReshape(tab, size):
    tabresh = np.zeros((size,576))
    for i in range(0,size):
        tabresh[i, :] = tab[:,:,i].reshape(576)
    return tabresh

  
label = np.loadtxt ( 'D:\\Data\\Work\\sy32\\projet\\label.txt' , delimiter=' ', dtype=int)
visagepos= np.zeros((1000, res, res))
visageneg= np.zeros((1000, res, res))    

for i in range(1,1000):
    im = util.img_as_float(color.rgb2grey(io.imread("D:\\Data\\Work\\sy32\\projet\\train\\"+ "%04d"%(i) + ".jpg")))
    cl = label[i-1]
    
    #prendre le carré qui envelope le rectangle
    x = cl[1]
    y = cl[2]
    le = cl[3]
    ha = cl[4]
    x2 = x + le
    y2 = y + ha
    if(ha > le):
        x =(int)  (x - (ha - le)/2)
        x2 = x + le + (ha - le)
        y2 = y + ha
    elif(le > ha):
        y = (int) (y - (le - ha)/2)
        x2 = x + le
        y2 = y + ha + (le - ha)
    
    #Si on a un x ou un y inferieur à zero, on les remets à zero. Il y aura un léger étirage de l'image, qui ne devrait pas être génant
    if(x<0):
        x=0
    if(y<0):
        y=0
        
    visage = im[y:y2, x:x2]
    visage = transform.resize(visage, (res,res))
    visagepos[i-1, :, :] = visage 
    
    while True:
        xrand = random.randint(0,np.shape(im)[1]-res)
        yrand = random.randint(0,np.shape(im)[0]-res)
        
        long_intersection = x2 - xrand
        haut_intersection = y2 - yrand
        if(long_intersection > 0 and haut_intersection > 0):
            aire_intersection = (float) (long_intersection * haut_intersection)
            aire_rand = res**2
            aire_visage = cl[3] * cl[4]
            aire_union = (float) (aire_rand + aire_visage - aire_intersection)
            taux = (aire_intersection / aire_union)
            if(taux > 0.5):
                break
            
    negvisage= im[yrand:yrand+res, xrand:xrand+res]
    visageneg[i-1, :, :] = negvisage
        
        



io.imshow(visage)
io.imshow(negvisage)
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
res = 32
path = "D:\\Data\\Work\\sy32\\projet\\"
      
def loadimg(size, pathimg):
    tab = np.zeros((24,24,size))
    for i in range(1, size):
        im = io.imread(pathimg + "%05d"%(i) + ".png")
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
def Validation(xtrain, ytrain):
    clf = svm.SVC(kernel='linear')
    clf.fit(xtrain, ytrain)
    return clf
def tabReshape(tab, size):
    tabresh = np.zeros((size,res**2))
    for i in range(0,size):
        tabresh[i, :] = tab[i,:,:].reshape(res**2)
    return tabresh

def exempleneg(im):
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
            if(aire_union == 0):
                aire_union = 1
            taux = (aire_intersection / aire_union)
            if(taux > 0.5):
                break
    return im[yrand:yrand+res, xrand:xrand+res]


def fenetreglissante(im):
    listxy = []
    ensemble = []
    scale = [0.4, 0.3, 0.2, 0.1]
#    scale = [0.4]
    pas = round(res/4)
    for k in scale:
        imscaled = transform.rescale(im, k, mode='reflect')
        longueur = np.shape(imscaled)[1]
        hauteur = np.shape(imscaled)[0]
        for i in range(0, hauteur-res,pas):
            for j in range(0, longueur-res, pas):
                if(i+res < longueur and j+res < hauteur):
                    carre = imscaled[i:i+res, j:j+res]
                    #carrepredict = clf.predict(carre.reshape(res**2))
                    #if(carrepredict[0] == 1):
                    listxy.append([x, y, res] * (1/k))
                    ensemble.append(carre)
            
        
    return ensemble

def aireCarre():
    #donne vrai ou faux si aires > 0.5
    xrand = random.randint(0,np.shape(im)[1]-res)
    yrand = random.randint(0,np.shape(im)[0]-res)
    
    long_intersection = x2 - xrand
    haut_intersection = y2 - yrand
    if(long_intersection > 0 and haut_intersection > 0):
        aire_intersection = (float) (long_intersection * haut_intersection)
        aire_rand = res**2
        aire_visage = cl[3] * cl[4]
        aire_union = (float) (aire_rand + aire_visage - aire_intersection)
        if(aire_union == 0):
            aire_union = 1
        taux = (aire_intersection / aire_union)
        if(taux > 0.5):
            break
    
    
def fusion():
    # Fusionner les carres qui sont proches en taille et en position (reprendre les calculs d'aires)
    
def calculScore():
    

def predictface(clf, tab):
    return clf.decision_function(np.reshape(tab, (-1, res**2)))


  
label = np.loadtxt ( path + 'label.txt' , delimiter=' ', dtype=int)
visagepos= np.zeros((1000, res, res))
visageneg= np.zeros((3000, res, res))    

for i in range(1,1001):
    im = util.img_as_float(color.rgb2grey(io.imread(path + "train\\"+ "%04d"%(i) + ".jpg")))
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
    visage = transform.resize(visage, (res,res), mode='reflect')
    visagepos[i-1, :, :] = visage 
    
    j= (i-1)*3
    negvisage = exempleneg(im)
    visageneg[j, :, :] = negvisage
    
    negvisage = exempleneg(im)
    visageneg[j+1, :, :] = negvisage
    
    negvisage = exempleneg(im)
    visageneg[j+2, :, :] = negvisage
 


xpos = tabReshape(visagepos, 1000)
ypos = np.ones((1000))
xneg = tabReshape(visageneg, 3000)
yneg = np.zeros((3000))

xtrain = np.concatenate((xpos, xneg), 0)
ytrain = np.concatenate((ypos, yneg))

clf = Validation(xtrain, ytrain)
#clf = CrossValidation(xtrain, ytrain, 5)

im = util.img_as_float(color.rgb2grey(io.imread(path + "train\\"+ "%04d"%(1) + ".jpg")))
imgtest = fenetreglissante(im)
imgvisage = predictface(clf, imgtest)
maxn = max(imgvisage)
i = np.argmax(imgvisage)
imgvisage = np.delete(imgvisage, i)
io.imshow(imgtest[i])

print(np.shape(imgvisage[0]))
# -*- coding: utf-8 -*-
"""
Created on Oct 26 2020

@author: Jhonatan, Juanpipí
"""

import os
import sys
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as sio


path_base = os.path.dirname(os.path.abspath(__file__))
sys.path.append('./Feature_extraction/')


import Feature_extraction.emotions as em

################################################################################
################################################################################
################################################################################


path_files = path_base+'/Data'
list_wavs = os.listdir(path_files)
feat_mat = []
fileName = 's17.mat'

db = sio.loadmat(path_files+'/'+fileName)
print('='*50)
print(path_files+'/'+fileName)

X = db['data']
fs = 128

for sigs in X:
    #######################################################################        
    #################Read the speech signal################################
    """
    fs: Sampling frequency. MAKE SURE ALL OF YOUR FILES HAVE THE SAME SAMPLING FREQUENCY
    sig: Speech signal->Time series with the sequences of amplitude values
    """
    #fs,sig = read(path_files+'/'+idx_wav)

    

    #######################################################################
    ###################BASIC PREPROCESSING#################################
    sigs = sigs-np.mean(sigs, axis=1)[:, np.newaxis] #Remove any possible DC level
    sigs = sigs/np.max(np.abs(sigs), axis=1)[:, np.newaxis] #Re-scale the signal between -1 and 1
    #######################################################################
    #######################################################################
    
    #Emotions
#    features = em.extract_features(sig,fs,f0,0.02,0.01)
    features = em.extract_features(sigs, fs)
    
    #Append the static feature vector to a list
    feat_mat.append(features)

#Convert the list into a numpy feature matrix
feat_mat = np.vstack(feat_mat)

#Save feature matrix as a TEXT file
np.savetxt(path_base+'/Features/articulation_features_test_16000'+group+'.txt',feat_mat)
print('!'*50)
print('!'*50)
print('Articulation feature matrix saved in',path_base+'/Features/articulation_features_test_16000'+group+'.txt')
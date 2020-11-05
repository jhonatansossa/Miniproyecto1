# -*- coding: utf-8 -*-
"""
Created on Oct 26 2020

@author: Jhonatan Sossa, Juan Pablo Quinchía

"""
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

import Feature_extraction.emotions_functions as emF

def extract_features(sigs, fs):
    avg_sr, avg_d, avg_nd = GSR(sigs[36], fs)
    BVP(sigs[38], fs)
    Respiration(sigs[37],fs)
    Temperature(sigs[39],fs)
    EMGEOG(sigs[32],sigs[33],sigs[34],sigs[35],fs)
    EEG_1(sigs[0],sigs[16],fs)
    
    
    return np.hstack([avg_sr, avg_d, avg_nd])

#************************************************
def EEG_1(feat1,feat2,fs):
    sig = feat1-feat2
    order=2
    power =[]
    for x in range(3):
        if x == 0:
            lowcut = 3.5
            highcut = 7.5
        if x == 1:
            lowcut = 8
            highcut = 13
        if x == 2:
            lowcut = 12
            highcut = 30
        if x == 3:
            lowcut = 25
            highcut = 100
            
        sig_filt = emF.butter_passband_filter(sig, lowcut, highcut, fs, order)
        f, Pxx = signal.welch(sig_filt, fs, nperseg=64)
        AbsPxx = np.sum(Pxx)
        LogPxx = 20*np.log(AbsPxx)
        power.append(AbsPxx, LogPxx)
    
    #alpha 8-13Hz
    #beta 12-30Hz
    #theta 3.5-7.5Hz
    #gamma 25-100Hz
    
    
    return power
    
    
    
    

#************************************************
def EMGEOG(feat1, feat2, feat3, feat4, fs):
    
    feat_one = emF.mu_ener_var(feat1)
    feat_two = emF.mu_ener_var(feat2)
    feat_three = emF.mu_ener_var(feat3)
    feat_four = emF.mu_ener_var(feat4)
    
    return np.hstack([feat_one, feat_two, feat_three, feat_four])

#************************************************
def Temperature(feat,fs):
    
    feat_mu = np.mean(feat)
    feat_diff = np.mean(np.diff(feat))
    
    order=2
        
    cutoff = 0.1
    lowcut = 0.1
    highcut = 0.2
    
    size_frameS=0.02*float(fs)
    size_stepS=0.01*float(fs)
    overlap=size_stepS/size_frameS
    nF=int((len(feat)/size_frameS/overlap))-1
    power_first = []
    power_second = []
    
    first_feat = emF.butter_lowpass_filter(feat, cutoff, fs, order)
    second_feat = emF.butter_passband_filter(feat, lowcut, highcut, fs, order)
    
    for l in range(nF):
        feat_frame = first_feat[int(l*size_stepS):int(l*size_stepS+size_frameS)]
        power_f = emF.logEnergy(feat_frame)
        power_f = np.power(10,power_f/10)
        power_first.append(power_f)
        
        feat_frame = second_feat[int(l*size_stepS):int(l*size_stepS+size_frameS)]
        power_s = emF.logEnergy(feat_frame)
        power_s = np.power(10,power_s/10)
        power_second.append(power_s)
        
        
    power_first_mu = np.mean(power_first)
    power_second_mu = np.mean(power_second)
    
    return np.hstack([feat_mu, feat_diff, power_first_mu, power_second_mu])

        #frequency = np.linspace(0, 128/2, len(power_spectrum_first))
     
#************************************************
def Respiration(feat,fs):
    
    order=2
        
    #first passband filter
    lowcut = 0.05
    highcut = 0.25
        
    first_feat = emF.butter_passband_filter(feat, lowcut, highcut, fs, order)
        
    #second passband filter
    lowcut = 0.25
    highcut = 0.5
        
    second_feat = emF.butter_passband_filter(feat, lowcut, highcut, fs, order)
    
    energy_feat_first = emF.logEnergy(first_feat)
    energy_feat_second = emF.logEnergy(second_feat)
    
    energy_rate = energy_feat_first-energy_feat_second
    
    feat_mu = np.mean(feat)
    
    feat_diff_mu = np.mean(np.diff(feat))
    
    feat_max = np.max(feat)
    
    return np.hstack([energy_rate, feat_mu, feat_diff_mu, feat_max])

#************************************************
def BVP(feat, fs):
    
    size_frameS=10*float(fs)
    size_stepS=5*float(fs)
    overlap=size_stepS/size_frameS
    nF=int((len(feat)/size_frameS/overlap))-1

    hrs=[]
    hrvs=[]
    rrs_f=[]

    for l in range(nF):
        feat_frame = feat[int(l*size_stepS):int(l*size_stepS+size_frameS)]
        
        feat_frame = 10*np.diff(feat_frame,1)
        
        peaks, _ = signal.find_peaks(feat_frame, prominence = 0.03)
        peaks_copy = peaks
        
        feat_copy = feat_frame[peaks]
        
        peaks1, _ = signal.find_peaks(np.concatenate(([0],feat_frame[peaks],[0])), prominence = 0.03)
        
        feat_copy1 = feat_copy[peaks1-1]
        
        peaks2 = []
        for i in range(len(feat_copy1)):
            idx= np.where(feat_copy1[i]==feat_copy)[0][0]
            peaks2.append(peaks_copy[idx])
            
            feat_copy = np.delete(feat_copy, idx)
            peaks_copy = np.delete(peaks_copy, idx)
            
        peaks2= np.hstack(peaks2)
        
        # plt.plot(feat)
        # plt.plot(peaks, feat[peaks])
        # plt.plot(peaks, feat[peaks], 'x')
        # plt.plot(peaks2, feat[peaks2], 'o')
        # plt.plot(np.zeros_like(feat), "--", color="gray")
        # plt.show()
        
       
        
        rrs = []
        rmssd = []
        cont = 0
        for i in range(len(peaks2)-2):
            one = peaks2[i]
            two = peaks2[i+1]
            interval = (two - one)/fs
            rrs.append(interval)
        rrs_copy = rrs.copy()
        rrs.reverse()
        rmssd = np.diff(rrs)
        rmssd = np.power(rmssd,2)
        rmssd = rmssd[::-1]
        
            # cont = cont + 1
            # if i != 0:
            #     pw = np.power((rrs[i-1]-rrs[i]),2)
            #     rmssd.append(pw)
            #     cont = 1
        
        rmssd = np.power(np.sum(rmssd),1/2)
        hr = len(peaks2)/(size_frameS/fs)
        
        hrs.append(hr)
        hrvs.append(rmssd)
        rrs_f.append(rrs_copy)
        
   
    hrs = static_feats(hrs)
    hrvs = static_feats(hrvs)
    rrs_f = static_feats(np.hstack(rrs_f))    
    
    
     #Filter parameters
    # order=2
        
    #     #first passband filter
    # lowcut = 0.04
    # highcut = 0.15
        
    # first_feat = emF.butter_passband_filter(feat_frame, lowcut, highcut, fs, order)
        
    # #second passband filter
    # lowcut = 0.15
    # highcut = 0.5
        
    # second_feat = emF.butter_passband_filter(feat_frame, lowcut, highcut, fs, order)
    
    # f, Pxx = signal.welch(feat, fs, nperseg=64)
    # plt.semilogy(f, Pxx)
    # plt.show()
   
    # # Plot output and frequency response
    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(feat)
    # plt.title("non filtered signal")
    
    # plt.subplot(3, 1, 2)
    # plt.plot(first_feat)
    # plt.title("0.04 and 0.15")

    # plt.subplot(3, 1, 3)
    # plt.plot(second_feat)
    # plt.title("0.15 and 0.5")
    
    # plt.show

    
    return np.hstack([hrs, hrvs, rrs_f])

#************************************************
def GSR(feat, fs):
    
    dfeat = np.diff(feat, 1, axis=0)
    
    #Average skin resistance, average of derivative, average of derivative for negative values only
    feat_mu = np.mean(feat,0)
    dfeat_mu = np.mean(dfeat,0)
    ndfeat_mu = np.mean(np.where(dfeat < 0)[0],0)
    
    #Filter parameters
    cutoff = 2.4
    order=2
    
    #filter
    feat = emF.butter_lowpass_filter(feat, cutoff, fs, order)
    
    #spectral power
    fourier_transform = np.fft.rfft(feat)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, 128/2, len(power_spectrum))
    #    plt.plot(frequency, abs_fourier_transform)
    
    #select the first ten spectral power
    peaks, _ = signal.find_peaks(power_spectrum, height=0)
    
#    plt.figure()
#    plt.plot(power_spectrum)
#    plt.plot(peaks, power_spectrum[peaks], "x")
#    plt.plot(np.zeros_like(power_spectrum), "--", color="gray")
#    plt.show()
    
    return np.hstack([feat_mu, dfeat_mu, ndfeat_mu])
    
#************************************************

#def segment_features(segment,fs,win,step):
#    """
#    Segment: onset or offset transitions list
#    """
#    if len(segment)==0:
#        print('!'*50)
#        print('!'*50)
#        print('Transitions were not detected')
#        print('!'*50)
#        print('!'*50)
#    feats = []
#    for idx_seg in segment:
#        frames = extract_windows(idx_seg,int(win*fs),int(step*fs))
#        for frame in frames:
#            #Mfcc
#            fmfcc = artF.mfcc(frame,fs)
#            #Bark
#            fbark = artF.barke(frame,fs)
#            feats.append(np.hstack([fbark,fmfcc]))
#    feats = np.vstack(feats)
#    dfeats = np.diff(feats,1,axis=0)
#    ddfeats = np.diff(feats,2,axis=0)
#    
#    #Compute statistics
#    feats = static_feats(feats)
#    dfeats = static_feats(dfeats)
#    ddfeats = static_feats(ddfeats)
#    return np.hstack([feats,dfeats,ddfeats])

#*****************************************************************************
#def extract_windows(signal, size, step):
#    # make sure we have a mono signal
#    assert(signal.ndim == 1)
#    
##    # subtract DC (also converting to floating point)
##    signal = signal - signal.mean()
#    
#    n_frames = int((len(signal) - size) / step)
#    
#    # extract frames
#    windows = [signal[i * step : i * step + size] 
#               for i in range(n_frames)]
#    
#    # stack (each row is a window)
#    return np.vstack(windows)
##**************************************************************************
def static_feats(featmat):
    """Compute static features
    :param featmat: Feature matrix
    :returns: statmat: Static feature matrix
    """
    mu = np.mean(featmat,0)
    st = np.std(featmat,0)
    statmat = np.hstack([mu,st])    
    return statmat
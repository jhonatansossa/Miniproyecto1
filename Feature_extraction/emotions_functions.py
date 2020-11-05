import numpy as np
from scipy import signal


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def butter_passband_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5*fs
    normal_lowcut = lowcut / nyq
    normal_highcut = highcut / nyq
    b, a = signal.butter(order, [normal_lowcut, normal_highcut], btype='band', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def logEnergy(sig):
    logE=-1e30
    if len(sig)>0:
        sig2=np.power(sig,2)
        sumsig2 = np.sum(np.absolute(sig2))/len(sig2)
        logE=10*np.log10(sumsig2)
    return logE

def mu_ener_var(sig):
    feat_mu = np.mean(sig)
    feat = np.power(sig,2)
    feat_energy = np.sum(np.absolute(feat))
    feat_variance = np.var(sig)
    return np.hstack([feat_mu, feat_energy, feat_variance])


   
    
      
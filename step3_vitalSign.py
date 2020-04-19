from scipy.io import loadmat
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft

fileRef = {'time': '/t_stmp.mat', 'bin': 'range_bins.mat', 'mag': 'rawscans.mat'}
radarRef = {'time': 't_stmp', 'bin': 'range_bins', 'mag': 'rawscans'}

pre_path = './DataSet/Vital Sign/'
for participant in ['BR_st1', 'BR_st2']:
    for pattern in ['browsing', 'fetal_left', 'fetal_right', 'freefall', 'left_turned', 'right_turned', 'soldier']:
        radarData = {}
        for files in fileRef:
            radarData[files] = loadmat(pre_path + participant + '/Radar 1/' + pattern + '/' + fileRef[files])[radarRef[files]]
            if files == 'time' or files == 'bin':
                radarData[files] = radarData[files].flatten()
        
        binIdx = np.where(np.logical_and(radarData['bin'] > 0.8128, radarData['bin'] < 1.8161))
        signal = []
        iterations = np.shape(radarData['mag'])[0]
        for i in range(iterations):
            maxVal = np.max(radarData['mag'][i, :][binIdx])
            signal.append(maxVal)
        
        timeElapsed = radarData['time'][-1] - radarData['time'][0]
        sampleRate = timeElapsed/iterations

        signal = np.asarray(signal)
        signal = np.subtract(signal, np.mean(signal))

        signalFFT = fft(signal)
        freqs = np.multiply(np.linspace(0.0, 1.0/(2.0*sampleRate), iterations//2), 60)
        signalFFT = 2.0/iterations * np.abs(signalFFT[0:iterations//2])
        idx = np.where(freqs < 90)[0][-1]
        
        plt.figure()
        plt.subplot(211)
        plt.plot(np.linspace(0, timeElapsed,iterations), signal)
        plt.title('Time Domain - Breathing Signal')
        plt.xlabel('Time Elapsed (in Seconds)')
        plt.ylabel('Magnitude')

        plt.subplot(212)
        plt.plot(freqs[:idx], signalFFT[:idx])
        plt.title('Freq Domain - FFT Breathing Signal')
        plt.xlabel('Frequency (breaths per min)')
        plt.ylabel('Magnitude')
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle('Participant: ' + participant + '  Activity: ' + pattern)
        plt.savefig('./step3_vital_sign/Vital_Sign_Radar1 ' + participant + '_' + pattern + '.png', bbox_inches="tight")

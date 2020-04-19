from scipy.io import loadmat
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft

fileRef = {'time': '/t_stmp.mat', 'bin': 'range_bins.mat', 'mag': 'rawscans.mat'}
radarRef = {'time': 't_stmp', 'bin': 'range_bins', 'mag': 'rawscans'}
signalTrimRef = {
                'BR_st1': {'browsing': [20, 40], 'fetal_left': [15, 60], 'fetal_right': [15, 60], 'freefall': [0, 55], 'left_turned': [20, 60], 'right_turned': [10, 60]}, 
                'BR_st2': {'browsing': [20, 29], 'fetal_left': [15, 60], 'fetal_right': [10, 40], 'freefall': [37, 55], 'left_turned': [33, 50], 'right_turned': [10, 35], 'soldier': [21, 60]} 
                }

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
        trimStart = 0
        trimEnd = timeElapsed
        if pattern in signalTrimRef[participant]:
            trimStart = signalTrimRef[participant][pattern][0]
            trimEnd = signalTrimRef[participant][pattern][1]
            iterStart = trimStart*iterations//timeElapsed
            iterEnd = trimEnd*iterations//timeElapsed
            signal = signal[iterStart: iterEnd + 1]
            iterations = len(signal)
        
        signal = np.subtract(signal, np.mean(signal))

        signalFFT = fft(np.multiply(signal, np.hamming(iterations)))
        freqs = np.multiply(np.linspace(0.0, 1.0/(2.0*sampleRate), iterations//2), 60)
        signalFFT = 2.0/iterations * np.abs(signalFFT[0:iterations//2])
        idx = np.where(freqs < 90)[0][-1]
        
        plt.figure()
        plt.subplot(211)
        plt.plot(np.linspace(trimStart, trimEnd,iterations), signal)
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
    
from scipy.io import loadmat
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Trilateration: returns point of intersection
def triLat(r1, r2, r3):
    x1, y1 = 0, 0 # R 103
    x2, y2 = 0, 2.032 # R 108
    x3, y3 = 1.016, 0 # R 109

    A = 2*x2 - 2*x1
    B = 2*y2 - 2*y1
    C = r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2
    D = 2*x3 - 2*x2
    E = 2*y3 - 2*y2
    F = r2**2 - r3**2 - x2**2 + x3**2 - y2**2 + y3**2
    x = (C*E - F*B) / (E*A - B*D)
    y = (C*D - A*F) / (B*D - A*E)

    return x, y

fileRef = {'time': '/T_stmp.mat', 'bin': 'range_bins.mat', 'mag': 'envNoClutterscans.mat'}
radarRef = {'103': {'time': 'T_stmp_1033', 'bin': 'Rbin_1033', 'mag': 'envNoClutterscansV_1033'}, 
            '108': {'time': 'T_stmp_103', 'bin': 'Rbin_103', 'mag': 'envNoClutterscansV_103'}, 
            '109': {'time': 'T_stmp_102', 'bin': 'Rbin_102', 'mag': 'envNoClutterscansV_102'}
            }
groundTruthMap = {'diag': {'x': np.asarray([0.762, 1.98]), 'y': np.asarray([0.9652, 3.4])},
                    'U': {'x': np.asarray([0.762, 0.762, 1.98, 1.98]), 'y': np.asarray([3.4, 0.9652, 0.9652, 3.4])},
                    'L': {'x': np.asarray([0.762, 1.98, 1.98]), 'y': np.asarray([3.4, 3.4, 0.9652])},
                    'gamma': {'x': np.asarray([0.762, 1.98, 1.98]), 'y': np.asarray([0.9652, 0.9652, 3.4])},
                    'four': {'x': np.asarray([0.762, 0.762, 1.98, 1.98]), 'y': np.asarray([3.4, 2.1844, 2.1844, 0.9652])}
                }

pre_path = './DataSet/Localization/'
for participant in ['participant1', 'participant2']:
    for pattern in ['diag', 'four', 'gamma', 'L', 'U']:
        print('Running Pattern: ' + pattern)
        radarData = {'103': {}, '108': {}, '109': {}}
        triLatRes = np.asarray([0, 0])
        for radar in ['103', '108', '109']:
            for files in fileRef:
                # load .mat files
                radarData[radar][files] = loadmat(pre_path + participant + '/' + radar + '/' + pattern + '/' + fileRef[files])[radarRef[radar][files]]
                if files == 'time' or files == 'bin':
                    radarData[radar][files] = radarData[radar][files].flatten()

        # time synchronization
        startStamp = max([radarData['103']['time'][0], radarData['108']['time'][0], radarData['109']['time'][0]])
        endStamp = min([radarData['103']['time'][-1], radarData['108']['time'][-1], radarData['109']['time'][-1]])

        timeSyncData = {}
        for radar in ['103', '108', '109']:
            idx1 = np.where(radarData[radar]['time'] > startStamp)[0][0]
            idx2 = np.where(radarData[radar]['time'] < endStamp)[0][-1]

            radarData[radar]['time'] = radarData[radar]['time'][idx1:idx2+1]
            radarData[radar]['mag'] = radarData[radar]['mag'][idx1:idx2+1, :]

            # resampling to 48Hz
            series = pd.DataFrame(data = radarData[radar]['mag'], index = radarData[radar]['time'])
            series.index = pd.to_datetime(series.index, unit='s')
            series = series.resample('20.8L').mean().bfill(axis ='rows')

            timeSyncData[radar] = series.to_numpy()

        # based on the sampling frequency
        window_size = 12
        window_shift = 6
        threshold = 50000
        minIterations = min([np.shape(timeSyncData['103'])[0], np.shape(timeSyncData['108'])[0], np.shape(timeSyncData['109'])[0]])
        iterations = math.floor((minIterations - window_size)/window_shift) + 1
        # windowing
        for i in range(iterations):
            radar1 = timeSyncData['103'][i*window_shift:(i*window_shift)+window_size, :]
            radar2 = timeSyncData['108'][i*window_shift:(i*window_shift)+window_size, :]
            radar3 = timeSyncData['109'][i*window_shift:(i*window_shift)+window_size, :]
            points = []
            for j in range(window_size):
                binIndices1 = np.where(radar1[j, :] > threshold)[0]
                binIndices2 = np.where(radar2[j, :] > threshold)[0]
                binIndices3 = np.where(radar3[j, :] > threshold)[0]

                if len(binIndices1) <= 0 or len(binIndices2) <= 0 or len(binIndices3) <= 0: 
                    continue
                else:
                    # closest range_bin values corresponding to the magnitude above the threshold
                    r1 = radarData['103']['bin'][binIndices1[0]]
                    r2 = radarData['103']['bin'][binIndices2[0]]
                    r3 = radarData['103']['bin'][binIndices3[0]]

                # trilateration
                intPoint = triLat(r1, r2, r3)
                # omitting circles that don't intersect
                if intPoint[0] > 0 and intPoint[0] < 4.04 and intPoint[1] > 0 and intPoint[1] < 4.04:
                    points.append([round(intPoint[0], 3), round(intPoint[1], 3)])
            
            # geometric centroid of points 
            if points:
                triLatRes = np.vstack((triLatRes, np.asarray(points).mean(axis = 0)))

        title = 'Participant: ' + participant + '\nActivity: ' + pattern
        print(title)
        print(triLatRes)

        # plotting the points with ground truth
        plt.scatter(triLatRes[1:, 0], triLatRes[1:, 1], s = 2, label='Localization')
        plt.scatter(np.asarray([0, 1.016, 0]), np.asarray([0, 0, 2.032]), c ='r', s = 100, label='Radars')
        radarLabel = ['R 103', 'R 109', ' R \n108']
        for i, txt in enumerate(radarLabel):
            plt.annotate(txt, ([0, 1.016, 0][i], [0, 0, 2.032][i]), ([0.1, 0.9, 0.1][i], [0.05, 0.15, 1.9][i]))
        if pattern in groundTruthMap:
            plt.plot(groundTruthMap[pattern]['x'], groundTruthMap[pattern]['y'], c='g', label='Ground Truth')
        plt.title(title)
        plt.xlim((0, 4.04))
        plt.ylim((0, 4.04))
        plt.legend(loc="upper right")
        plt.savefig('./localization_plots/'+participant + ' ' + pattern + '.png')
        plt.close()
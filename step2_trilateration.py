from scipy.io import loadmat
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def triLat(r1, r2, r3):
    x1, y1 = 0, 0
    x2, y2 = 0, 2.032
    x3, y3 = 1.016, 0

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

pre_path = './DataSet/Localization/'
for participant in ['participant1', 'participant2']:
    for pattern in ['diag', 'four', 'gamma', 'L', 'U']:
        print('Running Pattern: ' + pattern)
        radarData = {'103': {}, '108': {}, '109': {}}
        triLatRes = np.asarray([0, 0])
        for radar in ['103', '108', '109']:
            for files in fileRef:
                radarData[radar][files] = loadmat(pre_path + participant + '/' + radar + '/' + pattern + '/' + fileRef[files])[radarRef[radar][files]]
                if files == 'time' or files == 'bin':
                    radarData[radar][files] = radarData[radar][files].flatten()

        startStamp = max([radarData['103']['time'][0], radarData['108']['time'][0], radarData['109']['time'][0]])
        endStamp = min([radarData['103']['time'][-1], radarData['108']['time'][-1], radarData['109']['time'][-1]])

        timeSyncData = {}
        for radar in ['103', '108', '109']:
            idx1 = np.where(radarData[radar]['time'] > startStamp)[0][0]
            idx2 = np.where(radarData[radar]['time'] < endStamp)[0][-1]

            radarData[radar]['time'] = radarData[radar]['time'][idx1:idx2+1]
            radarData[radar]['mag'] = radarData[radar]['mag'][idx1:idx2+1, :]

            series = pd.DataFrame(data = radarData[radar]['mag'], index = radarData[radar]['time'])
            series.index = pd.to_datetime(series.index, unit='s')
            series = series.resample('20.8L').mean().bfill(axis ='rows')

            timeSyncData[radar] = series.to_numpy()

        window_size = 12
        window_shift = 6
        threshold = 50000
        minIterations = min([np.shape(timeSyncData['103'])[0], np.shape(timeSyncData['108'])[0], np.shape(timeSyncData['109'])[0]])
        iterations = math.floor((minIterations - window_size)/window_shift) + 1
        for i in range(iterations):
            radar1 = timeSyncData['103'][i*window_shift:(i*window_shift)+window_size, :]
            binIndices1 = set(np.where(radar1 > threshold)[1])
            radar2 = timeSyncData['108'][i*window_shift:(i*window_shift)+window_size, :]
            binIndices2 = set(np.where(radar2 > threshold)[1])
            radar3 = timeSyncData['109'][i*window_shift:(i*window_shift)+window_size, :]
            binIndices3 = set(np.where(radar3 > threshold)[1])

            print('Iteration: ' + str(i))
            print(str(len(binIndices1)) + ' ' + str(len(binIndices2)) + ' ' + str(len(binIndices3)))
            points = []
            for i1 in binIndices1:
                r1 = radarData['103']['bin'][i1]
                for i2 in binIndices2:
                    r2 = radarData['108']['bin'][i2]
                    for i3 in binIndices3:
                        r3 = radarData['109']['bin'][i3]
                        intPoint = triLat(r1, r2, r3)
                        if intPoint[0] > 0 and intPoint[1] > 0:
                            points.append([round(intPoint[0], 3), round(intPoint[1], 3)])
            
            if points:
                triLatRes = np.vstack((triLatRes, np.asarray(points).mean(axis = 0)))

        title = 'Participant: ' + participant + '\nActivity: ' + pattern
        print(title)
        print(triLatRes)

        plt.scatter(triLatRes[1:, 0], triLatRes[1:, 1], s = 2)
        plt.title(title)
        plt.xlabel('R 109 at 1.016m')
        plt.ylabel('R 108 at 2.032m')
        plt.savefig(participant + ' ' + pattern + '.png')

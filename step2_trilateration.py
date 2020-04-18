from scipy.io import loadmat
import numpy as np
import pandas as pd

fileRef = {'time': '/T_stmp.mat', 'bin': 'range_bins.mat', 'mag': 'envNoClutterscans.mat'}
radarRef = {'103': {'time': 'T_stmp_1033', 'bin': 'Rbin_1033', 'mag': 'envNoClutterscansV_1033'}, 
            '108': {'time': 'T_stmp_103', 'bin': 'Rbin_103', 'mag': 'envNoClutterscansV_103'}, 
            '109': {'time': 'T_stmp_102', 'bin': 'Rbin_102', 'mag': 'envNoClutterscansV_102'}
            }

pre_path = './DataSet/Localization/'
for participant in ['participant1', 'participant2']:
    for pattern in ['diag', 'four', 'gamma', 'L', 'U']:
        radarData = {'103': {}, '108': {}, '109': {}}
        for radar in ['103', '108', '109']:
            for files in fileRef:
                radarData[radar][files] = loadmat(pre_path + participant + '/' + radar + '/' + pattern + '/' + fileRef[files])[radarRef[radar][files]]
                if files == 'time' or files == 'bin':
                    radarData[radar][files] = radarData[radar][files].flatten()
            
        startStamp = max([radarData['103']['time'][0], radarData['108']['time'][0], radarData['109']['time'][0]])
        endStamp = min([radarData['103']['time'][-1], radarData['108']['time'][-1], radarData['109']['time'][-1]])

        for radar in ['103', '108', '109']:
            idx1 = np.where(radarData[radar]['time'] > startStamp)[0][0]
            idx2 = np.where(radarData[radar]['time'] < endStamp)[0][-1]

            radarData[radar]['time'] = radarData[radar]['time'][idx1:idx2+1]
            radarData[radar]['mag'] = radarData[radar]['mag'][idx1:idx2+1, :]

            series = pd.DataFrame(data = radarData[radar]['mag'], index = radarData[radar]['time'])
            series.index = pd.to_datetime(series.index, unit='s')
            series = series.resample('21.5L').mean()
            
        break
    break


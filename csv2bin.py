import pandas as pd
from pyntcloud import PyntCloud
import os
path = r"E:\互联网+\Hexagon_互联网+竞赛_3D缺陷检测原始点云"
listdir = os.listdir(path)
listdir.remove('33220焊道区域瑕疵点位映射图.xlsx')
for i in listdir:
    path1 = path+'\\'+i
    list = os.listdir(path1)
    for j in list:
        if j.split('.')[1] == 'tiff' or j.split('.')[1] == 'bin':
            continue
        data = pd.read_csv(path1+'\\'+j, names=["x", "y", "z"])
        data = data.drop(data.index[-1])
        data['x'] = pd.to_numeric(data['x'], errors='coerce')
        data['y'] = pd.to_numeric(data['y'], errors='coerce')
        data['z'] = pd.to_numeric(data['z'], errors='coerce')
        cloud = PyntCloud(data)
        cloud.to_file(path1+'\\'+j.split('.')[0]+'.obj', format='OBJ')

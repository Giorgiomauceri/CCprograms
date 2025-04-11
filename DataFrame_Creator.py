import pandas as pd
import awkward as ak
import numpy as np
import uproot

print("  DataFrame_Creator started! \n  Importing tracks...")
f = uproot.open("TNtuples.root")["trackingNtuple/tree;1"]
df = f.arrays(library="pd")

#------------------------------------------------------------------------------------------
print("  Creating Reduced DataFrame...")
temp = ak.argmax(df['trk_simTrkIdx'][0], keepdims=True, axis=-1, mask_identity=False)
DataFrameIds=[df["trk_simTrkIdx"][0][i][temp[i]][0] if temp[i]!=-1 else temp[i][0].astype(np.int32) for i in range(len(temp))]
Check = []
for i in range(len(DataFrameIds)):
    if ((DataFrameIds[i]>0 and df['sim_pt'][0][DataFrameIds[i]]>0.9 and abs(df['sim_eta'][0][DataFrameIds[i]])<2.4) or DataFrameIds[i]<0):
        Check.append(i)
DataFrameIds = [DataFrameIds[trk] for trk in Check]
d = {'truth':[], 'trk_dxy':[], 'trk_dz':[], 'trk_pt':[], 'trk_eta':[], 'trk_nChi2':[], 'trk_nPixel':[], 'trk_inner_px':[], 'trk_inner_py':[], 'trk_inner_pz':[], 'trk_inner_pt':[], 'trk_outer_px':[], 'trk_outer_py':[], 'trk_outer_pz':[], 'trk_outer_pt':[], 'trk_ndof':[]}
DataBase = pd.DataFrame(d)
f = {'truth':[1], 'trk_dxy':[1], 'trk_dz':[1], 'trk_pt':[1], 'trk_eta':[1], 'trk_nChi2':[1], 'trk_nPixel':[1], 'trk_inner_px':[1], 'trk_inner_py':[1], 'trk_inner_pz':[1], 'trk_inner_pt':[1], 'trk_outer_px':[1], 'trk_outer_py':[1], 'trk_outer_pz':[1], 'trk_outer_pt':[1], 'trk_ndof':[1]}
count=0
for ev in range(len(df)):
    temp = ak.argmax(df['trk_simTrkIdx'][ev], keepdims=True, axis=-1, mask_identity=False)
    DataFrameIds=[df["trk_simTrkIdx"][ev][i][temp[i]][0] if temp[i]!=-1 else temp[i][0].astype(np.int32) for i in range(len(temp))]
    #print(DataFrameIds)
    #print(len(DataFrameIds))
    for i in range(len(DataFrameIds)):
        #print(i)
        if ((DataFrameIds[i]>0 and df['sim_pt'][ev][DataFrameIds[i]]>0.9 and abs(df['sim_eta'][ev][DataFrameIds[i]])<2.4) or DataFrameIds[i]<0):
            f = {'truth':[bool(DataFrameIds[i]+1)], 'trk_dxy':[df["trk_dxy"][ev][i]], 'trk_dz':[df["trk_dz"][ev][i]], 'trk_pt':[df["trk_pt"][ev][i]], 'trk_eta':[df["trk_eta"][ev][i]], 'trk_nChi2':[df["trk_nChi2"][ev][i]], 'trk_nPixel':[df["trk_nPixel"][ev][i]], 'trk_inner_px':[df["trk_inner_px"][ev][i]], 'trk_inner_py':[df["trk_inner_py"][ev][i]], 'trk_inner_pz':[df["trk_inner_pz"][ev][i]], 'trk_inner_pt':[df["trk_inner_pt"][ev][i]], 'trk_outer_px':[df["trk_outer_px"][ev][i]], 'trk_outer_py':[df["trk_outer_py"][ev][i]], 'trk_outer_pz':[df["trk_outer_pz"][ev][i]], 'trk_outer_pt':[df["trk_outer_pt"][ev][i]], 'trk_ndof':[df["trk_ndof"][ev][i]]}
            DataBase.loc[count]=f
            #print(ev, i, DataFrameIds[i], df["trk_dxy"][ev][i], df["trk_dz"][ev][i], df["trk_pt"][ev][i], df["trk_eta"][ev][i], df["trk_nChi2"][ev][i], df["trk_nPixel"][ev][i])
            count+=1
print("  Number of selected tracks: ", count)

#------------------------------------------------------------------------------------------
print("  Creating Training and Validation Arrays...")
DataBase1=DataBase.sample(frac=1, ignore_index=True)
for el in DataBase1.columns:
    for i in range(DataBase1.shape[0]):
        DataBase1.loc[i, el]=DataBase1[el][i][0]
TrData=DataBase1.loc[:int(0.8*DataBase1.shape[0])]
ValData=DataBase1.loc[int(0.8*DataBase1.shape[0]):]

data = np.zeros((TrData.shape[0], TrData.shape[1]-1))
target = np.zeros(TrData.shape[0])
for i in range(TrData.shape[0]):
    data[i, 0] = TrData['trk_dxy'][i]
    data[i, 1] = TrData['trk_dz'][i]
    data[i, 2] = TrData['trk_pt'][i]
    data[i, 3] = TrData['trk_eta'][i]
    data[i, 4] = TrData['trk_nChi2'][i]
    data[i, 5] = TrData['trk_nPixel'][i]
    data[i, 6] = TrData['trk_inner_px'][i]
    data[i, 7] = TrData['trk_inner_py'][i]
    data[i, 8] = TrData['trk_inner_pz'][i]
    data[i, 9] = TrData['trk_inner_pt'][i]
    data[i, 10] = TrData['trk_outer_px'][i]
    data[i, 11] = TrData['trk_outer_py'][i]
    data[i, 12] = TrData['trk_outer_pz'][i]
    data[i, 13] = TrData['trk_outer_pt'][i]
    data[i, 14] = TrData['trk_ndof'][i]
    target[i]  = TrData['truth'][i]

data1 = np.zeros((ValData.shape[0], ValData.shape[1]-1))
target1 = np.zeros(ValData.shape[0])
for i in range(ValData.shape[0]):
    data1[i, 0] = ValData['trk_dxy'][TrData.shape[0]+i-1]
    data1[i, 1] = ValData['trk_dz'][TrData.shape[0]+i-1]
    data1[i, 2] = ValData['trk_pt'][TrData.shape[0]+i-1]
    data1[i, 3] = ValData['trk_eta'][TrData.shape[0]+i-1]
    data1[i, 4] = ValData['trk_nChi2'][TrData.shape[0]+i-1]
    data1[i, 5] = ValData['trk_nPixel'][TrData.shape[0]+i-1]
    data1[i, 6] = ValData['trk_inner_px'][TrData.shape[0]+i-1]
    data1[i, 7] = ValData['trk_inner_py'][TrData.shape[0]+i-1]
    data1[i, 8] = ValData['trk_inner_pz'][TrData.shape[0]+i-1]
    data1[i, 9] = ValData['trk_inner_pt'][TrData.shape[0]+i-1]
    data1[i, 10] = ValData['trk_outer_px'][TrData.shape[0]+i-1]
    data1[i, 11] = ValData['trk_outer_py'][TrData.shape[0]+i-1]
    data1[i, 12] = ValData['trk_outer_pz'][TrData.shape[0]+i-1]
    data1[i, 13] = ValData['trk_outer_pt'][TrData.shape[0]+i-1]
    data1[i, 14] = ValData['trk_ndof'][TrData.shape[0]+i-1]
    target1[i]  = ValData['truth'][TrData.shape[0]+i-1]
    
#------------------------------------------------------------------------------------------
print("  Saving files...")
np.savetxt('tempData/Training_Data.dat', data)
np.savetxt('tempData/Training_Truth.dat', target)
np.savetxt('tempData/Test_Data.dat', data1)
np.savetxt('tempData/Test_Truth.dat', target1)











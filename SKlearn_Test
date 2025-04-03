import pandas as pd
import matplotlib.pyplot as plt
import torch
import awkward as ak
import numpy as np
import uproot
import sklearn as sk

print("  Tracking Ntuple Analysis started \n  Importing tracks...")
f = uproot.open("TNtuples.root")["trackingNtuple/tree;1"]
df = f.arrays(library="pd")

#------------------------------------------------------------------------------------------
print("  Creating Neural Network Model...")
class TrackAnalysisModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(in_features=6, out_features=30)
        self.layer_2 = torch.nn.Linear(in_features=30, out_features=30)
        self.layer_3 = torch.nn.Linear(in_features=30, out_features=30)
        self.layer_4 = torch.nn.Linear(in_features=30, out_features=30)
        self.layer_5 = torch.nn.Linear(in_features=30, out_features=30)
        self.layer_6 = torch.nn.Linear(in_features=30, out_features=1)
        self.relu = torch.nn.ReLU() # <- add in ReLU activation function
        # Can also put sigmoid in the model 
        # This would mean you don't need to use it on the predictions
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       #return self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))))
       return self.layer_6(self.relu(self.layer_5(self.relu(self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))))))))


#------------------------------------------------------------------------------------------
print("  Creating Reduced DataFrame...")
temp = ak.argmax(df['trk_simTrkIdx'][0], keepdims=True, axis=-1, mask_identity=False)
DataFrameIds=[df["trk_simTrkIdx"][0][i][temp[i]][0] if temp[i]!=-1 else temp[i][0].astype(np.int32) for i in range(len(temp))]
Check = []
for i in range(len(DataFrameIds)):
    if ((DataFrameIds[i]>0 and df['sim_pt'][0][DataFrameIds[i]]>0.9 and abs(df['sim_eta'][0][DataFrameIds[i]])<2.4) or DataFrameIds[i]<0):
        Check.append(i)
DataFrameIds = [DataFrameIds[trk] for trk in Check]
d = {'truth':[], 'trk_dxy':[], 'trk_dz':[], 'trk_pt':[], 'trk_eta':[], 'trk_nChi2':[], 'trk_nPixel':[]}
DataBase = pd.DataFrame(d)
f = {'truth':[1], 'trk_dxy':[1], 'trk_dz':[1], 'trk_pt':[1], 'trk_eta':[1], 'trk_nChi2':[1], 'trk_nPixel':[1]}
count=0
for ev in range(len(df)):
    temp = ak.argmax(df['trk_simTrkIdx'][ev], keepdims=True, axis=-1, mask_identity=False)
    DataFrameIds=[df["trk_simTrkIdx"][ev][i][temp[i]][0] if temp[i]!=-1 else temp[i][0].astype(np.int32) for i in range(len(temp))]
    #print(DataFrameIds)
    #print(len(DataFrameIds))
    for i in range(len(DataFrameIds)):
        #print(i)
        if ((DataFrameIds[i]>0 and df['sim_pt'][ev][DataFrameIds[i]]>0.9 and abs(df['sim_eta'][ev][DataFrameIds[i]])<2.4) or DataFrameIds[i]<0):
            f = {'truth':[bool(DataFrameIds[i]+1)], 'trk_dxy':[df["trk_dxy"][ev][i]], 'trk_dz':[df["trk_dz"][ev][i]], 'trk_pt':[df["trk_pt"][ev][i]], 'trk_eta':[df["trk_eta"][ev][i]], 'trk_nChi2':[df["trk_nChi2"][ev][i]], 'trk_nPixel':[df["trk_nPixel"][ev][i]]}
            DataBase.loc[count]=f
            #print(ev, i, DataFrameIds[i], df["trk_dxy"][ev][i], df["trk_dz"][ev][i], df["trk_pt"][ev][i], df["trk_eta"][ev][i], df["trk_nChi2"][ev][i], df["trk_nPixel"][ev][i])
            count+=1
print("  Number of selected tracks: ", count)

#------------------------------------------------------------------------------------------
print("  Creating Training and Validation Tensors...")
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
    target1[i]  = ValData['truth'][TrData.shape[0]+i-1]

target = torch.from_numpy(target).type(torch.float)
data = torch.from_numpy(data).type(torch.float)
target1 = torch.from_numpy(target1).type(torch.float)
data1 = torch.from_numpy(data1).type(torch.float)

#------------------------------------------------------------------------------------------
print("  Analysis Tools Setup...")
# Calculate accuracy (a classification metric)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TrackAnalysisModel().to(device)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

# Setup loss and optimizer 
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#------------------------------------------------------------------------------------------
print("  Training the Neural Network...")
# Fit the model
torch.manual_seed(42)
epochs = 6000

# Put all data on target device
data, target = data.to(device), target.to(device)
data1, target1 = data1.to(device), target1.to(device)

for epoch in range(epochs):
    # 1. Forward pass
    y_logits = model(data).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
    
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, target) # BCEWithLogitsLoss calculates loss using logits
    acc = accuracy_fn(y_true=target, 
                      y_pred=y_pred)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model(data1).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
      # 2. Calculate loss and accuracy
      test_loss = loss_fn(test_logits, target1)
      test_acc = accuracy_fn(y_true=target1,
                             y_pred=test_pred)

    # Print out what's happening
    if epoch % 200 == 0:
        print(f"    Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

#------------------------------------------------------------------------------------------
print("  Printing ROC curve:")
PredVal = torch.sigmoid(model(data1).squeeze())
PredVal = PredVal.detach().numpy()
fpr, tpr, thr= sk.metrics.roc_curve(target1, PredVal, drop_intermediate=False)
plt.plot(fpr, tpr)
plt.show()
print("  ROC curve knee: ", fpr[1])

ROCdata = np.column_stack([fpr, tpr])
FakeData = np.column_stack([PredVal, target1])
np.savetxt("ROC_NN5x30.dat" , ROCdata)
np.savetxt("FAKE_NN5x30.dat", FakeData)
print("  Finished")

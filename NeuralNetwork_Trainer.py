import pandas as pd
import matplotlib.pyplot as plt
import torch
import awkward as ak
import numpy as np
import uproot
import sklearn as sk

print("  NeuralNetwork_Trainer started! \n  Importing Arrays from files...")
target =  np.loadtxt("tempData/Training_Truth.dat")
data = np.loadtxt("tempData/Training_Data.dat")
target1 = np.loadtxt("tempData/Test_Truth.dat")
data1 = np.loadtxt("tempData/Test_Data.dat")

#------------------------------------------------------------------------------------------
print("  Creating Training and Testing Tensors...")
target = torch.from_numpy(target).type(torch.float)
data = torch.from_numpy(data).type(torch.float)
target1 = torch.from_numpy(target1).type(torch.float)
data1 = torch.from_numpy(data1).type(torch.float)

#------------------------------------------------------------------------------------------
print("  Creating Neural Network Model...")
class TrackAnalysisModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(in_features=6, out_features=30)
        self.layer_2 = torch.nn.Linear(in_features=30, out_features=30)
        self.layer_3 = torch.nn.Linear(in_features=30, out_features=30)
        self.layer_4 = torch.nn.Linear(in_features=30, out_features=1)
        self.relu = torch.nn.ReLU() # <- add in ReLU activation function
        # Can also put sigmoid in the model 
        # This would mean you don't need to use it on the predictions
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))))
       #return self.layer_6(self.relu(self.layer_5(self.relu(self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))))))))

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
np.savetxt("tempData/ROC_NN3x30.dat" , ROCdata)
np.savetxt("tempData/FAKE_NN3x30.dat", FakeData)
print("  Finished")
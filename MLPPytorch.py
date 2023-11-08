
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pdb


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(17952, 6400)
        self.fc2 = nn.Linear(6400, 3200)
        self.fc3 = nn.Linear(3200, 320)
        self.fc4 = nn.Linear(320, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.log_softmax(self.fc4(x), dim=1)


def LoadData(l,s,a):
   for label in np.arange(10):
       print(f"Loading E{label} Data")
       tmp = np.load(f"../PDSaveData/SavedData_E{label}_l{l}_s{s}_a{a}.npy",allow_pickle=True)
       if label == 0:
          Zload=tmp.copy()
       else:
          Zload=np.concatenate((Zload,tmp),axis=0)

   X=Zload[:,:-1]
   y=Zload[:,-1]
   return X,y


if __name__=='__main__':
   torch.manual_seed(42)

   #Parameters
   l = 11
   s = 4
   a = 20

   batch_size = 100
   MaxEpoch = 20

   X,y = LoadData(l,s,a)

   tensor_x = torch.Tensor(X) # transform to torch tensor
   tensor_y = torch.Tensor(y)

   dataset = TensorDataset(tensor_x,tensor_y) # create your datset

   trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
   testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

   model = MLP()
   criterion = nn.NLLLoss()
   optimizer = optim.SGD(model.parameters(), lr=1e-5)

   for epoch in range(0,MaxEpoch):
      current_loss = 0.0
      current_correct = 0.0

      for i, data in enumerate(trainloader, 0):
          inputs, targets = data
          inputs, targets = inputs.float(), targets.long() # --> NLLLoss

          optimizer.zero_grad()

          outputs = model(inputs)

          loss = criterion(outputs, targets)


          loss.backward()

          optimizer.step()

          current_loss += loss.item()

          output = outputs.argmax(dim=1).float()
          current_correct += (output == targets).float().sum() 

          if i%10 == 0:
             print(f"Epoch {epoch+1}/{MaxEpoch} - minibatch {i+1}, Loss: {current_loss/((i+1)*batch_size)}, Accuracy: {current_correct/((i+1)*batch_size)}")

      print(f"Epoch {epoch+1}/{MaxEpoch}, Loss: {current_loss/len(trainloader)}, Accuracy: {current_correct/len(trainloader)}")

   print("Training has completed")




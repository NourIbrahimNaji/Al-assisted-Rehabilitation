import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

start_time = datetime.now()

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1, :, :])
        out = self.softmax(out)
        return out

def LoadData(l, s, a):
    for label in np.arange(10):
        print(f"Loading E{label} Data")
        tmp = np.load(f"../Al-assisted-Rehabilitation/SavedData_E{label}_l{l}_s{s}_a{a}.npy", allow_pickle=True)
        if label == 0:
            Zload = tmp.copy()
        else:
            Zload = np.concatenate((Zload, tmp), axis=0)

    X = Zload[:, :-1]
    y = Zload[:, -1]
    return X, y

if __name__ == '__main__':
    torch.manual_seed(42)

    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # You can print the selected device for verification
    print("Using device:", device)

    # Parameters
    l = 3
    s = 4
    a = 10

    batch_size = 100
    MaxEpoch = 3000

    X, y = LoadData(l, s, a)

    input_size = X.shape[1]
    print(f"Feature vector length (input_size) = {input_size}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tensor_x_train = torch.Tensor(X_train)  # transform to torch tensor
    tensor_y_train = torch.Tensor(y_train)

    tensor_x_test = torch.Tensor(X_test)  # transform to torch tensor
    tensor_y_test = torch.Tensor(y_test)

    dataset_train = TensorDataset(tensor_x_train, tensor_y_train)  # create your dataset
    dataset_test = TensorDataset(tensor_x_test, tensor_y_test)  # create your dataset

    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)

    lstm_model = LSTMClassifier(input_size, hidden_size=64, output_size=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=1e-4)

    for epoch in range(0, MaxEpoch):
        # Training
        current_loss = 0.0
        current_correct = 0.0

        # Set the model to train mode
        lstm_model.train()

        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float().to(device), targets.long().to(device)

            optimizer.zero_grad()
            outputs = lstm_model(inputs.unsqueeze(1))  # Add a dummy dimension for the sequence length
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            output = outputs.argmax(dim=1).float()
            current_correct += (output == targets).float().sum()

        train_loss = current_loss / (len(trainloader) * batch_size)
        train_accuracy = 100 * current_correct / (len(trainloader) * batch_size)

        print(f"Epoch {epoch + 1}/{MaxEpoch} - Training, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")



        # Testing
        current_loss = 0.0
        current_correct = 0.0

        # Set the model to evaluation mode
        lstm_model.eval()

        # Disable gradient computation for evaluation
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, targets = data
                inputs, targets = inputs.float().to(device), targets.long().to(device)

                outputs = lstm_model(inputs.unsqueeze(1))  # Add a dummy dimension for the sequence length
                loss = criterion(outputs, targets)

                current_loss += loss.item()
                output = outputs.argmax(dim=1).float()
                current_correct += (output == targets).float().sum()

        test_loss = current_loss / (len(testloader) * batch_size)
        test_accuracy = 100 * current_correct / (len(testloader) * batch_size)

        print(f"Epoch {epoch + 1}/{MaxEpoch} - Testing, Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%\n")


    print("Training has completed")

print("--- Time: %s  ---" % (datetime.now() - start_time))



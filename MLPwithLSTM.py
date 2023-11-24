import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1, :, :]

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 6400)
        self.fc2 = nn.Linear(6400, 3200)
        self.fc3 = nn.Linear(3200, 320)
        self.fc4 = nn.Linear(320, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.log_softmax(self.fc4(x), dim=1)

class CombinedModel(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, mlp_input_size, mlp_output_size):
        super(CombinedModel, self).__init__()
        self.lstm_encoder = LSTMEncoder(lstm_input_size, lstm_hidden_size, lstm_num_layers)
        self.mlp = MLP(mlp_input_size, mlp_output_size)

    def forward(self, x):
        lstm_output = self.lstm_encoder(x)
        mlp_input = lstm_output.view(lstm_output.size(0), -1)
        return self.mlp(mlp_input)

# Modify the LoadData function to include the LSTM encoder
def LoadData(l, s, a):
    for label in np.arange(10):
        print(f"Loading E{label} Data")
        tmp = np.load(
            f"C:\\Users\\shaher\\Desktop\\Introduction to graduation project\\Al-assisted-Rehabilitation\\saveData\\SavedData_E{label}_l{l}_s{s}_a{a}.npy",
            allow_pickle=True)

        if label == 0:
            Zload = tmp.copy()
        else:
            Zload = np.concatenate((Zload, tmp), axis=0)

    X = Zload[:, :-1]
    y = Zload[:, -1]

    # Reshape X to add a third dimension for the LSTM input
    X = X.reshape(X.shape[0], 1, X.shape[1])

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

    # Parameters for the LSTM encoder
    lstm_input_size = X.shape[2]  # Assuming the third dimension is the feature dimension
    lstm_hidden_size = 128
    lstm_num_layers = 1

    # Parameters for the MLP
    mlp_input_size = lstm_hidden_size  # Use the output size of the LSTM as input to the MLP
    mlp_output_size = 10  # Output size of the MLP (number of classes)

    # Create the combined model
    combined_model = CombinedModel(lstm_input_size, lstm_hidden_size, lstm_num_layers, mlp_input_size, mlp_output_size).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(combined_model.parameters(), lr=1e-6)

    for epoch in range(0, MaxEpoch):
        # Training
        current_loss = 0.0
        current_correct = 0.0

        # Set the model to train mode
        combined_model.train()

        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float().to(device), targets.long().to(device)

            optimizer.zero_grad()
            outputs = combined_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            output = outputs.argmax(dim=1).float()
            current_correct += (output == targets).float().sum()

        print(f"Epoch {epoch+1}/{MaxEpoch} - Training, Loss: {current_loss/(len(trainloader)*batch_size):.4f}, Accuracy: {100*current_correct/(len(trainloader)*batch_size):.2f}%")

        # Testing
        current_loss = 0.0
        current_correct = 0.0

        # Set the model to evaluation mode
        combined_model.eval()

        # Disable gradient computation for evaluation
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, targets = data
                inputs, targets = inputs.float().to(device), targets.long().to(device)

                outputs = combined_model(inputs)
                loss = criterion(outputs, targets)

                current_loss += loss.item()
                output = outputs.argmax(dim=1).float()
                current_correct += (output == targets).float().sum()

        print(f"Epoch {epoch+1}/{MaxEpoch} - Testing, Loss: {current_loss/(len(testloader)*batch_size):.4f}, Accuracy: {100*current_correct/(len(testloader)*batch_size):.2f}%\n")

    print("Training has completed")

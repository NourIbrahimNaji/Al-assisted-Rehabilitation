import os
import torch
import torch.nn as nn
from torch.nn import MSELoss
import numpy as np
import LSTM_AutoEncoder_Class as LSTMC
from sklearn.model_selection import train_test_split

import pdb

def LoadData(path):
   # ------------------------------------------- LOAD .npz FILES --------------------------------------------- #
   dictionary = {}  # dictionary to store all data
   # files path
   path = '/AIARUPD/Dataset_CVDLPT_Videos_Segments_npz_11_2023'
   # loop through all files and store them in the dictionary
   for npzFile in os.listdir(path):
      f = os.path.join(path, npzFile)
      if os.path.isfile(f):
         if "_3D" in f:
            fdata = np.load(f)
            dictionary[npzFile.split('_3D')[0]] = fdata['reconstruction'][0, :, :, :]

   return dictionary


if __name__=='__main__':
   torch.manual_seed(42)

   # Check if CUDA (GPU) is available
   if torch.cuda.is_available():
       device = torch.device("cuda")
   else:
       device = torch.device("cpu")

   # You can print the selected device for verification
   print("Using device:", device)

   # Specify the path to the dataset
   path = '/AIARUPD/Dataset_CVDLPT_Videos_Segments_npz_11_2023'

   # Load the data from the specified path
   data = LoadData(path)

   # Extract keys and labels from the data dictionary
   y = list(data.keys())
   X = [data[key].reshape(1,-1,51) for key in y]

   # Split the data into training and testing sets
   test_size = 0.2  # Adjust the test_size as needed
   random_seed = 42  # Set a random seed for reproducibility
   x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

   # Define the input size for the LSTM Autoencoder
   input_size = 51

   # Define the size of the hidden layer in the LSTM Autoencoder
   hidden_size = 16

   # Define the number of layers in the LSTM Autoencoder
   num_layers = 2

   # Create an instance of the LSTM Autoencoder model
   model = LSTMC.LSTM_AE(input_size, hidden_size, num_layers, device)

   # Move the model to the specified device (e.g., GPU)
   model.to(device)

   # Initialize the optimizer, loss criterion, and training history
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   criterion = nn.L1Loss(reduction='sum').to(device)
   history = dict(train=[], val=[])

   # Initialize the best loss for model checkpointing
   best_loss = 10000.0

   # Number of training epochs
   n_epochs = 20

   # Main training loop
   for epoch in range(1, n_epochs + 1):
      # Set the model to training mode
      model = model.train()
      train_losses = []

      # Training loop for each sequence in the training set
      for seq_true in x_train:
         optimizer.zero_grad()

         # Move the sequence to the specified device
         seq_true = torch.tensor(seq_true).to(device)

         # Forward pass through the model
         _, seq_pred = model(seq_true, seq_true.shape[1])

         # Calculate the loss and perform backpropagation
         loss = criterion(seq_pred, seq_true)
         loss.backward()
         optimizer.step()
         train_losses.append(loss.item())

      # Validation loop
      val_losses = []
      model = model.eval()

      # Disable gradient calculation during validation
      with torch.no_grad():
         for seq_true in x_test:
            optimizer.zero_grad()
            seq_true = torch.tensor(seq_true).to(device)
            _, seq_pred = model(seq_true, seq_true.shape[1])
            loss = criterion(seq_pred, seq_true)
            val_losses.append(loss.item())

      # Calculate mean training and validation losses
      train_loss = np.mean(train_losses)
      val_loss = np.mean(val_losses)

      # Store losses in the history dictionary
      history['train'].append(train_loss)
      history['val'].append(val_loss)

      # Print the training and validation losses for each epoch
      print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')


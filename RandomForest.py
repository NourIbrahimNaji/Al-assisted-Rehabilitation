import bdb

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the data from the .npz file
EX1 = np.load('E0_P0_T0_C0_3D.npz')
EX2=np.load('E1_P0_T0_C0_3D.npz')

# Extract the data from the loaded file
data_array1 = EX1['reconstruction']
data_array2 = EX2['reconstruction']
i=0
bdb.set_trace()
print(data_array1.shape[1])

# Define the shape of the empty array to store each 70 frame with joints and dimension
shape = (70, data_array1.shape[2], data_array1.shape[3])  # (blocks, rows, columns) == (70,17,3)

# defining an array to store each 70 frames value inside it..
training_array = np.empty(shape)

print(training_array.shape) # == (70,17,3)
# for each frame in frame range (0=>713)
for frame in range(data_array1.shape[1]):

    training_array = data_array1[0][frame]
    if frame % 70 == 0: # if we reach the frame 70, 140, 210.....
        # array for training.. (input)
        #print("==========",training_array)
        features = np.array(training_array)

        # make the array empty for the next 70 frame..
        training_array = np.empty(shape)
        print("eman")
 # array for testing.. (output)
labels = np.array(i+1)
print("features--------------->", features)
print("labeles--------------->",labels)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create an instance of the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Train the Random Forest classifier using the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
predictions = rf_classifier.predict(X_test)

# Evaluate the performance of the model
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy}")
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# ------------------------------------------- LOAD .npz FILES --------------------------------------------- #

start_time = datetime.now()
dictionary3D = {}  # dictionary to store all 3D files
dictionary2D = {}  # dictionary to store all 2D files
# files path
path = 'C:\\Users\\shaher\\Desktop\\GraduationProject\\GradProject\\3D_blurred - Copy\\'
# loop through all files and store them in the dictionary
for npzFile in os.listdir(path):
    f = os.path.join(path, npzFile)
    # checking if it is a file
    if os.path.isfile(f):
        if "_3D" in f:
            a = np.load(f)
            # load the files into the dictionary
            dictionary3D[npzFile.split('_3D')[0]] = a['reconstruction'][0, :, :, :]
        elif "_2D" in f:
            a = np.load(f)
            # load the files into the dictionary
            dictionary2D[npzFile.split('_2D')[0]] = a['reconstruction'][0, :, :, :]

# gat the max frame size from files
max_length = 0
max_length_file = None
for file_name, reconstruction_array in dictionary3D.items():
    current_length = reconstruction_array.shape[0]  # Assuming the length is along the first dimension
    if current_length > max_length:
        max_length = current_length
        max_length_file = file_name

for k in dictionary3D:
    if dictionary3D[k].shape[0] < max_length:
        dictionary3D[k] = np.concatenate((dictionary3D[k], np.zeros(
            (max_length - len(dictionary3D[k]), dictionary3D[k].shape[1], dictionary3D[k].shape[2]))), axis=0)

# ---------------------------- MLP Features Extraction -----------------------------

# Create the dataset for feature extraction
X = np.array([dictionary3D[key].flatten() for key in dictionary3D])
y = np.array([label.split('_')[0] for label in dictionary3D])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Extract features from the hidden layer
hidden_layer_index = 0  # Adjust this based on your network architecture
hidden_layer_output_train = np.maximum(0, np.dot(X_train, mlp.coefs_[hidden_layer_index]) + mlp.intercepts_[
    hidden_layer_index])
hidden_layer_output_test = np.maximum(0, np.dot(X_test, mlp.coefs_[hidden_layer_index]) + mlp.intercepts_[
    hidden_layer_index])

# Train a new classifier on the extracted features
feature_classifier = MLPClassifier(hidden_layer_sizes=(10,), max_iter=5000, random_state=42)
feature_classifier.fit(hidden_layer_output_train, y_train)

# Calculate training accuracy
train_accuracy = feature_classifier.score(hidden_layer_output_train, y_train)
print("Training Accuracy MLP: ", train_accuracy)

# Calculate the test accuracy
test_accuracy = feature_classifier.score(hidden_layer_output_test, y_test)
print("Testing Accuracy MLP: ", test_accuracy)

# Predict on testing data using the feature classifier
y_feature_pred = feature_classifier.predict(hidden_layer_output_test)
accuracy_feature = accuracy_score(y_test, y_feature_pred)
print(f"Accuracy using extracted features: {accuracy_feature}")
precision_MLP = precision_score(y_test, y_feature_pred, average='weighted', zero_division=1)
recall_MLP = recall_score(y_test, y_feature_pred, average='weighted', zero_division=1)
f1_MLP = f1_score(y_test, y_feature_pred, average='weighted', zero_division=1)
class_report_MLP = classification_report(y_test, y_feature_pred)

print("precision", precision_MLP)
print("recall", recall_MLP)
print("f1 score ", f1_MLP)
print("Classification Report:\n", class_report_MLP)
with open('ExtractFeatures_MLP.txt', 'w') as f:
    f.write(f"Execution time: {(datetime.now() - start_time)} seconds ---  \n")
    f.write(f"      ----------- MLP -------------\n")
    f.write(f"Accuracy: {accuracy_feature}\n")
    f.write(f"precision: {precision_MLP}\n")
    f.write(f"recall: {recall_MLP}\n")
    f.write(f"f1 score: {f1_MLP}\n")
    f.write(f"Classification Report:\n {class_report_MLP}\n")

print("--- %s seconds ---" % (datetime.now() - start_time))

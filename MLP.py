import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from datetime import datetime

# Parameters
a = 10
l = 3
s = 4

for label in np.arange(10):
    print(label)
    tmp = np.load(f"../Al-assisted-Rehabilitation/SavedData_E{label}_l{l}_s{s}_a{a}.npy",allow_pickle=True)
    if label == 0:
        Zload = tmp.copy()
    else:
        Zload = np.concatenate((Zload, tmp), axis=0)

X = Zload[:, :-1]
y = Zload[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("------------------------------------------------------------------------------")
print("                                MLP Classifier                              \n")
# Create an MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(6400,3200,1600,800,400,200,10), max_iter=1000, random_state=42)

start_time = datetime.now()
# Train the MLP on the training data
mlp_classifier.fit(X_train, y_train)
MLP_training_time = datetime.now() - start_time
print(" MLP training time: %s seconds --- " % (MLP_training_time.total_seconds()))

start_time = datetime.now()
# Make predictions on the test data
y_pred_MLP = mlp_classifier.predict(X_test)
MLP_testing_time = datetime.now() - start_time
print(" MLP testing time: %s seconds --- " % (MLP_testing_time).total_seconds())

# Calculate accuracy...
accuracy_MLP = accuracy_score(y_test, y_pred_MLP)
precision_MLP = precision_score(y_test, y_pred_MLP, average='weighted', zero_division=1)
recall_MLP = recall_score(y_test, y_pred_MLP, average='weighted', zero_division=1)
f1_MLP = f1_score(y_test, y_pred_MLP, average='weighted', zero_division=1)
class_report_MLP = classification_report(y_test, y_pred_MLP)

# Calculate training accuracy
train_accuracy = mlp_classifier.score(X_train, y_train)
print("Training Accuracy MLP: ", train_accuracy)

# Calculate testing accuracy
test_accuracy = mlp_classifier.score(X_test, y_test)
print("Testing Accuracy MLP: ", test_accuracy)

print("Accuracy:", accuracy_MLP)
print("precision", precision_MLP)
print("recall", recall_MLP)
print("f1 score ", f1_MLP)
print("Classification Report:\n", class_report_MLP)

with open('results_MLP.txt', 'w') as f:
       f.write(f"{train_accuracy}      {test_accuracy}     {MLP_training_time}      {MLP_testing_time}\n")
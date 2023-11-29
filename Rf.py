import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from datetime import datetime


# Parameters

l = 21
s = 4
a = 10

for label in np.arange(10):
    print(label)
    tmp = np.load(
        f"C:\\Users\\shaher\\Desktop\\Introduction to graduation project\\Al-assisted-Rehabilitation\\saveData\\SavedData_E{label}_l{l}_s{s}_a{a}.npy",
        allow_pickle=True)
    if label == 0:
        Zload = tmp.copy()
    else:
        Zload = np.concatenate((Zload, tmp), axis=0)

X = Zload[:, :-1]
y = Zload[:, -1]


#           -------------- Random Forest --------------

print("                                Random-forest Classifier                              \n")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=5, random_state=42)


start_time_training = datetime.now()
# Train the classifier
rf_classifier.fit(X_train, y_train)
RF_training_time = (datetime.now() - start_time_training).total_seconds()

start_time_testing = datetime.now()
# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)
RF_testing_time = (datetime.now() - start_time_testing).total_seconds()

# Evaluate the accuracy of the classifier
accuracy_RF = accuracy_score(y_test, y_pred)
precision_RF = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall_RF = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1_RF = f1_score(y_test, y_pred, average='weighted', zero_division=1)
class_report_RF = classification_report(y_test, y_pred)

# Calculate training accuracy
train_accuracy = rf_classifier.score(X_train, y_train)
print("Training Accuracy RF: ", train_accuracy)

# Calculate testing accuracy
test_accuracy = rf_classifier.score(X_test, y_test)
print("Testing Accuracy RF: ", test_accuracy)

print("Accuracy:", accuracy_RF)


with open('results_RF.txt', 'a') as f:

    f.write(f"{train_accuracy}      {test_accuracy}     {RF_training_time}      {RF_testing_time}\n")
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  cross_validate

# Parameters
l = 3
s = 4
a = 10
for label in np.arange(10):
    print(label)
    tmp = np.load( f"C:\\Users\\shaher\\Desktop\\Introduction to graduation project\\Al-assisted-Rehabilitation\\saveData\\SavedData_E{label}_l{l}_s{s}_a{a}.npy",
        allow_pickle=True)
    if label == 0:
        Zload = tmp.copy()
    else:
        Zload = np.concatenate((Zload, tmp), axis=0)

x = Zload[:, :-1]
y = Zload[:, -1]

# Create an instance of the model
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Replace with your chosen model

# Perform cross-validation and return specified metrics
cv_results = cross_validate(
    estimator=model,
    X=x,
    y=y,
    cv=5,
    n_jobs=None,
    return_train_score=True,
    return_estimator=False,
    return_indices=False,
    scoring=None,
    groups=None,
    verbose=0,
    fit_params=None,
    pre_dispatch='2*n_jobs',
    error_score=float('nan')
)

# Extract the relevant metrics
test_score = cv_results['test_score']
train_score = cv_results['train_score']
fit_time = cv_results['fit_time']
score_time = cv_results['score_time']


# Print or use the metrics as needed
print("Train Accurecy:", np.mean(train_score))
print("Test Accurecy:", np.mean(test_score))
print("Train Time (Sec):", np.mean(fit_time))
print("Test Time (Sec):", np.mean(score_time))

with open('results_CV.txt', 'w') as f:
    f.write(f"{np.mean(train_score)}      {np.mean(test_score)}     {np.mean(fit_time)}      {np.mean(score_time)}\n")
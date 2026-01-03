"""
Baseline Non-Pipelined ANN for Alzheimer's Disease Detection
This implementation serves as a centralized baseline for future
experiments.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

"""## Part 1 - Data Preprocessing

### Importing the dataset
"""

df = pd.read_csv("alzheimers_disease_data.csv")
df = df.drop(columns=["PatientID", "DoctorInCharge"])
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

"""### Encoding categorical data

One Hot Encoding the "Geography" column
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

"""### Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

"""### Feature Scaling"""

#apply feature scaling to all variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) #feature scaling applied to all variables in x_train
X_test = sc.transform(X_test)


"""## Part 2 - Building the ANN

### Initializing the ANN
"""

ann = tf.keras.models.Sequential()

"""### Adding the input layer and the first hidden layer"""

ann.add(tf.keras.layers.Dense(units=4, activation='relu'))

"""### Adding the second hidden layer"""

ann.add(tf.keras.layers.Dense(units=4, activation='relu'))

"""### Adding the output layer"""

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

"""## Part 3 - Training the ANN

### Compiling the ANN
"""

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""### Training the ANN on the Training set"""

ann.fit(X_train, y_train, batch_size = 32, epochs=50, validation_data=(X_test, y_test))

"""## Part 4 - Making the predictions and evaluating the model

### Predicting the Test set results
"""

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5) #if greater than 0.5 the output=1 else 0 for all observations
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.to_numpy().reshape(len(y_test),1)),1))

"""### Making the Confusion Matrix"""

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score, classification_report, ConfusionMatrixDisplay,
    precision_recall_curve
)
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Depression", "Depression"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

print(f"Accuracy is: {accuracy_score(y_test, y_pred)}")

"""###Classification report"""

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Depression", "Depression"]))

loss, acc = ann.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Loss: {loss:.4f}")

"""###Model Losss and Accuracy Curves"""

plt.figure()
plt.plot(ann.history.history['loss'], label='Train Loss')
plt.plot(ann.history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(ann.history.history['accuracy'], label='Train Accuracy')
plt.plot(ann.history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
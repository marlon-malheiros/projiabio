import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn              as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import torch
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from keras.optimizers import Adam

import pyswarms as ps


# PSO to optmize weights and bias of a neural network to classify EEG signals
df = pd.read_csv('subj07_df.csv')
df = df.sort_values('label')
df = df[df['label'].isin([0, 4])]
df = df.groupby('label').apply(lambda x: x.sample(n=5000)).reset_index(drop=True)
# Reorder dataset


n_inputs = 32
n_hidden = 64
n_classes = 2

num_samples = df.shape[0]

# [0:2048],[2048,2112],[2112,2496],[2496,2502]

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Remap labels
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# normalize X 
X = (X - X.min()) / (X.max() - X.min())

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(32,)))
model.add(Dense(2, activation='softmax'))  # replace 6 with the number of classes you have

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

history = model.fit(X_train, y_train, epochs=50, batch_size=100)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

# plot accuracy history
plt.plot(history.history['accuracy'], label='train')
plt.title('Evolution of best accuracy along iterations')
plt.xlabel('Iterations')
plt.ylabel('Best Accuracy')
plt.legend()
plt.show()

# acuracy, precision, f1-score, recall
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)




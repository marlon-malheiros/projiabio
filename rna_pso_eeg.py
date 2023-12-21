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

import pyswarms as ps
from pyswarms.backend.topology import Pyramid
from pyswarms.utils.functions import single_obj as fx

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


# normalize X with standar scaler
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

X = (X - X.min()) / (X.max() - X.min())

# Define the model
model = Sequential()
model.add(Dense(n_hidden, input_dim=n_inputs, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train, epochs=50, batch_size=100)


def relu(x):
    return np.maximum(0, x)

def logits_function(p):
    """ Calculate roll-back the weights and biases

    Inputs
    ------
    p: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    numpy.ndarray of logits for layer 2

    """
    # Roll-back the weights and biases
    W1 = p[0:2048].reshape((n_inputs,n_hidden))
    b1 = p[2048:2112].reshape((n_hidden,))
    W2 = p[2112:2240].reshape((n_hidden,n_classes))
    b2 = p[2240:2242].reshape((n_classes,))

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    #a1 = np.tanh(z1)     # Activation in Layer 1
    a1 = relu(z1)     # Activation in Layer 1
    logits = a1.dot(W2) + b2 # Pre-activation in Layer 2
    return logits          # Logits for Layer 2

# Forward propagation
def forward_prop(params):
    """Forward propagation as objective function

    This computes for the forward propagation of the neural network, as
    well as the loss.

    Inputs
    ------
    params: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    float
        The computed negative log-likelihood loss given the parameters
    """

    logits = logits_function(params)

    # Compute for the softmax of the logits
    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute for the negative log likelihood

    corect_logprobs = -np.log(probs[range(num_samples), y])
    loss = np.sum(corect_logprobs) / num_samples

    return loss

def f(x):
    """Higher-level method to do forward_prop in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)



def predict(pos):
    """
    Use the trained weights to perform class predictions.

    Inputs
    ------
    pos: numpy.ndarray
        Position matrix found by the swarm. Will be rolled
        into weights and biases.
    """
    logits = logits_function(pos)
    y_pred = np.argmax(logits, axis=1)
    return y_pred



# Initialize swarm
options = {'c1': 0.5, 'c2': 0.7, 'w':0.9}

# Get the weights and biases from the model
weights_biases = model.get_weights()

# Flatten the weights and biases and concatenate them into a single vector
init_pos = np.concatenate([wb.flatten() for wb in weights_biases])

# Repeat the initial position for all particles (this is optional)
init_pos = np.repeat(init_pos[None, :], 40, axis=0)

# Initialize the PSO optimizer with the initial position
optimizer = ps.single.GlobalBestPSO(n_particles=40, dimensions=init_pos.shape[1], options=options, init_pos=init_pos)

cost, pos = optimizer.optimize(f, iters=50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X = X_test
y = y_test

y_pred = predict(pos)

print((predict(pos) == y).mean())

print("Accuracy: ", accuracy_score(y, y_pred))
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))



def calculate_accuracy(pos):
    y_pred = predict(pos)
    return (y_pred == y).mean()

# Get the positions at each iteration
pos_history = optimizer.pos_history

# Calculate the accuracy for each particle at each iteration
accuracy_history = [[calculate_accuracy(pos) for pos in pos_iter] for pos_iter in pos_history]

# Calculate the best accuracy at each iteration
best_accuracy_history = np.max(accuracy_history, axis=1)

# Plot the best accuracy history
plt.plot(best_accuracy_history)
plt.title('Evolution of best accuracy along iterations')
plt.xlabel('Iterations')
plt.ylabel('Best Accuracy')
plt.show()

# Juntando os 3 graficos
g1 = best_accuracy_history
g2 = best_accuracy_history
# rna -> plt.plot(history.history['accuracy'], label='train')
plt.plot(g1)
plt.plot(g2)
plt.plot(history.history['accuracy'], label='train')


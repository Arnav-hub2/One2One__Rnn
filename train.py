import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("ecg.csv", header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# -------------------------------
# Label Encoding
# -------------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(y)

joblib.dump(encoder, "label_encoder.pkl")

# -------------------------------
# Feature Scaling
# -------------------------------
scaler = StandardScaler()

X = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# -------------------------------
# Reshape for RNN
# (140 timesteps, 1 feature)
# -------------------------------
X = X.reshape((X.shape[0], X.shape[1], 1))

# -------------------------------
# One Hot Encoding
# -------------------------------
y = to_categorical(y)

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Build RNN
# -------------------------------
model = Sequential()

model.add(
    SimpleRNN(
        64,
        input_shape=(140,1),
        activation="tanh"
    )
)

model.add(Dense(32, activation="relu"))

model.add(Dense(y.shape[1], activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# Train
# -------------------------------
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# -------------------------------
# Evaluate
# -------------------------------
loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy:", accuracy)

# -------------------------------
# Save Model
# -------------------------------
model.save("model.h5")

print("Training Complete.")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
np.random.seed(0)
meses = pd.date_range(start='1/1/2022', periods=24, freq='ME')
ventas = np.random.randint(10000, 50000, size=(24,))
data = pd.DataFrame({'Fecha': meses, 'Ventas': ventas})
scaler = MinMaxScaler()
data['Ventas_Normalizadas'] = scaler.fit_transform(data['Ventas'].values.reshape(-1,1))
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)
TIME_STEPS = 12
X, y = create_dataset(data['Ventas_Normalizadas'], data['Ventas_Normalizadas'], TIME_STEPS)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    shuffle=False
)
predicted_sales = model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), scaler.inverse_transform(y_test.reshape(-1,1)), marker='.', label="True")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), scaler.inverse_transform(predicted_sales), 'r', label="Predicted")
plt.title('Ventas Reales vs Predicciones')
plt.xlabel('Mes')
plt.ylabel('Ventas')
plt.legend()
plt.grid(True)
plt.show()
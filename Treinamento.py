import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense


# Carregar dados de preços de ações para aprendizado
data = pd.read_csv('C:\\Users\\conta\\OneDrive\\Documentos\\Projetos\\ProjectP\\Ativos_Analisar\\WINFUT.csv', sep=',')
prices = data['Close'].values.reshape(-1, 1)

print(data.columns)  # Isso imprimirá as colunas disponíveis no DataFrame
prices = data['Close'].values.reshape(-1, 1)


# Normalização dos dados
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# Divisão entre dados de treinamento e teste
train_size = int(len(prices_scaled) * 0.8)
train_data, test_data = prices_scaled[:train_size], prices_scaled[train_size:]

# Preparação dos dados para entrada na rede neural
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)

seq_length = 10  # Tamanho da sequência
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Criação do modelo RNN
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Treinamento do modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Previsão dos preços
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualização dos resultados
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Preços Reais')
plt.plot(predicted_prices, label='Previsões')
plt.legend()
plt.show()

model.save('ModelosTreinados/WINFUT.h5')#Não esqueça de renomear o arquivo com base no ativo

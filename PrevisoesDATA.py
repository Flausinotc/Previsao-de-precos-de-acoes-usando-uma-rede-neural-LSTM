import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# Carregar dados de preços de ações 
data = pd.read_csv('C:\\Users\\conta\\OneDrive\\Documentos\\Projetos\\ProjectP\\Ativos_Analisar\\WINFUT.csv', sep=',')
prices = data['Close'].values.reshape(-1, 1)
dates = pd.to_datetime(data['Date'])  # Coluna de datas

# Normalização dos dados
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# Preparação dos dados para entrada na rede neural
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 10  # Tamanho da sequência

# Carregar modelo treinado
model = keras.models.load_model('ModelosTreinados/WINFUT.h5') #Caminho onde salvou o arquivo
model.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)

# Preparar dados para previsões
latest_data = prices_scaled[-seq_length:]  # Usar os dados mais recentes para prever
latest_data = latest_data.reshape(1, seq_length, 1)  # Formato de entrada para a rede neural

# Definir o período de previsão desejado (por exemplo, 30 dias)
forecast_period = 30

# Fazer previsões sequenciais para o período desejado
predicted_prices = []
input_data = latest_data.copy()
for _ in range(forecast_period):
    prediction = model.predict(input_data)
    predicted_prices.append(prediction[0][0])
    input_data = np.roll(input_data, -1, axis=1)
    input_data[0, -1, 0] = prediction

# Remodelar os resultados das previsões para o formato correto
predicted_prices = np.array(predicted_prices).reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Criar uma sequência de datas para as previsões futuras
forecast_dates = pd.date_range(start=dates.iloc[-1], periods=forecast_period, freq='D')

# Visualização das previsões
plt.figure(figsize=(12, 6))
plt.plot(dates, data['Close'], label='Histórico de Preços')
plt.plot(forecast_dates, predicted_prices, label='Previsões Futuras')
plt.xlabel('Data')
plt.ylabel('Preço')
plt.legend()
plt.xticks(rotation=45)
plt.show()

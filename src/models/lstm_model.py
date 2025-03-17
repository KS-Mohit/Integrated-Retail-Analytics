import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

class SalesLSTM:
    def __init__(self, sequence_length, n_features):
        self.model = self._build_model(sequence_length, n_features)
        
    def _build_model(self, sequence_length, n_features):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(sequence_length, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X):
        return self.model.predict(X) 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

# Define os dados de treinamento e teste
train_texts = ['Este filme é ótimo', 'O serviço é horrível', 'Gostei muito do produto']
train_labels = [1, 0, 1]
test_texts = ['Não recomendo este produto', 'Adorei o atendimento', 'O filme foi uma decepção']
test_labels = [0, 1, 0]

# Cria o tokenizador e ajusta o tamanho do vocabulário
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(train_texts)

# Converte os textos em sequências de números e preenche com zeros ou trunca para ter o mesmo tamanho
max_len = 20
train_sequences = tokenizer.texts_to_sequences(train_texts)
train_data = pad_sequences(train_sequences, maxlen=max_len)
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_data = pad_sequences(test_sequences, maxlen=max_len)

# Converte as etiquetas em vetores binários
num_classes = 2
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Cria a rede neural
model = Sequential()
model.add(Dense(100, input_dim=max_len))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compila a rede neural
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Treina a rede neural
batch_size = 32
epochs = 100
model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_data, test_labels))

# Avalia a acurácia da rede neural
score = model.evaluate(test_data, test_labels, batch_size=batch_size)
print('Test accuracy:', score[1])

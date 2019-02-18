import time
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard

parser = argparse.ArgumentParser()

parser.add_argument("--lr", type=float, help="the learning rate", default=1e-3)
parser.add_argument("--decay", type=float, help="the learning rate decay", default=1e-5)
parser.add_argument("--epochs", type=int, help="the number of epochs", default=100)
parser.add_argument("--shuffle", type=bool, help="shuffle between train and validation data while training", default=True)

args = parser.parse_args()

#hyperparameters
_learning_rate=args.lr
_decay_rate=args.decay
_epochs=args.epochs
_shuffle=args.shuffle

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_train[0].shape) 

model = Sequential()

model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=_learning_rate, decay=_decay_rate)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)

model.fit(x_train, y_train, epochs=_epochs, validation_data=(x_test,y_test), shuffle=_shuffle, callbacks=[tensorboard])

model.save('models/model'+ str(time.ctime(int(time.time()))).replace(' ','_') + '_' + str(_epochs) +'.h5')
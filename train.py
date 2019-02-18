import time
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

parser = argparse.ArgumentParser()

parser.add_argument("--lr", type=float, help="the learning rate", default=1e-3)
parser.add_argument("--decay", type=float, help="the learning rate decay", default=1e-5)
parser.add_argument("--vs", type=float, help="proportion of validation set to be seperated from the main training set", default=0.2)
parser.add_argument("--epochs", type=int, help="the number of epochs", default=100)
parser.add_argument("--bs", type=int, help="the number of samples in one batch", default=10)
parser.add_argument("--shuffle", type=bool, help="shuffle between train and validation data while training", default=True)

args = parser.parse_args()

#hyperparameters
_learning_rate=args.lr
_decay_rate=args.decay
_epochs=args.epochs
_shuffle=args.shuffle
_batch_size=args.bs
_validation_split=args.vs

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

filepath="models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [tensorboard, checkpoint]

model.fit(x_train, y_train, validation_split=_validation_split, epochs=_epochs, batch_size=_batch_size, validation_data=(x_test,y_test), shuffle=_shuffle, callbacks=callbacks_list)

# model.save('models/model'+ str(time.ctime(int(time.time()))).replace(' ','_') + '_' + str(_epochs) +'.h5')
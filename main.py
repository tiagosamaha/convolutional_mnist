import os
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

from keract import get_activations, display_activations

from dataset import x_train, x_test, y_train, y_test

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size  = 128
epochs      = 12

layers = [
    Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
]

model = Sequential(layers)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

images = get_activations(model, x_test[:1])
display_activations(images)
print('\n####  REPORT  ####')
print('Input number is =>', y_test[0].argmax())
print('Output number is =>', model.predict(x_test[:1]).argmax())

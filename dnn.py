import parse_data as data_loader
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.optimizers import SGD
import keras
import time


def get_model(inp, n_classes):
    activation = 'relu'
    
    H = Dense(64)(inp)
    H = Activation(activation)(H)
    # H = Dense(32, activation=activation)(H)
    # H = Dense(16, activation=activation)(H)
    H = Dense(n_classes)(H)
    out = Activation('softmax')(H)

    model = Model(inp, out)
    
    opt = keras.optimizers.Adagrad()

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


folds, X_data, y_data = data_loader.load_data_kfold(10)
print(X_data.shape)


x = Input(shape=(X_data.shape[1],))
model = get_model(inp=x, n_classes=3)

for j, (train_idx, val_idx) in enumerate(folds):
    print('\nFold ', j)
    X_train_cv = X_data[train_idx]
    y_train_cv = y_data[train_idx]
    X_valid_cv = X_data[val_idx]
    y_valid_cv = y_data[val_idx]

    print("X_train: {}".format(X_train_cv))
    print("y_train: {}".format(y_train_cv))

    # time.sleep(5)

    print("X_validation: {}".format(X_valid_cv.shape))
    print("y_validation: {}".format(y_valid_cv.shape))

    x = Input(shape=(X_train_cv.shape[1],))
    model = get_model(inp=x, n_classes=3)

    model.fit(X_train_cv,
              y_train_cv,
              epochs=200,
              verbose=1,
              validation_data=(X_valid_cv, y_valid_cv)
              )

    del model

    print(x)
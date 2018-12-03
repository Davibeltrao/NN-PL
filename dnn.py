import parse_data as data_loader
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.optimizers import SGD
import keras
import time
import numpy as np
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def get_model(inp, n_classes):
    activation = 'relu'
    
    H = Dense(64)(inp)
    H = Activation(activation)(H)

    H = Dense(64)(H)
    H = Activation(activation)(H)

    H = Dense(64)(H)
    H = Activation(activation)(H)

    H = Dense(64)(H)
    H = Activation(activation)(H)

    H = Dense(64)(H)
    H = Activation(activation)(H)
    # H = Dense(32, activation=activation)(H)
    # H = Dense(16, activation=activation)(H)
    H = Dense(n_classes)(H)
    out = Activation('softmax')(H)

    model = Model(inp, out)
    
    opt = keras.optimizers.Adadelta()

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


folds, X_data, y_data = data_loader.load_data_kfold(10)
print(X_data.shape)

x = Input(shape=(X_data.shape[1],))
model = get_model(inp=x, n_classes=3)

for j, (train_idx, test_idx) in enumerate(folds):
    print('\nFold ', j)
    X_train_cv = X_data[train_idx]
    y_train_cv = y_data[train_idx]

    X_train, X_test, y_train, y_test = train_test_split(X_train_cv,
                                                        y_train_cv,
                                                        test_size=0.33,
                                                        random_state=42
                                                        )

    X_test_cv = X_data[test_idx]
    y_test_cv = y_data[test_idx]

    x = Input(shape=(X_train_cv.shape[1],))
    model = get_model(inp=x, n_classes=3)

    model.fit(X_train,
              y_train,
              epochs=80,
              verbose=1,
              validation_data=(X_test, y_test)
              )

    y_pred = model.predict(X_test_cv)
    y_pred = np.argmax(y_pred, axis=1)

    y_true = np.argmax(y_test_cv, axis=1)

    acc_fold = accuracy_score(y_true, y_pred)
    print("\n\n Accuracy: {}".format(acc_fold))
    time.sleep(2)
    del model

    print(x)

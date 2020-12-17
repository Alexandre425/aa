from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

def linear_predict (x, coefs):
    res = coefs[0]
    for i in range(len(x)):
        res = res + x[i]*coefs[i+1]
    return res

if __name__ == "__main__":
    train = [np.load("regression_data/xtrain.npy"), np.load("regression_data/ytrain.npy")]
    test = [np.load("regression_data/xtest.npy"), np.load("regression_data/ytest.npy")]

    model = keras.models.Sequential([
        keras.layers.Dense(32, input_shape=(13,), activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="relu")
    ])

    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=50, 
        restore_best_weights=True
    )

    model.compile(
        optimizer=keras.optimizers.Adam(lr = 0.001, clipnorm = 1),
        loss="mean_absolute_error",
        metrics=["mean_absolute_error", "mean_absolute_percentage_error"]
    )

    try:
        model = keras.models.load_model("neural_net.mdl")
        print("Model loaded!")
    except:
        fit = model.fit(
            train[0],
            train[1],
            validation_data=(test[0], test[1]),
            epochs=2000,
            batch_size=5,
            callbacks=[early_stop]
        )
        model.save("neural_net.mdl")
        plt.figure()
        plt.plot(fit.history['loss'], label = 'Custo de treino')
        plt.plot(fit.history['val_loss'], label = 'Custo de validação')
        plt.title('Treino da rede neuronal')
        plt.xlabel('Épocas')
        plt.ylabel('Erro absoluto médio')
        plt.legend()
        plt.show()



    print("Evaluating model")
    model.evaluate(x=test[0], y=test[1])

    print("\nLinear regression")
    # Add a 1 to the beggining of each line
    X = np.array([np.append([1], line) for line in train[0]])
    y = train[1]
    # Do a linear regression
    coefs = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    # 
    pred = [[linear_predict(test[0][i], coefs), test[1][i]] for i in range(len(test[1]))]
    # Calculating the mean absolute and mean absolute percentage errors
    err = 0
    err_pct = 0
    n = len(pred)
    for i in range(n):
        y_pred = pred[i][0]
        y_true = pred[i][1]
        abs_err = abs(y_pred - y_true)
        err = err + abs_err
        err_pct = err_pct + abs_err / y_true
    err = err / n
    err_pct = err_pct / n

    print(f"Mean abs. error: {err} Mean abs. pct. error {err_pct}")
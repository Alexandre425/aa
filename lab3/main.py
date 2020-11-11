from tensorflow.keras.datasets import fashion_mnist as fmds
from tensorflow import keras as keras
from tensorflow.keras import layers as ly
import numpy as np
from matplotlib import pyplot as plt
from visualize_activations import visualize_activations

FILE_TRAIN_X = "train_x.npy"
FILE_TRAIN_Y = "train_y.npy"
FILE_TEST_X = "test_x.npy"
FILE_TEST_Y = "test_y.npy"

if __name__ == "__main__":
    dataset = fmds.load_data()
    data_train = dataset[0]
    data_test = dataset[1]

    try:
        # Load the files if they exist
        train_x = np.load(FILE_TRAIN_X)
        train_y = np.load(FILE_TRAIN_Y)
        test_x = np.load(FILE_TEST_X)
        test_y = np.load(FILE_TEST_Y)
    except:
        print("Files don't exist, generating...")
        # Normalize the input data
        train_x = [np.expand_dims([[c / 255 for c in l] for l in img], 2) for img in data_train[0]]
        test_x  = [np.expand_dims([[c / 255 for c in l] for l in img], 2) for img in data_test[0]]
        # Convert the output data to one-hot encoding
        train_y = np.array([keras.utils.to_categorical(y, num_classes=10) for y in data_train[1]])
        test_y  = np.array([keras.utils.to_categorical(y, num_classes=10) for y in data_test[1]])
        # Save the data to files
        np.save(FILE_TRAIN_X, train_x)
        np.save(FILE_TRAIN_Y, train_y)
        np.save(FILE_TEST_X, test_x)
        np.save(FILE_TEST_Y, test_y)

    print("Data loaded")

    #######
    # MLP #
    #######

    #######
    # CNN #
    #######

    print(train_x.shape)

    ans = input("CNN: Fit (f) the model or evaluate (e)? ")
    if (ans == "f"):
        # Declare the model and all it's layers
        model = keras.models.Sequential([
            ly.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(28,28,1)),
            ly.MaxPool2D(pool_size=2),
            ly.Conv2D(filters=16, kernel_size=3, activation='relu'),
            ly.MaxPool2D(pool_size=2),
            ly.Flatten(),
            ly.Dense(units=32, activation='relu'), 
            ly.Dense(units=10, activation='softmax')
        ])
        try:    # Try loading the model
            model = keras.models.load_model("CNN")
            print("Loaded model from memory")
            
        except: # If there is no model in memory
            print("Cannot find model, compiling")
            # Compile the model with an apropriate optimizer and loss function
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['categorical_crossentropy', 'accuracy']
            )

        callbacks = [
            # Early stopping monitors validation loss, so that it stops if the model is becoming overfit
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint("CNN", monitor="loss"),
            keras.callbacks.History()
        ]
        print(model.summary())
        # Train the model
        history = model.fit(x=train_x, y=train_y, batch_size=200, epochs=200, validation_data=(test_x, test_y), callbacks=callbacks)

        # Open the loss history file
        try:
            loss_hist = np.loadtxt("cnn_history")
        except:
            loss_hist = np.empty((2,))
        # Append the new loss values and write the file
        loss_hist = np.array([np.append(loss_hist[0], history.history["loss"]), np.append(loss_hist[1], history.history["val_loss"])])
        np.savetxt("cnn_history", loss_hist)


    elif (ans == "e"):
        # Plot the history of the loss
        loss_hist = np.loadtxt("cnn_history")
        epochs = range(len(loss_hist[0]))
        plt.plot(epochs, loss_hist[0], label="Loss")
        plt.plot(epochs, loss_hist[1], label="Validation loss")
        # Load the model from a checkpoint
        model = keras.models.load_model("CNN")
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['categorical_crossentropy', 'accuracy']
        )
        # Evaluate and extract the validation loss, plotting it
        val = model.evaluate(x=test_x, y=test_y)
        plt.hlines(val[0], epochs[0], epochs[-1], label="Best val. loss")
        plt.xlabel("Epoch")
        plt.ylabel("Categ. Crossentropy")
        plt.title("Evolution of the loss throughout epochs")
        plt.legend()
        plt.show()

        # Plot the activation for an example image
        test_img = np.reshape(test_x[9], (28,28))
        visualize_activations(model, [0,2], test_img)

    else:
        print("Skipping...")
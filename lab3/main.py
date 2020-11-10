from tensorflow.keras.datasets import fashion_mnist as fmds
from tensorflow import keras as keras
from tensorflow.keras import layers as ly
import numpy as np
from matplotlib import pyplot as plt

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

    model = keras.models.Sequential([
        ly.Conv2D(filters=16, kernel_size=3)
    ])
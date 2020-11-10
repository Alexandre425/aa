from tensorflow.keras.datasets import fashion_mnist as fmds
from tensorflow import keras as keras
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    dataset = fmds.load_data()
    img = dataset[0][0][1]

    new_img = [[c / 255 for c in l] for l in img]

    plt.imshow(img)
    plt.show()
    one_hot = keras.utils.to_categorical(dataset[0][1][0], num_classes=9)
    print(one_hot)
    print(dataset[0][1][0])
    
from tensorflow.keras.datasets import fashion_mnist as fmds
import numpy as np


if __name__ == "__main__":
    dataset = fmds.load_data()
    print(dataset)
    print()
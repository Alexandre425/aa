import pandas
import numpy as np
from sklearn import naive_bayes as bayes

# Generates a matrix line from a .tsv file
def matLine(filepath):
    values = pandas.read_csv(filepath, sep='\t').values
    return [x[2] for x in values]

# Generates a training matrix from the file
def trainMatrixGenerate(languages):
    matrix = np.array([matLine(lan + "_trigram_count.tsv") for lan in languages])
    labels = np.array(languages)
    return matrix, labels

if __name__ == "__main__":
    # Generate training matrix, each line is a sample (a language)
    # Each column is a feature (a trigram)
    languages = ["en", "es", "fr", "pt"]
    matrix, labels = trainMatrixGenerate(languages)

    model = bayes.MultinomialNB()
    model.fit(matrix, labels)
    print(model.predict(matrix))
    

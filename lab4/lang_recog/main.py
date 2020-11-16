import pandas
import numpy as np
from sklearn import naive_bayes as bayes
from sklearn.feature_extraction.text import CountVectorizer

# Generates a matrix line from a .tsv file
def matLine(filepath):
    values = pandas.read_csv(filepath, sep='\t').values
    return [x[2] for x in values]

# Generates a training matrix from the file
def trainMatrixGenerate(languages):
    matrix = np.array([matLine(lan + "_trigram_count.tsv") for lan in languages])
    labels = np.array(languages)
    return matrix, labels

# Generates a look up table to translate between trigrams and indexes
def triToIndexLUT(filepath):
    values = pandas.read_csv(filepath, sep='\t').values
    lut = {}
    for l in values:
        lut[l[1]] = l[0]-1
    return lut

# Processes a phrase into matrix line format (sample)
def phraseToMatLine(lut, phrase):
    trigrams = [phrase[i:i+3] for i in range(0, len(phrase), 3)]
    [trigrams.append(phrase[i:i+3]) for i in range(1, len(phrase), 3)]
    [trigrams.append(phrase[i:i+3]) for i in range(2, len(phrase), 3)]
    line = np.zeros((1,len(lut)))
    for tri in trigrams:
        try:
            i = lut[tri]
            line[0][i] = line[0][i] + 1
        except:
            pass
    return line


if __name__ == "__main__":
    # Generate training matrix, each line is a sample (a language)
    # Each column is a feature (a trigram)
    languages = ["en", "es", "fr", "pt"]
    matrix, labels = trainMatrixGenerate(languages)
    # Train
    model = bayes.MultinomialNB()
    model.fit(matrix, labels)

    # Process the phrases and predict
    lut = triToIndexLUT("en_trigram_count.tsv")
    phrases = [
        "Que fácil es comer peras.",
        "Que fácil é comer peras.",
        "Today is a great day for sightseeing.",
        "Je vais au cinéma demain soir.",
        "Ana es inteligente y simpática.",
        "Tu vais à escola hoje."
    ]

    for p in phrases:
        line = phraseToMatLine(lut, p.lower())
        pred = model.predict(line)
        prob = sorted(model.predict_proba(line)[0])

        print(f"Phrase: \"{p}\"")
        print(f"    Label:  {pred[0]}")
        print(f"    Score:  {prob[-1]}")
        print(f"    Margin: {prob[-1] - prob[-2]}")


    

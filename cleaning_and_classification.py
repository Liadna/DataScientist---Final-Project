import pandas as pd
import nltk
import numpy as np
import re
from nltk import PorterStemmer
from nltk.corpus import stopwords
import string

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest, chi2

letters = set(string.letters)
digits = set(string.digits)
# Read csv file containing speeches as txt and there classification (obama, clinton, bush)

stop_words = set(stopwords.words("english"))

def readData(path):
    # data["speech"]
    # data["class"]
    print "Reading data..."
    data = pd.read_csv(path, header=0)

    print "Preprocessing..."
    # Clean punctuation
    data.speech = data.speech.apply(lambda x: re.sub("[^a-zA-Z0-9]", " ", x).lower())

    # Tokenize
    data.tokens = data.speech.apply(lambda x: nltk.word_tokenize(x))

    # stem
    stemmer = PorterStemmer()
    data.stemmed = data.tokens.apply(lambda x: np.array([stemmer.stem(word) for word in x]))

    # remove stop words

    data.stop_words_free = data.stemmed.apply(lambda x: np.array([w for w in x if w not in stop_words]))

    print "Creating a bag of words"
    # Create bag of words
    vocab = set()
    # Count document frequency
    df_dict = dict()

    data.dict = dict()

    for index, tokens in enumerate(data.stop_words_free):
        data.dict[index] = dict()
        for token in tokens:

            vocab.add(token)

            if token in data.dict[index]:
                data.dict[index][token] += 1
            else:
                data.dict[index][token] = 1

            # DF
            if token in df_dict:
                if index not in df_dict[token]:
                    df_dict[token].add(index)
            else:
                df_dict[token] = set()
                df_dict[token].add(index)

    features = sorted(vocab)
    features.append("classs")

    columns = []
    count = 0
    for feature in features[:-1]:
        col = []
        for key, value in data.dict.iteritems():
            if feature in value:
                col.append(value[feature])
            else:
                col.append(0)
        columns.append(col)
        count += 1

    columns.append(data["classs"])
    columns = np.array(columns)

    pd_data = pd.DataFrame(np.column_stack(columns), columns=features)

    pres = {"obama": 0, "bush": 1, "clinton": 2}
    pd_data["class_"] = pd_data["classs"].apply(lambda x: pres[x] if x in pres else 3)

    predictors = np.asarray(features)[0:-1]
    target = "class_"

    print "Prepring data for classification"
    # indices = np.random.rand(len(pd_data)) < 0.8
    x = pd_data[predictors]
    y = pd_data[target]


    # Select features
    new_data = feature_selection(x, y)

    print "Initiating Decision Tree"
    # Decision Tree
    clf = DecisionTreeClassifier(random_state=0)
    dt_scores = score_model(new_data, clf, y)

    print "Initiating Logistic Regression"
    # Logistic Regression
    lr = LogisticRegression()
    lr_scores = score_model(new_data, lr, y)

    print "Initiating Naive Bayes"
    # Naive Bayes
    nb = GaussianNB()
    nb_scores = score_model(new_data, nb, y)

    print "Plotting"
    plt.figure()
    plt.plot(top_array, [score[1] for score in dt_scores], "b", label="Decision Tree")
    plt.plot(top_array, [score[1] for score in lr_scores], "r", label="Logistic Regression")
    plt.plot(top_array, [score[1] for score in nb_scores], "g", label="Naive Bayes")

    plt.xlabel("TOP")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.title("Model comparison")

    plt.show()


top_array = [10, 20, 30, 40, 50, 60, 70, 80 ,90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def feature_selection(x, y):

    data_tops = []
    for num in top_array:
        data_tops.append(SelectKBest(chi2, k=num).fit_transform(x, y))

    # returns array containing data with different number of features selected
    return data_tops


def score_model(data, model, y):

    score_list = []
    for x in data:

        score = cross_val_score(model, x, y, cv=10)
        score_list.append((x.shape[1], score.mean()))

    return score_list


path = "D:\Programs\PythonWorkspace\DataScienceProject\PoliticalSpeeches\data_short.csv"
readData(path)





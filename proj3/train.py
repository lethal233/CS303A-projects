import argparse
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
import numpy as np

if __name__ == '__main__':
    model_file_name = 'model'
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training_data_file', type=str, default='train.json')

    args = parser.parse_args()
    training_data_file = args.training_data_file
    with open(training_data_file, 'rb') as f:
        train_data = json.loads(f.read())

    x_train = [t['data'] for t in train_data]
    y_train = [t['label'] for t in train_data]

    tv = TfidfVectorizer(max_features=50000, ngram_range=(1, 5))
    tv.fit(x_train)

    classifier = svm.LinearSVC()
    classifier.fit(tv.transform(x_train),y_train)
    joblib.dump(classifier,model_file_name)


    # Pipeline = ([('vect', TfidfVectorizer()), ('lsvc', svm.LinearSVC())])

    # parameter = {'vect__max_feature': 15000, 'vect__ngram_range': [(1, 5)], 'lsvc': }

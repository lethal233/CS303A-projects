import argparse
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

if __name__ == '__main__':
    out = 'output.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--test_data_file', type=str, default='github.json')
    parser.add_argument('-m', '--model_file', type=str, default='model')

    args = parser.parse_args()
    test_data_file = args.test_data_file
    model_file = args.model_file
    with open(test_data_file, 'rb') as f:
        test_data = json.loads(f.read())
    test_x = [t['review'] for t in test_data]
    test_y = [t['sentiment'] for t in test_data]
    # tv = TfidfVectorizer(max_features=50000, ngram_range=(1, 5))
    # tv.fit(test_x)
    clf = joblib.load(model_file)
    tv = clf.fit
    # print(clf.fit)
    cnt = 0
    re = clf.predict(tv.transform(test_x))
    for i in range(len(re)):
        print(re[i])
        if re[i] == test_y[i]:
            cnt += 1
    print(cnt / len(re))

import json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import argparse

if __name__ == '__main__':
    out = './output.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--test_data_file', type=str, default='./github.json')
    parser.add_argument('-t', '--training_data_file', type=str, default='./train.json')

    args = parser.parse_args()
    test_data_file = args.test_data_file
    training_data_file = args.training_data_file

    with open(training_data_file) as f:
        train_data = json.loads(f.read())

    # with open("./github.json") as f:
    #     test_data = json.loads(f.read())

    with open(training_data_file) as f:
        dataset = json.loads(f.read())
    data = []
    for i in dataset:
        data.append(i['data'])

    # test_path = "./github.json"
    # with open(test_path) as f:
    #     test_data2 = json.loads(f.read())

    x_train, x_test, y_train, y_test = train_test_split(
        [x['data'] for x in dataset],
        [x['label'] for x in dataset],
        random_state=5,
        test_size=0.01
    )

    # text_clf_svm = Pipeline([('vect', CountVectorizer()),
    #                          ('tfidf', TfidfTransformer()),
    #                          ('clf-svm', LogisticRegression()), ])
    clf = Pipeline([('vect', TfidfVectorizer()), ('svc', svm.SVC())])
    parameters = {
        'svc__gamma': np.logspace(-2, 1, 10),
        'svc__C': np.logspace(-1, 2, 10),
    }
    # _ = text_clf_svm.fit(X=x_train, y=y_train)
    # predicted_svm = text_clf_svm.predict_proba(x_test)
    # predicted_svm2 = text_clf_svm.predict_proba(x_train1)
    import numpy as np

    # print(np.mean(predicted_svm == y_test))
    # print(np.mean(predicted_svm2 == y_train1))

    gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=10)

    _ = gs.fit(x_train, y_train)
    print(gs.best_params_, gs.best_score_)
    print("--------")
    print(gs.score(x_test, y_test))

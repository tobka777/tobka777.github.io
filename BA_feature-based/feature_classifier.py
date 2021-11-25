import nltk, random, math
from sklearn.metrics import f1_score,classification_report,accuracy_score,matthews_corrcoef

class Classifier:
    """ feature based - NaiveBayesClassifier classifier """
    train_data = []
    dev_data = []
    test_data = []

    def __init__(self, data, get_feature_set):
        self.set_data(data)
        self.get_feature_set = get_feature_set

    def compute_prf(self, gold, predicted, class_label):
        """ calc metrics """
        if len(gold) != len(predicted) or len(gold) == 0:
            raise ValueError(
                'Sizes of gold standard and predicted value need to be equal.')
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(len(gold)):
            if gold[i] == class_label and predicted[i] == class_label:
                tp = tp + 1
            elif gold[i] != class_label and predicted[i] == class_label:
                fp = fp + 1
            elif gold[i] == class_label and predicted[i] != class_label:
                fn = fn + 1
            else:
                tn = tn + 1
        if tp + fp == 0:
            p = 0
        else:
            p = tp / (tp + fp)

        if tp + fn == 0:
            r = 0
        else:
            r = tp / (tp + fn)

        if p + r == 0:
            f = 0
        else:
            f = (2 * p * r) / (p + r)

        #acc = (tn+tp)/(tn+fp+fn+tp)

        return (
            round(p, 3),  # precision - percent of positive predictions
            round(r, 3),  # recall - percent of the positive cases
            round(f, 3)  # f-measure
        )

    def init_classifier(self, show_features):
        """ prepare features and classifier """
        train_features = [(self.get_feature_set(fileid), category)
                          for (fileid, category) in self.train_data]
        self.classifier = nltk.NaiveBayesClassifier.train(train_features)
        if show_features:
            self.classifier.show_most_informative_features(20)

    def evaluate(self, class_label):
        """ evaluate classifier """
        results = self.classifier.classify_many([
            self.get_feature_set(fileid)
            for (fileid, category) in self.test_data
        ])
        gold = [category for (fileid, category) in self.test_data]
        return self.compute_prf(gold, results, class_label)

    def f1(self):
        """ calc f1-score and classification report """
        results = self.classifier.classify_many([
            self.get_feature_set(fileid)
            for (fileid, category) in self.test_data
        ])
        gold = [category for (fileid, category) in self.test_data]
        f1score = f1_score(gold,
                           results,
                           average='weighted')
        report=classification_report(gold, results)
        return f1score, report

    def accuracy(self):
        """ calc accuracy of classifier """
        test_set = [(self.get_feature_set(fileid), category)
                    for (fileid, category) in self.test_data]
        return nltk.classify.accuracy(self.classifier, test_set)

    def classify(self, text):
        """ predict one text """
        return self.classifier.classify(self.get_feature_set(text))

    def set_data(self, data, dev=False):
        """split in training, develop and test set"""
        random.seed(1)
        random.shuffle(data)

        train_size = int(0.8 * len(data))
        train_data, test_data = data[:train_size], data[train_size:]
        if dev:
            train_size = int(0.7 * len(data))
            train_data, dev_data = data[:train_size], data[train_size:]
            self.dev_data = dev_data

        self.train_data = train_data
        self.test_data = test_data
        #return train_data, dev_data, test_data

    def get_train_data(self):
        return self.train_data

    def get_dev_data(self):
        return self.dev_data

    def get_test_data(self):
        return self.test_data

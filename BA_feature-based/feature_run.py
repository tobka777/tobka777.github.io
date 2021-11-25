import csv
import nltk
from nltk.corpus import PlaintextCorpusReader
from feature_classifier import Classifier
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import math

import sys
path = '../'
sys.path.append(path)
import data
import helper
import config

from nltk.corpus import names

top_words = []
fd_all = nltk.FreqDist()
names = names.words("male.txt") + names.words("female.txt")


def get_feature_set(text):
    """ get features from given text """
    features = {}

    txarr = text.split(" - ")

    if len(txarr) >= 2:
        features['marijuana legalization'] = (
            txarr[0] == 'marijuana legalization')
        features['school uniforms'] = (txarr[0] == 'school uniforms')
        features['nuclear energy'] = (txarr[0] == 'nuclear energy')
        features['death penalty'] = (txarr[0] == 'death penalty')
        features['minimum wage'] = (txarr[0] == 'minimum wage')
        features['gun control'] = (txarr[0] == 'gun control')
        features['abortion'] = (txarr[0] == 'abortion')
        features['cloning'] = (txarr[0] == 'cloning')
        #print(txarr[0])
        text = ' - '.join(txarr[1:])

    features['is_name'] = False
    for tok in nltk.word_tokenize(text):
        if tok in names:
            features['is_name'] = True

    features['is_question'] = '?' in text
    features['is_quote'] = '"' in text

    text = data.cleanText(text)

    tokens = [tok.lower() for tok in nltk.word_tokenize(text)]

    features['is_digit'] = False
    for tok in tokens:
        if tok.isdigit():
            features['is_digit'] = True

    #TODO: Bigram
    #bigrams = nltk.bigrams(tokens)
    #fd2 = nltk.FreqDist(bi1 + ' ' + bi2 for (bi1, bi2) in bigrams)
    #for token in bigrams:
    #    features[token] = token

    #TODO: Tuning mit haeufigst vorkommende Woerter pro Typ

    features[
        'word_hypothetical'] = 'will' in text or 'might' in text or 'in the future' in text or 'imagine' in text

    features[
        'word_philosophical'] = 'should' in text or 'must' in text or 'if' in text

    keywords = [w for w in tokens if len(w) > 5 and fd_all[w] > 10]
    for tok in keywords:
        features['key_' + tok] = tok

    fd = nltk.FreqDist(token for token in tokens if token in top_words)
    for token in fd.keys():
        features[token] = token

    return features


def train_data(threshold_topwords, detail=False):
    """ train and evaluate model with examples"""
    global top_words
    global fd_all
    #read_data = data.getDataTuple(path + config.TRAIN_DATA, topic=7)
    read_data = data.getDataTuple(path + config.TRAIN_DATA, textidx=3, labelidx=5,topic=4)

    # Prepare Classifier, train_data/dev_data/test_data
    classify = Classifier(read_data, get_feature_set)

    # Tokenize Words from Train Data
    train_data = classify.get_train_data()
    wordlist = []
    for date in train_data:
        wordlist.extend(nltk.word_tokenize(data.cleanText(date[0])))

    fd_all = nltk.FreqDist(wordlist)

    # stopwords
    stop_words = set(stopwords.words('english'))
    fd_train_words = nltk.FreqDist(w.lower() for w in wordlist
                                   if not w.lower() in stop_words)
    top_words = [t for t, _ in fd_train_words.most_common(threshold_topwords)]

    # train Classifier and evaluate
    classify.init_classifier(detail)

    if detail:
        for val in ['1', '2', '3', '4', '5', '6', '7', '8']:
            prf = classify.evaluate(val)
            print("f-measure ", prf[2], " - precision ", prf[0], " - recall ",
                  prf[1])
    acc = classify.accuracy()
    print("accuracy: ", acc)

    f1, report = classify.f1()
    print("f1-score weighted: ", f1)
    print(report)

    #Classify Test Examples
    testexamples = [
    "Apart from being boring , uniforms are highly uncomfortable as well .",
    "And you can make these statements when you understand the dynamics of the criminal justice system , when you understand how the State makes deals with more culpable defendants in a capital case , offers them light sentences in exchange for their testimony against another participant or , in some cases , in fact , gives them immunity from prosecution so that they can secure their testimony",
    "We want to encourage our children to be expressive and to think outside the box .",
    "School uniforms are a financial burden to low income families .",
    "Unfortunately there are many misguided individuals in our society .",
    "The stench of burning flesh was nauseating .",
    "The mother very much wanted the baby .",
    "Their guns did nothing to dissuade a suicidal shooter and they were unable to kill him before he harmed others .",
    "[ They ] reported that his heart was still beating , and that he was still alive .",
    "He had to lay off 5 of his 35 employees who worked at his small book store in New Mexico .",
    "We 've all seen the tragic headlines screaming of the death of a teenager who was killed for a pair of sneakers or jewelry or a designer jacket .",
    "One of the most egregious acts of enforcement occurred on August 12 , 1999 .",
    "Had no idea there was so much risks .",
    "If I had my way on weapon control , I may improve the background check system ( specifically updating information about people ) , & make it illegal for a violent felon to own weapons or knowingly transfer weapons to violent felons .",
    "Minimum wage laws have never worked in terms of helping the middle class attain more prosperity ... I do n't think a minimum wage law works .",
    "He remarked that followers of Christ , “ the staggering victim of a judicial error , ” should hesitate to execute anyone else .",
    "Milton Friedman called them a form of discrimination against low-skilled workers .",
    "“ It ’s become clear that no mass shooting , no matter how big or bloody , will inspire Republicans to put children and innocent Americans over the interests of the N.R.A. , ” he said in endorsing President Obama ’s gun proposals .",
    "He said that while he is skeptical of the 2.45 million figure , even the smaller number is compelling : 108,000 “ would represent a significant reduction in criminal activity . ”",
    "Will background checks on private transfers of guns make us safer ?",
    "Would anyone that you know support a mother killing her toddler in the name of choice and who decides ?",
    "Maybe today , but what happens in 15 years when they find a 22 year old who has not passed a background check in possession of a gun ?"
    ]
    result = [int(classify.classify(date)) for date in testexamples]
    print(result)

    return acc


train_data(30, True)

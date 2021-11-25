import csv
import random
import config
import nltk
from nltk.corpus import stopwords
import pandas as pd

"""
Helper methods for data conversions, preparing, spliting of csv and tsv files
"""

class Item:
    def __init__(self, text, label):
        self.text = text
        self.label = label


def getDataDict(file, delimiter, hashidx, textidx, labelidx):
    """ Load data from csv (idx coloum of csv) and returned a dictionary with Item (text and label) """
    resultdata = {}
    # Read Check Corpus
    with open(file, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        next(reader)
        for row in reader:
            resultdata[row[hashidx]] = Item(row[textidx], row[labelidx])
    return resultdata


def checkCSVFormat(file, delimiter, cntcol):
    """ Check each line of a csv for given length, to check an wellformed csv """
    with open(file, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        next(reader)
        for row in reader:
            if len(row) != cntcol:
                print(row)


def convertToTsv(file=config.TRAIN_DATA,
                 delimiter=';',
                 textidx=1,
                 labelidx=2,
                 skiprows=1,
                 devSet=True):
    """ Convert a given csv file with given textidx coloum, labelidx coloum into randomize train, dev (if devSet True), test tsv files
        Please defined the Output files (TRAIN_TSV, DEV_TSV, TEST_TSV) in the config.py file."""
    #import without header
    df = pd.read_csv(file, sep=delimiter, skiprows=skiprows, header=None)
    #df.sample(frac=1)
    df = df.sample(frac=1).reset_index(drop=True)
    df_data = pd.DataFrame({
        '0_id':
        range(len(df)),
        '1_label':
        df[labelidx],
        '2_alpha': ['a'] * df.shape[0],
        '3_text':
        df[textidx].replace(r'\n', ' ', regex=True)
    })
    df_data = df_data.sample(frac=1)
    if devSet:
        train_data, dev_data, test_data = splitTsvData(df_data, dev=True)
        saveInTsv(dev_data, file=config.DEV_TSV)
    else:
        train_data, test_data = splitTsvData(df_data, dev=False)
    saveInTsv(train_data, file=config.TRAIN_TSV)
    saveInTsv(test_data, file=config.TEST_TSV)


def saveInTsv(data, file):
    """ save given data in tsv file """
    data.to_csv(file, sep='\t', index=False, header=False)


def splitTsvData(df, dev=True):
    """ split dataframe into train, (dev), test """
    if dev:
        train_size = int(0.7 * len(df))
        val_size = int(0.8 * len(df))
        return df.iloc[:train_size, :], df.iloc[
            train_size:val_size, :], df.iloc[val_size:, :]
    else:
        train_size = int(0.8 * len(df))
        return df.iloc[:train_size, :], df.iloc[train_size:, :]


def getDataTuple(file=config.TRAIN_DATA,
                 delimiter=';',
                 textidx=1,
                 labelidx=2,
                 topic=-1):
    """ Load data from csv and returned a list of tuples (text, label)
        If topic is not -1 it will be added to the end of the text
    """
    resultdata = []
    # Read Check Corpus
    with open(file, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        next(reader)
        for row in reader:
            if topic != -1:
                text = row[topic] + ' - ' + row[textidx]
            else:
                text = row[textidx]
            #if row[1] == 'crowd': # for filter only crowd data
            #if row[1] == 'expert': # for filter only expert data
            resultdata.append((text, row[labelidx]))
    return resultdata


def getData(hashidx, textidx, labelidx):
    """ Load data from csv (idx coloum of csv) and returned a list of Items (text and label) """
    #for wikipedia,news getDataDict(config.TRAIN_DATA, ';', 0, 2, 3)
    #return list(getDataDict(config.TRAIN_DATA, ';', 0, 1, 2).values())
    return list(getDataDict(config.TRAIN_DATA, ';', hashidx, textidx, labelidx).values())


def getSplitData(shuffle=True, dev=False, hashidx=0, textidx=2, labelidx=1):
    """ Load data from csv (idx coloum of csv) and split """
    data = getData(hashidx, textidx, labelidx)
    return splitData(data, shuffle, dev)


def splitData(data, shuffle=True, dev=False):
    """ split data in train and test (optional in dev if dev=True), randomize if shuffle=True """
    if shuffle:
        random.seed(1)
        random.shuffle(data)

    if dev:
        train_size = int(0.7 * len(data))
        val_size = int(0.8 * len(data))
        print("Traindata Size " + str(train_size) + " of " + str(len(data)))
        return data[:train_size], data[train_size:val_size], data[val_size:]
    else:
        train_size = int(0.8 * len(data))
        print("Traindata Size " + str(train_size) + " of " + str(len(data)))
        return data[:train_size], data[train_size:]


def cleanText(text):
    """ preprocess text to get clean text without special character """
    text = text.lower()  # lowercase
    text = config.REPLACE_BY_SPACE_RE.sub(
        ' ', text
    )  # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = config.BAD_SYMBOLS_RE.sub(
        '', text
    )  # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    #text = ' '.join(word for word in text.split() if word not in set(
    #    stopwords.words('english')))  # remove stopwords from text
    return text


def getLabels():
    """ get all labels which are defined in config.py """
    return config.LABELS
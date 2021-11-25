import re

# folder
DATA_DIR = "data/"
OUTPUT_DIR = "output/"
CACHE_DIR = 'cache/'

# csv files for conversion
TRAIN_DATA = DATA_DIR + "2020-01-19_results_6000_CROWD-EXPERTS_v4.csv"

# tsv files used for training, evaluation and testing
DEV_TSV = DATA_DIR + ""
TRAIN_TSV = DATA_DIR + "argtex_6000_CE_train.tsv"
TEST_TSV = DATA_DIR + "argtex_6000_CE_test.tsv"
DEV_TSV = DATA_DIR + "argtex_6000_CE_dev.tsv"

# preprocessing for feature based
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

OUTPUT_MODE = 'classification'

# labels in data set
LABELS = ['1','2','3','4','5','6','7','8'] # argumenText


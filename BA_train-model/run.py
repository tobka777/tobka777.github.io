from pytorch_transformers import *
import data
from bert import TextClassification
import config


# example for bert
cl = TextClassification('20190924_bert_argtex', # name of new model, storage location in output/
                        'bert', # bert, roberta, xlnet
                        BertForSequenceClassification,
                        BertTokenizer,
                        'bert-large-cased', # pretrained model
                        NUM_TRAIN_EPOCHS=10)

# train model
cl.createModel()

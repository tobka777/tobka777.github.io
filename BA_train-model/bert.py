import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from pytorch_transformers import *
import pickle
from multiprocessing import Pool, cpu_count
import os
from sklearn.metrics import classification_report, accuracy_score

from inputExample import *
import inputFeatures
from tqdm import tqdm, trange
import config
import helper
import data

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


class TextClassification(object):
    """Text Classification with BERT, XLNet, RoBERTa, ...
    Based on https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04
    """
    def __init__(
            self,
            TASK_NAME,
            type,
            model_class,
            tokenizer_class,
            pretrained_weights='bert-base-cased',
            MAX_SEQ_LENGTH=128,
            TRAIN_BATCH_SIZE=24,
            EVAL_BATCH_SIZE=8,
            LEARNING_RATE=2e-5,  #1e-3
            NUM_TRAIN_EPOCHS=1,
            RANDOM_SEED=42,
            GRADIENT_ACCUMULATION_STEPS=1):
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        print(self.device)
        #torch.cuda.empty_cache()
        # The maximum total input sequence length after WordPiece tokenization.
        # Sequences longer than this will be truncated, and sequences shorter than this will be padded.
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
        self.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
        self.EVAL_BATCH_SIZE = EVAL_BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.NUM_TRAIN_EPOCHS = NUM_TRAIN_EPOCHS
        self.RANDOM_SEED = RANDOM_SEED
        self.GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS
        self.WARMUP_PROPORTION = 0.1
        self.MAX_GRAD_NORM = 1.0

        self.TASK_NAME = TASK_NAME
        self.type = type
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.pretrained_weights = pretrained_weights

        # Threshold, if the best model was found, how many models can be considered further,
        # because it is possible that even better models could be created after a bad model.
        self.thresholdSavingModels = 2

        self.outputdir = config.OUTPUT_DIR + self.TASK_NAME + '/'

        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

        self.processor = ClassificationProcessor()

    def runTokenizer(self, pretrained_model, train=True, dev=False, sent=[]):
        """ Tokenize train (train=True), dev (dev=True), test (train=False and dev=False) """
        logger.info("***** Load data *****")
        if train:
            train_examples = self.processor.get_train_examples(config.DATA_DIR)
        elif dev:
            train_examples = self.processor.get_dev_examples(config.DATA_DIR)
        elif len(sent) > 0:
            train_examples = self.processor.get_sentences(sent)
        else:
            train_examples = self.processor.get_test_examples(config.DATA_DIR)

        examples_len = len(train_examples)

        if train:
            self.num_train_optimization_steps = int(
                examples_len / self.TRAIN_BATCH_SIZE /
                self.GRADIENT_ACCUMULATION_STEPS) * self.NUM_TRAIN_EPOCHS

        label_list = self.processor.get_labels()
        self.num_labels = len(label_list)

        logger.info("***** Load pre-trained tokenizer *****")
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = self.tokenizer_class.from_pretrained(
            pretrained_model, do_lower_case=False, cache_dir=config.CACHE_DIR)

        """
        label_map = {label: i for i, label in enumerate(label_list)}
        examples_for_processing = [
            (example, label_map, self.MAX_SEQ_LENGTH, self.tokenizer,
             config.OUTPUT_MODE, self.type) for example in train_examples
        ]
        """

        features = inputFeatures.convert_examples_to_features(train_examples, label_list, self.MAX_SEQ_LENGTH, self.tokenizer, config.OUTPUT_MODE,
                cls_token_at_end=bool(self.type in ['xlnet']),            # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                cls_token_segment_id=2 if self.type in ['xlnet'] else 0,
                sep_token=self.tokenizer.sep_token,
                #sep_token_extra=bool(self.type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=False, 
                pad_on_left=bool(self.type in ['xlnet']),                 # pad on the left for xlnet
                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                #pad_token=0,
                pad_token_segment_id=4 if self.type in ['xlnet'] else 0,
                add_prefix_space=bool(self.type in ['roberta']),
                #padding=bool(self.type in ['xlnet'])
        )

        """
        logger.info("***** Train features *****")
        #process_count = cpu_count() - 1
        logger.info("Preparing to convert %s examples...", examples_len)
        #logger.info("Spawning %s processes...", process_count)
        features = []
        for ex in tqdm(examples_for_processing,
                       desc="Preparing",
                       total=examples_len):
            #features.append(inputFeatures.convert_example_to_feature(ex))
        """
        """
        with Pool(process_count) as p:
            features = list(
                tqdm(p.imap(inputFeatures.convert_example_to_feature,
                            examples_for_processing),
                     total=examples_len))
        """

        self.train_examples_len = examples_len
        return features

    def prepareModel(self, pretrained_model, runOptimizer=True):
        """ load pretrained model, run optimizer and scheduler """
        logger.info("***** Load pre-trained model *****")
        self.model = self.model_class.from_pretrained(
            pretrained_model,
            cache_dir=config.CACHE_DIR,
            num_labels=self.num_labels)

        self.model.to(self.device)

        if runOptimizer:
            logger.info("***** Running optimizer *****")
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [{
                'params': [
                    p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.01
            }, {
                'params': [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.0
            }]
            self.optimizer = AdamW(optimizer_grouped_parameters,
                                   lr=self.LEARNING_RATE,
                                   correct_bias=False)
            logger.info("***** Running scheduler *****")
            self.scheduler = WarmupLinearSchedule(
                self.optimizer,
                warmup_steps=self.WARMUP_PROPORTION,
                t_total=self.num_train_optimization_steps)

    def prepareTraining(self, train_features, train=True):
        """ prepare data for training """
        all_input_ids = torch.tensor([f.input_ids for f in train_features],
                                     dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features],
                                      dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features],
                                       dtype=torch.long)

        if config.OUTPUT_MODE == "classification":
            self.all_label_ids = torch.tensor(
                [f.label_id for f in train_features], dtype=torch.long)
        elif config.OUTPUT_MODE == "regression":
            self.all_label_ids = torch.tensor(
                [f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask,
                                   all_segment_ids, self.all_label_ids)
        if train:
            batch_size = self.TRAIN_BATCH_SIZE
            train_sampler = RandomSampler(train_data)
        else:
            batch_size = self.EVAL_BATCH_SIZE
            train_sampler = SequentialSampler(train_data)

        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=batch_size)
        return train_dataloader

    def runTraining(self, train_dataloader):
        """ train model with given data """
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        epochnr = 0
        savedModel = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.train_examples_len)
        logger.info("  Batch size = %d", self.TRAIN_BATCH_SIZE)
        logger.info("  Num steps = %d", self.num_train_optimization_steps)

        self.model.train()
        loss_arr = []
        dev_accurancy = []
        for _ in trange(int(self.NUM_TRAIN_EPOCHS), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(
                    tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                #logits = model(input_ids, segment_ids, input_mask, labels=None)
                loss, logits = self.model(input_ids,
                                          segment_ids,
                                          input_mask,
                                          labels=label_ids)[:2]

                #if config.OUTPUT_MODE == "classification":
                #    loss_fct = CrossEntropyLoss()
                #    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                #elif config.OUTPUT_MODE == "regression":
                #    loss_fct = MSELoss()
                #    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if self.GRADIENT_ACCUMULATION_STEPS > 1:
                    loss = loss / self.GRADIENT_ACCUMULATION_STEPS

                loss.backward()
                print("\r%f" % loss, end='')
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.MAX_GRAD_NORM
                )  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.optimizer.zero_grad()
                    global_step += 1
            self.saveModel(epochnr)
            """
            if epochnr == 0:
                #self.saveModel()
                self.saveModel(epochnr)
            else:
                acc = self.accDevModel(epochnr-1)
                self.saveModel(epochnr)
                print(acc)
                if all(i > tr_loss for i in loss_arr) and all(i < acc for i in dev_accurancy):
                    print("Saving Model Epoch ", epochnr)
                    #self.saveModel()
                    savedModel = 0
                else:
                    savedModel += 1
                    print("Model not saved")
                    #if savedModel >= self.thresholdSavingModels:
                    #    loss_arr.append(tr_loss)
                    #    dev_accurancy.append(acc)
                    #    break
                dev_accurancy.append(acc)
            """
            if all(i > tr_loss for i in loss_arr):
                print("Saving Model Epoch ", epochnr)
                self.saveModel()
                savedModel = 0
            else:
                savedModel += 1
                print("Model not saved")
                if savedModel >= self.thresholdSavingModels:
                    loss_arr.append(tr_loss)
                    break
            epochnr += 1
            loss_arr.append(tr_loss)
            print(tr_loss)

        print(loss_arr)
        #print(dev_accurancy)
        # save epoches loss in epoches.txt
        output_eval_file = os.path.join(self.outputdir, "epoches.txt")
        with open(output_eval_file, "w") as writer:
            writer.write("%s Iterationen\n" % (str(len(train_dataloader))))
            writer.write("%s Epochen\n" % (str(self.NUM_TRAIN_EPOCHS)))
            z = 1
            for lo in loss_arr:
                writer.write("%s: %s\n" % (str(z), str(round(lo, 2))))
                z = z + 1
            writer.write("\n")

    def runEval(self, eval_dataloader):
        """ evalutate model with given data """
        self.model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(
                eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)

            with torch.no_grad():
                #logits = model(input_ids, segment_ids, input_mask, labels=None)
                tmp_eval_loss, logits = self.model(input_ids,
                                                   segment_ids,
                                                   input_mask,
                                                   labels=label_ids)[:2]

            # create eval loss and other metric required by the task
            #if OUTPUT_MODE == "classification":
            #    loss_fct = CrossEntropyLoss()
            #    tmp_eval_loss = loss_fct(logits.view(-1, num_labels),
            #                             label_ids.view(-1))
            #elif OUTPUT_MODE == "regression":
            #    loss_fct = MSELoss()
            #    tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0],
                                     logits.detach().cpu().numpy(),
                                     axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if config.OUTPUT_MODE == "classification":
            preds = np.argmax(preds, axis=1)
        elif config.OUTPUT_MODE == "regression":
            preds = np.squeeze(preds)

        return preds, eval_loss

    def computeMetrics(self, preds, eval_loss):
        """ calc metrics confusion matrix, accuracy, ... """ 
        result = helper.compute_metrics(self.TASK_NAME,
                                        self.all_label_ids.numpy(), preds)
        report = classification_report(self.all_label_ids.numpy(),
                                       preds,
                                       target_names=self.processor.get_labels())

        result['eval_loss'] = eval_loss

        output_eval_file = os.path.join(self.outputdir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in (result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            logger.info("%s", report)
            writer.write("%s\n" % (report))

    def saveModel(self, epoch=-1):
        """ save trained model """
        logging.info("***** Saving fine-tuned model *****")
        model_to_save = self.model.module if hasattr(
            self.model, 'module'
        ) else self.model  # Take care of distributed/parallel training

        output_dir = self.getOutputDir(epoch)
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def createModel(self):
        """ global function to train model """
        train_features = self.runTokenizer(self.pretrained_weights, train=True)
        self.prepareModel(self.pretrained_weights)
        train_dataloader = self.prepareTraining(train_features, train=True)
        self.runTraining(train_dataloader)
        #self.saveModel()
        return self.model

    def evalutateModel(self, sent=[]):
        """ global function to evaluate model """
        eval_features = self.runTokenizer(self.outputdir, train=False, sent=sent)
        eval_dataloader = self.prepareTraining(eval_features, train=False)
        self.prepareModel(self.outputdir, runOptimizer=False)
        preds, eval_loss = self.runEval(eval_dataloader)
        print(preds)
        self.computeMetrics(preds, eval_loss)

    def getOutputDir(self, epochnr=-1):
        """ helper function to get the output_dir considering tmp folder """
        if epochnr != -1:
            output_dir = self.outputdir+'tmp'+str(epochnr)+'/'
        else:
            output_dir = self.outputdir
        
        print(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return output_dir

    def accDevModel(self, epoch=-1):
        """ calc accuracy of dev-data """
        output_dir = self.getOutputDir(epoch)
        eval_features = self.runTokenizer(output_dir, train=False, dev=True)
        eval_dataloader = self.prepareTraining(eval_features, train=False)
        self.prepareModel(output_dir, runOptimizer=False)
        preds, eval_loss = self.runEval(eval_dataloader)
        return accuracy_score(self.all_label_ids.numpy(), preds)

    def evaluatePretrainedModel(self):
        """ global function to evaluate pretrained (without finetuning) model """
        eval_features = self.runTokenizer(self.pretrained_weights, train=False)
        eval_dataloader = self.prepareTraining(eval_features, train=False)
        self.prepareModel(self.pretrained_weights, runOptimizer=False)
        preds, eval_loss = self.runEval(eval_dataloader)
        self.computeMetrics(preds, eval_loss)

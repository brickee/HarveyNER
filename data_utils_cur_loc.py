import os
import sys
import logging
import random
import numpy as np
from scipy.stats import rankdata
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, tags = None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.tags = tags

def scale(array):
    array = np.array(array)
    return (array-np.min(array)) / (np.max(array) - np.min(array))


def number_of_entity(labels):
    num = 0
    for label in labels:
        if label[0] == 'B':
            num += 1
    if num == 0:
        return 10
    return num

def average_entity_length(text, labels):
    num_of_entity = []
    for word, label in zip(text, labels):
        if label[0] == 'B':
            num_of_entity.append(len(word))
        elif label[0] == 'I':
            num_of_entity[-1] += len(word)
    if len(num_of_entity) == 0:
        return 60
    return np.mean(num_of_entity)

def average_entity_word_length(labels):
    num_of_entity_word = []
    for label in labels:
        if label[0] == 'B':
            num_of_entity_word.append(1)
        elif label[0] == 'I':
            num_of_entity_word[-1] += 1
    if num_of_entity_word == []:
        return 20
    return np.mean(num_of_entity_word) 

def normalization(matrix):
    mean = matrix.mean(axis=0)
    cov = np.cov(matrix.T)
    v, s, u = np.linalg.svd(cov)
    inv_std = np.linalg.pinv(u * (s ** 0.5))
    return (matrix - mean).dot(inv_std)

def maximum_entity_length(labels):
    best = 0
    current = 0
    for label in labels:
        if label == 'O':
            best = max(current, best)
            current = 0
        else:
            current += 1
    best = max(current, best)
    return best


def generate_complexity(words, labels):
    clues = ['and','&', 'at', '@', 'in', 'on', 'near', 'between', 'of']
    easy = ['of']
    medium = ['at', '@', 'in', 'on', 'near', 'between']
    hard = ['and','&']
    complexity = 0
    for word, label in zip(words, labels):
        if word.lower() in easy and label != 'O':
            complexity = max(complexity, 1)
        if word.lower() in medium and label != 'O':
            complexity = max(complexity, 2)
        if word.lower() in hard and label != 'O':
            complexity = max(complexity, 3)
    return complexity 

def generate_cumulative_complexity(words, labels):
    clues = ['and','&', 'at', '@', 'in', 'on', 'near', 'between', 'of']
    easy = ['of']
    medium = ['at', '@', 'in', 'on', 'near', 'between']
    hard = ['and','&']
    complexity = 0
    for word, label in zip(words, labels):
        if word.lower() in easy and label != 'O':
            complexity += 1
        elif word.lower() in medium and label != 'O':
            complexity += 2
        elif word.lower() in hard and label != 'O':
            complexity += 3
    return complexity


def generate_density_difficulty_score(complexity, oov, word_average):
    complexity = np.array(complexity) / np.max(complexity)
    oov = np.array(oov) / np.max(oov)
    word_average = np.array(word_average) / np.max(word_average)
    vectors = []
    density = []

    vectors = np.stack((complexity, word_average, oov)).T
    mean = vectors.mean(axis=0)
    precision = np.linalg.inv(np.cov(vectors.T))
    vectors = (vectors - mean).dot(precision)
    for v in vectors.tolist():
        z = 0
        for u in vectors:
            z += np.linalg.norm(v-u)
        density.append(z + np.linalg.norm(v))
    return density


def generate_norm_difficulty_score(length, complexity, average, oov, cumulative, maximum, ratio, number, weights):
    length = scale(length)
    complexity = scale(complexity)
    average = scale(average)
    oov = scale(oov)
    cumulative = scale(cumulative)
    maximum = scale(maximum)
    ratio = scale(ratio)
    number = scale(number)
    vectors = np.stack((length, complexity, average, oov, cumulative, maximum, ratio, number)).T
    mean = vectors.mean(axis=0)
    std = np.std(vectors, axis=0)
    vectors = (vectors - mean)/std
    vectors = vectors * weights
    norm = []
    for v in vectors:
        norm.append(np.linalg.norm(v))
    return norm

def generate_norm2_difficulty_score(length, complexity, average, oov, cumulative, maximum, ratio, number, weights):
    length = scale(length)
    complexity = scale(complexity)
    average = scale(average)
    oov = scale(oov)
    cumulative = scale(cumulative)
    maximum = scale(maximum)
    ratio = scale(ratio)
    number = scale(number)
    vectors = np.stack((length, complexity, average, oov, cumulative, maximum, ratio, number)).T
    mean = vectors.mean(axis=0)
    std = np.std(vectors, axis=0)
    vectors = (vectors - mean)/std
    vectors = vectors * weights
    vectors = vectors + abs(np.min(vectors, axis=0))
    norm = []
    for v in vectors:
        norm.append(np.linalg.norm(v))
    return norm

def generate_norm3_difficulty_score(length, complexity, average, oov, cumulative, maximum, ratio, number, weights):
    length = scale(length)
    complexity = scale(complexity)
    average = scale(average)
    oov = scale(oov)
    cumulative = scale(cumulative)
    maximum = scale(maximum)
    ratio = scale(ratio)
    number = scale(number)
    vectors = np.stack((length, complexity, average, oov, cumulative, maximum, ratio, number)).T
    vectors = normalization(vectors)
    vectors = vectors * weights
    norm = []
    for v in vectors:
        norm.append(np.linalg.norm(v))
    return norm


def generate_rank_score(length, complexity, average, oov, cumulative, maximum, ratio, number, weights):
    length = scale(length)
    complexity = scale(complexity)
    average = scale(average)
    oov = scale(oov)
    cumulative = scale(cumulative)
    maximum = scale(maximum)
    ratio = scale(ratio)
    number = scale(number)
    vectors = np.stack((length, complexity, average, oov, cumulative, maximum, ratio, number)).T
    vectors = vectors * weights 
    rank = rankdata(vectors, axis=0, method='average').mean(axis=1)
    return rank


def generate_rank2_score(length, complexity, average, oov, cumulative, maximum, ratio, number, weights):
    length = scale(length)
    complexity = scale(complexity)
    average = scale(average)
    oov = scale(oov)
    cumulative = scale(cumulative)
    maximum = scale(maximum)
    ratio = scale(ratio)
    number = scale(number)
    vectors = np.stack((length, complexity, average, oov, cumulative, maximum, ratio, number)).T
    vectors = vectors * weights
    rank = np.max(rankdata(vectors, axis=0, method='average'),axis=1)
    return rank




class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None, pos=None, dep = None, head = None, adj_a=None, adj_f=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask






def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.strip().split(' ')
        sentence.append(splits[0])
        label.append(splits[1:])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "tweets.train10.bio")), "train")
        #
        # return self._read_tsv(os.path.join(data_dir, "tweets.train8.bio"))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "tweets.dev10.bio")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "tweets.test10.bio")), "test")

    def get_labels(self, data_dir): # last one has to be 'SEP' ！！！！！
        # TODO: check if O should be first!
        # return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
        # print(self._read_tsv(os.path.join(data_dir, "labels.txt"))[0][0])
        label_list = ['O'] # make O to be in the first place
        label_list.extend([i.strip() for i in self._read_tsv(os.path.join(data_dir, "labels.txt"))[0][0][:-1]])
        label_list.extend(["[CLS]", "[SEP]"])
        # print(label_list)
        return label_list

    def _create_examples(self,lines,set_type):
        examples = []
        # print(lines)
        for i,(sentence,labels) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = [lbl[0] for lbl in labels]
            tags = {}
            # print(labels, label, tags)
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label, tags= tags))
        return examples

def get_statistics(examples):
    number_of_labeled_samples = 0
    number_of_adverbial = 0
    number_of_labeled_adverbial = 0
    for sample in examples:
        sentence = sample.text_a.split(' ')
        labels = sample.label
        if 'at' in sentence or 'from' in sentence or 'near' in sentence or 'on' in sentence or 'between' in sentence or 'in' in sentence:
            number_of_adverbial += 1
            if labels.count('O') != len(labels):
                number_of_labeled_adverbial += 1
                number_of_labeled_samples += 1
        elif labels.count('O') != len(labels):
            number_of_labeled_samples += 1
    print("The number of samples ", len(examples), " the number of labeled samples ", number_of_labeled_samples)
    print("The number of adverbial ", number_of_adverbial, " the number of labeled adverbial ", number_of_labeled_adverbial)
    return number_of_labeled_samples / len(examples), number_of_labeled_adverbial / number_of_adverbial


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            elif embedd_dim + 1 != len(tokens):
                ## ignore illegal embedding line
                continue
                # assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, training=False, curriculum=None, neutral=False, diversity=False, ordered=False, word_emb_dir=None, weights=None):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label : i for i, label in enumerate(label_list,1)}
    # print(label_map)
    if curriculum in ['rank2', 'rank', 'norm3', 'norm2', 'length-norm', 'vocabulary', 'oov', 'density', 'norm'] and word_emb_dir != None:
        embedd_dict, embedd_dim = load_pretrain_emb(word_emb_dir)
        out_vocabulary = []
    features = []
    easy_features, hard_features = [], []
    easy_metric, hard_metric = [], []
    l, s = [], []
    length = []
    frequency = []
    num_of_label = []
    entity_length = []
    word_level_average = []
    max_entity_length = []
    cumulative_complexity = []
    labeled_features = []
    unlabeled_features = []
    labeled_metric = []
    complexity = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            #TODO:单个tokenize word的结果可能和整个tokenize sentence不一样！
            # input_id 基于tokens(ntokens)建立。 因为有valid mask，所以对BERT本身没有影响。
            # 但是图是在只保留了valid id上的构建的，所以要回归原句子的id
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]

            if not len(token): # deal with special token
                tokens.append('?')
                labels.append(label_1)
                valid.append(1)
                label_mask.append(1)
            else:
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                        valid.append(1)
                        label_mask.append(1)
                    else:
                        valid.append(0)


        assert labels == labellist # if not, try to modify tag processing


        # print(pos,dep,head)
        if len(tokens) >= max_seq_length - 1:  # TODO: only remove longer part!
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]


        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)  # label mask take the SEP and CLS into account
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])  # label_ids and label_mask include SEP and CLS; but not label.
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)


        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length
        # tag process

        assert len(label_ids)==max_seq_length




        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("valid_ids: %s" % " ".join([str(x) for x in valid]))

            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, " ".join([str(x) for x in label_ids])))
            logger.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))



        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              ))
        if 'at' in textlist or 'in' in textlist or 'on' in textlist or 'from' in textlist or 'near' in textlist or 'between' in textlist:
            easy_features.append(features[-1])
            easy_metric.append(len(textlist))
        else:
            hard_features.append(features[-1])
            hard_metric.append(len(textlist))
        times = labellist.count('O') / len(labellist)
        if not neutral:
            length.append(len(textlist))
            frequency.append(times)
            num_of_label.append(number_of_entity(labellist))
            entity_length.append(average_entity_length(textlist,labellist))
            word_level_average.append(average_entity_word_length(labellist))
            if curriculum in ['rank2', 'rank', 'norm3', 'norm2', 'length-norm', 'vocabulary', 'oov', 'density', 'norm']:
                out_vocab = 0
                for word in textlist:
                    if word in embedd_dict or word.lower() in embedd_dict:
                        out_vocab += 1
                out_vocabulary.append(out_vocab / len(textlist))
        else:
            if times < 1:
                length.append(len(textlist))
                frequency.append(1-times)
                num_of_label.append(number_of_entity(labellist))
                entity_length.append(average_entity_length(textlist,labellist))
                word_level_average.append(average_entity_word_length(labellist))
                complexity.append(generate_complexity(textlist,labellist))
                max_entity_length.append(maximum_entity_length(labellist))
                cumulative_complexity.append(generate_cumulative_complexity(textlist, labellist))
                if curriculum in ['rank2', 'rank', 'norm3', 'norm2', 'length-norm', 'vocabulary', 'oov', 'density', 'norm']:
                    out_vocab = 0
                    for word in textlist:
                        if word in embedd_dict or word.lower() in embedd_dict:
                            out_vocab += 1
                    out_vocabulary.append(out_vocab / len(textlist))
                labeled_features.append(features[-1])
            else:
                #frequency.append(np.mean(frequency))
                #num_of_label.append(np.mean(num_of_label))
                #entity_length.append(np.mean(entity_length))
                #word_level_average.append(np.mean(word_level_average))
                unlabeled_features.append(features[-1])    
    if training:    
        #return s + l, sorted(np.array(length)), len(s) / (len(s) + len(l))
        
        features = np.array(features)
        length = np.array(length)
        frequency = np.array(frequency)
        num_of_label = np.array(num_of_label)
        entity_length = np.array(entity_length)
        #difficulty = frequency + 0.75 * num_of_label + 0.5 * length + 0.25 * entity_length
        if curriculum == "length":
            labeled_metric = length
            #order_features = features[inds].tolist()
            #return order_features, sorted(length)
        elif curriculum == "frequency" or curriculum == 'ratio':   
            #inds = frequency.argsort()
            #order_features = features[inds].tolist()
            #return order_features, sorted(frequency)
            labeled_metric = frequency
        elif curriculum == 'average':
            #inds = entity_length.argsort()
            #order_features = features[inds].tolist()
            #return order_features, sorted(entity_length) 
            labeled_metric = entity_length
        elif curriculum == 'number':
            #inds = num_of_label.argsort()
            #order_features = features[inds].tolist()
            #return order_features, sorted(num_of_label)
            labeled_metric = num_of_label
        elif curriculum == 'average-word':
            #inds = np.argsort(word_level_average)
            #order_features = features[inds]
            #return order_features, sorted(word_level_average)
            labeled_metric = word_level_average
        elif curriculum == 'complex':
            labeled_metric = complexity
        elif curriculum == 'vocabulary' or curriculum == 'oov':
            labeled_metric = out_vocabulary
        elif curriculum == 'density':
            labeled_metric = generate_density_difficulty_score(complexity, out_vocabulary, word_level_average)
        elif curriculum == 'norm':
            labeled_metric = generate_norm_difficulty_score(length, complexity, word_level_average, out_vocabulary, cumulative_complexity, max_entity_length, frequency, num_of_label, weights)
        elif curriculum == 'norm2':
            labeled_metric = generate_norm2_difficulty_score(length, complexity, word_level_average, out_vocabulary, cumulative_complexity, max_entity_length, frequency, num_of_label, weights)
        elif curriculum == 'norm3':
            labeled_metric = generate_norm3_difficulty_score(length, complexity, word_level_average, out_vocabulary, cumulative_complexity, max_entity_length, frequency, num_of_label, weights)        
        elif curriculum == 'rank':
            labeled_metric = generate_rank_score(length, complexity, word_level_average, out_vocabulary, cumulative_complexity, max_entity_length, frequency, num_of_label, weights)
        elif curriculum == 'rank2':
            labeled_metric = generate_rank2_score(length, complexity, word_level_average, out_vocabulary, cumulative_complexity, max_entity_length, frequency, num_of_label, weights)
        elif curriculum == 'length-norm':
            labeled_metric = generate_length_norm_difficulty_score(length, complexity, out_vocabulary, word_level_average, weights) 
        elif curriculum == 'adverbial-length':
            easy_metric, hard_metric = np.array(easy_metric), np.array(hard_metric)
            easy_features, hard_features = np.array(easy_features), np.array(hard_features)
            easy_inds, hard_inds = np.argsort(easy_metric), np.argsort(hard_metric)
            easy_features, hard_features = easy_features[easy_inds].tolist(), hard_features[hard_inds].tolist()
            hard_metric += np.max(easy_metric)
            return easy_features + hard_features, sorted(np.concatenate((easy_metric, hard_metric), axis = None))
        elif curriculum == 'length-adverbial':
            easy_metric, hard_metric = np.array(easy_metric), np.array(hard_metric)
            easy_features, hard_features = np.array(easy_features), np.array(hard_features)
            easy_inds, hard_inds = np.argsort(easy_metric), np.argsort(hard_metric)
            easy_features, hard_features = easy_features[easy_inds].tolist(), hard_features[hard_inds].tolist()
            easy_metric, hard_metric = sorted(easy_metric.tolist()), sorted(hard_metric.tolist())
            features, difficulty_score = [], []
            while easy_metric and hard_metric:
                if easy_metric[0] <= hard_metric[0]:
                    features.append(easy_features.pop(0))
                    difficulty_score.append(easy_metric.pop(0))
                else:
                    features.append(hard_features.pop(0))
                    difficulty_score.append(hard_metric.pop(0))
            features += easy_features + hard_features
            difficulty_score += easy_metric + hard_metric
            return features, np.array(difficulty_score)
        else:
            return features, []
        if neutral:
            labeled_features, labeled_metric = np.array(labeled_features), np.array(labeled_metric)
            inds = np.argsort(labeled_metric)
            labeled_features = labeled_features[inds].tolist()
            features = []
            i = 0
            while labeled_features and unlabeled_features:
                rand = random.random()
                if rand <= 0.3:
                    rand = random.random()
                    if rand >= 0.2 or ordered:
                        i += 1
                        features.append(labeled_features.pop(0))
                    else:
                        idx = random.randrange(len(labeled_features))
                        features.append(labeled_features.pop(idx))
                else:
                    idx = random.randrange(len(unlabeled_features))
                    features.append(unlabeled_features.pop(idx))
            while labeled_features:
                rand = random.random()
                if rand >= 0.2 or ordered:
                    i += 1
                    features.append(labeled_features.pop(0))
                else:
                    idx = random.randrange(len(labeled_features))
                    features.append(labeled_features.pop(idx))
            while unlabeled_features:
                idx = random.randrange(len(unlabeled_features))
                features.append(unlabeled_features.pop(idx))
            print('-------------------------------')
            print('Unbalanced:', i == len(labeled_metric))
            print(labeled_metric.tolist().count(1))
            print('-------------------------------')
            return features, []
        else:
            inds = np.argsort(labeled_metric)
            features = features[inds].tolist()
            return features, sorted(labeled_metric)
    else:
        return features



def write2file(examples,y_true , y_pred,file_name):
    with open(file_name, "w") as writer:
        for i,y_sen in enumerate(y_true):
            eg = examples[i].text_a.split(' ')
            for j,lbl in enumerate(y_sen):
                line = ' '.join([eg[j], lbl, y_pred[i][j]])
                writer.write(line)
                writer.write('\n')
            writer.write('\n')

def write2report(output_test_file, report):
    with open(output_test_file, "w") as writer:
        writer.write(report)



def subtokens2tokens(tokens):
    def is_subtoken(word):
        if word[:2] == "##":
            return True
        else:
            return False
    # tokens = ['why', 'isn', "##'", '##t', 'Alex', "##'", 'text', 'token', '##izing']
    restored_text = []
    for i in range(len(tokens)):
        if not is_subtoken(tokens[i]) and (i + 1) < len(tokens) and is_subtoken(tokens[i + 1]):
            restored_text.append(tokens[i] + tokens[i + 1][2:])
            if (i + 2) < len(tokens) and is_subtoken(tokens[i + 2]):
                restored_text[-1] = restored_text[-1] + tokens[i + 2][2:]
        elif not is_subtoken(tokens[i]):
            restored_text.append(tokens[i])
    return restored_text

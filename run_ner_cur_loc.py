from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys
import datetime
from torch.utils.data.sampler import Sampler
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import torch
import torch.nn.functional as F
from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                                  BertForTokenClassification, BertTokenizer,
                                  WarmupLinearSchedule,BertModel,BertPreTrainedModel)

# TODO: bugs in the neweset verstion.
# from transformers import (WEIGHTS_NAME, AdamW, BertConfig,
#                                   BertForTokenClassification, BertTokenizer,
#                                   get_linear_schedule_with_warmup,BertModel,BertPreTrainedModel)

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from seqeval.metrics import classification_report,f1_score


from data_utils_cur_loc import NerProcessor,convert_examples_to_features,write2file,write2report
from model_loc import Ner




# set up log file
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
logger = logging.getLogger(__name__)




# set up seed
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.deterministic = True

class CurriculumSampler(Sampler):
    def __init__(self, dataset, difficulty_score=None, epoch = 80, competence=0.5):
        self.dataset = dataset
        self.init_competence = competence
        self.competence = competence
        self.epoch = epoch
        self.difficulty_score = difficulty_score

    def update_competence(self, t):
        square = self.init_competence ** 2
        root = np.sqrt(t * (1 - square) / self.epoch + square)
        self.competence = min(1, root)

    def __iter__(self):
        i = 0
        if self.difficulty_score is not None:
            while i in range(len(self.difficulty_score)):
                if self.difficulty_score[i] > self.competence:
                    break
                i += 1
        else:
            i = int(self.competence * len(self.dataset))
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        yield from torch.randperm(i, generator=generator).tolist()





def prepare_data(features):


    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                             all_lmask_ids)


    return data


def evaluate(eval_dataloader,model,label_map,args,tokenizer,device):
    y_true = []
    y_pred = []


    for batch in tqdm(eval_dataloader,desc="Evaluating"):

        batch = tuple(t.to(device) for t in batch)

        input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch


        with torch.no_grad():
            if args.use_crf:
                logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids,
                                     attention_mask_label=l_mask)

            else:
                logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids,
                           attention_mask_label=l_mask)
                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                # print(logits)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        # print(logits.shape, logits)

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_map):  # end of sentence
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    try:
                        if label_map[logits[i][j]] not in ["[CLS]", "[SEP]"]:
                            temp_2.append(label_map[logits[i][j]])
                        else:
                            temp_2.append('O')
                    except:
                        #the longest sentence lose some, predict as 'O', this is from the data process and cannot be avoided when valid/label masks both exist
                        temp_2.append('O')
    return y_true,y_pred





def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    # parser.add_argument("--output_dir",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument("--eval_on",
                        default="dev",
                        help="Whether to run eval on the dev set or test set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to predict on test set or not.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--use_rnn",
                        action='store_true',
                        help="Whether to use rnn.")
    parser.add_argument("--use_crf",
                        action='store_true',
                        help="Whether to use crf.")

    # parser.add_argument("--relearn_embed",
    #                     action='store_true',
    #                     help='Whether to relearn embeddings.')

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--bert_lr",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--curriculum', type=str, default='', help="Determine difficulty score for curriculum learning: length, frequency, number, average")
    args = parser.parse_args()
    output_dir = '_'.join(['./saver/',args.data_dir.split('/')[-1], args.bert_model, args.curriculum, str(args.max_seq_length), str(args.learning_rate), str(args.bert_lr), str(args.warmup_proportion),str(args.train_batch_size),str(int(args.num_train_epochs)), str(args.seed) ])
    if args.use_crf:
        output_dir+='_crf'
    if args.use_rnn:
        output_dir += '_rnn'
    if args.do_lower_case:
        output_dir += '_lower'
    
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
     
    if os.path.exists(output_dir) and os.listdir(output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fh = logging.FileHandler(output_dir+'/logging.log', mode="w", encoding="utf-8")
    logger.addHandler(fh)




    processors = {"ner":NerProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps


    seed_torch(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")



    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list) + 1 # why plus 1? because of the pad id 0
    label_map = {i : label for i, label in enumerate(label_list,1)}   # start from 1!  0 is keeped!
    tag2ix = {label: i for i, label in enumerate(label_list, 1)}

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    #print(train_examples[0].label)
    #exit()
    # Prepare model
    config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
    model = Ner(args.bert_model,from_tf = False, config = config,tag_to_ix=tag2ix, device=device, use_rnn=args.use_rnn, use_crf=args.use_crf)


    #logger.info("Mode: %s", str(model))
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if ((not any(nd in n for nd in no_decay)) and  ('bert' in n )) ],'lr': args.bert_lr, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if ((not any(nd in n for nd in no_decay)) and  ( not 'bert' in n )) ],'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if ((any(nd in n for nd in no_decay)) and ('bert' in n))],'lr': args.bert_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if ((any(nd in n for nd in no_decay)) and (not 'bert' in n))] ,'lr': args.learning_rate, 'weight_decay': 0.0}

        ]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    # text = 'murine bone marrow contains endothelial progenitors'
    # tokens = tokenizer.tokenize(text)

    
  
    if args.do_train:
        train_features, difficulty_score = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, True, curriculum=args.curriculum)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        train_data = prepare_data(train_features)
        #print(difficulty_score)
        #difficulty_score = None
        difficulty_score = np.array(difficulty_score) / max(difficulty_score)
        #print(difficulty_score)
        #exit()
        '''
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        '''
        train_sampler = CurriculumSampler(train_data, difficulty_score = difficulty_score, competence=np.quantile(difficulty_score, 0.3))
        #TODO: why not use shuffle?
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # Dev data
        if args.do_eval:
            logger.info("***** evaluation data process*****")
            eval_examples = processor.get_dev_examples(args.data_dir)
            eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)

            eval_data =  prepare_data(eval_features)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)


        max_eval_f1 = -1
        for epoch in range(int(args.num_train_epochs)):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                
                input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
                #print(input_mask.shape)
                loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if nb_tr_steps %20 ==0:
                    logger.info('loss[%d,%d]: %f' %(epoch+1,nb_tr_steps,tr_loss/nb_tr_steps))


                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
            #print("Current epoch: ", epoch)
            # Evaluation after each epoch
            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                model.eval()
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                # print(next(model.parameters()).is_cuda)
                y_true, y_pred = evaluate(eval_dataloader,model,label_map,args,tokenizer, device)
                report = classification_report(y_true, y_pred, digits=6)
                logger.info("***** Eval results *****")
                logger.info("\n%s", report)
                eval_f1 = f1_score(y_true,y_pred)
                print(datetime.datetime.now(), eval_f1)
                if eval_f1 > max_eval_f1:
                    max_eval_f1 = eval_f1
                    output_eval_file = os.path.join(output_dir, "eval_results.txt")
                    write2report(output_eval_file, report)
                    write2file(eval_examples,y_true, y_pred, os.path.join(output_dir, "eval_results.conll"))
                    # Save a trained model and the associated configuration
                    save_dir = output_dir + '/best_model/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    logger.info("Save best model to: %s", save_dir)
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
                    # model_to_save = model.module if hasattr(model,
                    #                                         'module') else model  # Only save the model it-self
                    # model_to_save.save_pretrained(save_dir)
                    # tokenizer.save_pretrained(save_dir)
                    model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                                    "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                                    "label_map": label_map}
                    json.dump(model_config, open(os.path.join(save_dir, "model_config.json"), "w"))
                logger.info("best f1 till now: %f", max_eval_f1)
            train_dataloader.sampler.update_competence(epoch)

    if args.do_predict:
        # Load a trained model and vocabulary that you have fine-tuned
        # model = Ner(save_dir)

        config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        model = Ner(args.bert_model, from_tf=False, config=config, tag_to_ix=tag2ix, device=device, use_rnn=args.use_rnn, use_crf=args.use_crf)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)
        save_dir = output_dir + '/best_model/'
        model.load_state_dict(torch.load(os.path.join(save_dir,'best_model.pt')))


        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        test_data = prepare_data(test_features)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        model.eval()
        y_true, y_pred = evaluate(test_dataloader,model,label_map,args,tokenizer,  device)

        report = classification_report(y_true, y_pred, digits=6)
        logger.info("***** Test results *****")
        logger.info("\n%s", report)
        output_test_file = os.path.join(output_dir, "test_results.txt")
        write2report(output_test_file, report)
        write2file(test_examples, y_true, y_pred, os.path.join(output_dir, "test_results.conll"))



if __name__ == "__main__":
    main()

    # # label list config
    # processors = {"ner": NerProcessor}
    # processor = processors['ner']()
    # data_dir = './data/tweets/'
    # # label_list = processor.get_labels('./data/tweets/')
    # label_list = set()
    # train_exmps = processor.get_train_examples(data_dir)
    # label_list = label_list.union(set([lbl[0] for doc in train_exmps for lbl in doc[1]]))
    # label_list = list(label_list)
    # label_list.remove('O')
    # label_list.append('O')
    # with open(data_dir+'labels.txt', 'w') as f:
    #     for lbl in label_list:
    #         f.write(lbl)
    #         f.write('\n')



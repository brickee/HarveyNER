import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                                  BertForTokenClassification, BertTokenizer,
                                  WarmupLinearSchedule,BertModel,BertPreTrainedModel)
from torch import nn
from torch.nn.parameter import Parameter
import torch.autograd as autograd
import copy


from data_utils_loc import subtokens2tokens


START_TAG: str = "[START]"
STOP_TAG: str = "[STOP]"

def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_

class Ner(nn.Module):

    def __init__(self,
            bert_model,
            from_tf,
            config,
            tag_to_ix,
            device,
            use_crf,
            use_rnn,
            hidden_size: int=256,
            rnn_input_dim: int= 768,
            rnn_layers: int = 1,
            reproject_embeddings: bool = False,
            train_initial_hidden_state: bool = False,
            use_dropout: bool = True,
            rnn_type: str = "LSTM",

    ):
        super(Ner, self).__init__()
        if 'base' in bert_model:
            rnn_input_dim = 768
        elif 'large' in bert_model:
            rnn_input_dim = 1024

        self.bert =  BertModel.from_pretrained(bert_model,
              from_tf = from_tf,config = config)
        self.num_labels = config.num_labels # labels include PAD, CLS, SEP but not START and STOP
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)   
        self.device = device

        self.reproject_embeddings = reproject_embeddings
        if self.reproject_embeddings:
            self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)


        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.rnn_layers: int = rnn_layers
        self.nlayers: int = rnn_layers
        self.bidirectional = True
        self.rnn_type = rnn_type
        self.rnn_input_dim = rnn_input_dim
        self.train_initial_hidden_state = train_initial_hidden_state

        self.use_crf: bool = use_crf


        # if we use a CRF, we must add special START and STOP tags to the dictionary  # we already have it.
        self.tag_to_ix = copy.deepcopy(tag_to_ix)
        self.tag_to_ix[START_TAG] = len(tag_to_ix)+1
        self.tag_to_ix[STOP_TAG] = len(tag_to_ix) + 2
        self.tagset_size = len(self.tag_to_ix)+1  # plus one <unk> tag, all NER systems have such settings. 


        # bidirectional LSTM on top of embedding layer
        if self.use_rnn:
            num_directions = 2 if self.bidirectional else 1

            if self.rnn_type in ["LSTM", "GRU"]:

                self.rnn = getattr(torch.nn, self.rnn_type)(
                    rnn_input_dim,
                    hidden_size,
                    num_layers=self.nlayers,
                    dropout=0.0 if self.nlayers == 1 else 0.5,
                    bidirectional=True,
                    batch_first=True,
                )
                # Create initial hidden state and initialize it
                if self.train_initial_hidden_state:
                    self.hs_initializer = torch.nn.init.xavier_normal_

                    self.lstm_init_h = Parameter(
                        torch.randn(self.nlayers * num_directions, self.hidden_size),
                        requires_grad=True,
                    )

                    self.lstm_init_c = Parameter(
                        torch.randn(self.nlayers * num_directions, self.hidden_size),
                        requires_grad=True,
                    )



        if self.use_crf:
            self.output_size = self.tagset_size
        else:
            self.output_size = self.num_labels

        if self.use_rnn:
            self.linear = torch.nn.Linear(
                hidden_size * num_directions, self.output_size
            )
        else:
            self.linear = torch.nn.Linear(
                rnn_input_dim, self.output_size
            )

        if self.use_crf:
            self.transitions = torch.nn.Parameter(
                torch.randn(self.tagset_size, self.tagset_size)
            )
            # Matrix of transition parameters.  Entry i,j is the score of
            # transitioning *to* i *from* j.
            self.transitions.detach()[
            self.tag_to_ix[START_TAG], :
            ] = -10000

            self.transitions.detach()[
            :, self.tag_to_ix[STOP_TAG]
            ] = -10000


    @staticmethod
    def _softmax(x, axis):
        # reduce raw values to avoid NaN during exp
        x_norm = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x_norm)
        return y / y.sum(axis=axis, keepdims=True)


    def _forward_alg(self, feats, lens_):

        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        # START_TAG has all of the score.
        init_alphas[[self.tag_to_ix[START_TAG]]] = 0.0

        # Wrap in a variable so that we will get automatic backprop
        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1]+1,
            feats.shape[2],
            dtype=torch.float,
            device=self.device,
        )

        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)  # initialize the starts of all the sentences

        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(feats.shape[0], 1, 1)  # broadcast on the batches

        for i in range(feats.shape[1]): # i is the token index in a sentence, iterate through the sentence.
            emit_score = feats[:, i, :]

            tag_var = (
                    emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                    + transitions
                    + forward_var[:, i, :][:, :, None]
                    .repeat(1, 1, transitions.shape[2])
                    .transpose(2, 1)
            )

            max_tag_var, _ = torch.max(tag_var, dim=2)

            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lens_, :] # only take the final states in the sentences

        # final one in the sentence is SEP always test later if the same
        terminal_var = forward_var + self.transitions[
                                         self.tag_to_ix[STOP_TAG]
                                     ][None, :].repeat(forward_var.shape[0], 1)

        alpha = log_sum_exp_batch(terminal_var)

        return alpha

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long()
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        ## record the position of best score
        back_points = list()
        partition_history = list()
        ##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask =  (1 - mask.long()).bool()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.tag_to_ix[START_TAG], :].clone().view(batch_size, tag_size)  # bat_size * to_target_size
        partition_history.append(partition)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: batch_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            ## cur_bp: (batch_size, tag_size) max source score position in current tag
            ## set padded label as 0, which will be filtered in post processing
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        ### add score to final STOP_TAG
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1,0).contiguous() ## (batch_size, seq_len. tag_size)
        ### get the last position for each setences, and select the last partitions using gather()
        last_position = length_mask.view(batch_size,1,1).expand(batch_size, 1, tag_size) -1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size,tag_size,1)
        ### calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long().to(self.device)
        back_points.append(pad_zero)
        back_points  =  torch.cat(back_points).view(seq_len, batch_size, tag_size)

        ## select end ids in STOP_TAG
        pointer = last_bp[:, self.tag_to_ix[STOP_TAG]]
        insert_last = pointer.contiguous().view(batch_size,1,1).expand(batch_size,1, tag_size)
        back_points = back_points.transpose(1,0).contiguous()
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1,0).contiguous()
        ## decode from the end, padded position ids are 0, which will be filtered if following evaluation
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size)).to(self.device)
        decode_idx[-1] = pointer.detach()
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.detach().view(batch_size)
        path_score = None
        decode_idx = decode_idx.transpose(1,0)
        return  decode_idx

    def _score_sentence(self, feats, tags, lens_):

        start = torch.tensor(
            [self.tag_to_ix[START_TAG]], device=self.device
        )
        start = start[None, :].repeat(tags.shape[0], 1)

        stop = torch.tensor(
            [self.tag_to_ix[STOP_TAG]], device=self.device
        )
        stop = stop[None, :].repeat(tags.shape[0], 1)

        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)

        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i]:] = self.tag_to_ix[STOP_TAG]


        score = torch.FloatTensor(feats.shape[0]).to(self.device)

        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i])).to(self.device)

            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, : lens_[i] + 1], pad_start_tags[i, : lens_[i] + 1]
                ]
            ) + torch.sum(feats[i, r, tags[i, : lens_[i]]])

        return score
    #
    def _calculate_loss(
            self, logits, labels, lengths
    ) :
        forward_score = self._forward_alg(logits, lengths)
        gold_score = self._score_sentence(logits, labels, lengths)
        loss = forward_score - gold_score  # batch_size NLL of the gold label
        return loss.mean()






    def forward(self,  input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None):


        # Here labels are label_ids, they include [CLS]->26 and [SEP]->27
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        lengths = []

        valid_input_ids = []
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device=self.device)
        for i in range(batch_size):
            jj = -1
            valid_input_id = []
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1 and attention_mask[i][j]==1:   # take the head of those divided tokens! appdend 0 after it TODO: better way to make it faster?
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
                        valid_input_id.append(int(input_ids[i][j].to('cpu').numpy()))
            valid_input_ids.append(valid_input_id)
            lengths.append(jj+1)

        if self.use_dropout:
            sequence_output = self.dropout(valid_output)


        if self.use_rnn:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                sequence_output, lengths, enforce_sorted=False, batch_first=True
            )
            # if initial hidden state is trainable, use this state
            if self.train_initial_hidden_state:
                initial_hidden_state = [
                    self.lstm_init_h.unsqueeze(1).repeat(1, batch_size, 1),
                    self.lstm_init_c.unsqueeze(1).repeat(1, batch_size, 1),
                ]
                rnn_output, hidden = self.rnn(packed, initial_hidden_state)
            else:
                rnn_output, hidden = self.rnn(packed)

            sequence_output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True, total_length=max_len  # Must use total length to run multilple GPUs
            )
            if self.use_dropout:
                sequence_output = self.dropout(sequence_output)


        logits = self.linear(sequence_output)

        if self.use_crf:
            if labels is not None:
                loss  = self._calculate_loss(logits, labels,lengths)
                return loss
            else:

                best_path = self._viterbi_decode(logits,attention_mask_label)
                return  best_path
        else:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=0)
                # Only keep active parts of the loss
                # attention_mask_label = None
                if attention_mask_label is not None:
                    active_loss = attention_mask_label.reshape(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.reshape(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.reshape(-1))
                return loss
            else:
                return logits  #TODO: bug here when not using crf but use rnn in evaluation



if __name__ == "__main__":

    ner = Ner()
    ner._score_sentence()

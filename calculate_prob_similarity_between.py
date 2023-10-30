import csv
import sys

import torch
from transformers import *
import torch.nn.functional as F
import numpy as np
from scipy import spatial
from collections import defaultdict

#Roberta and Gpt2 use bype-level BPE
def init_model(model_name):
    if model_name == "xlnet":
        pretrained_name = 'xlnet-base-cased'
        tokenizer = XLNetTokenizer.from_pretrained(pretrained_name)
        model = XLNetModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
        model_lm = XLNetLMHeadModel.from_pretrained(pretrained_name).eval()
    elif model_name == "distillbert":
        pretrained_name = 'distilbert-base-cased'
        tokenizer = DistilBertTokenizer.from_pretrained(pretrained_name)
        model = DistilBertModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
        model_lm = DistilBertForMaskedLM.from_pretrained(pretrained_name).eval()
    elif model_name == "bert":
        pretrained_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        model = BertModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
        model_lm = BertForMaskedLM.from_pretrained(pretrained_name).eval()
    elif model_name == "bertlarge":
        pretrained_name = 'bert-large-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        model = BertModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
        model_lm = BertForMaskedLM.from_pretrained(pretrained_name).eval()
    elif model_name == "roberta":
        pretrained_name = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_name)
        model = BertModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
        model_lm = RobertaForMaskedLM.from_pretrained(pretrained_name).eval()
    elif model_name == "gpt":
        pretrained_name = 'openai-gpt'
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        model = OpenAIGPTModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
        model_lm = OpenAIGPTLMHeadModel.from_pretrained(pretrained_name).eval()
    elif model_name == "gpt2":
        pretrained_name = 'gpt2'
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_name)
        model = GPT2Model.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
        model_lm = GPT2LMHeadModel.from_pretrained(pretrained_name).eval()
    elif model_name == "gpt2large":
        pretrained_name = 'gpt2-large'
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_name)
        model = GPT2Model.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
        model_lm = GPT2LMHeadModel.from_pretrained(pretrained_name).eval()
    else:
        logger.error("unsupported model: {}".format(model_name))

    return tokenizer, model, model_lm

def match_piece_to_word(piece, word):
    mapping = defaultdict(list)
    word_index = 0
    piece_index = 0
    while (word_index < len(word.split()) and piece_index < len(piece)):
        if piece[piece_index] != '[UNK]':
            mid = piece[piece_index].strip('Ġ').strip('▁').strip('##')
            mid = mid.replace('</w>', '')
            t = len(mid)
        else:
            t = 1
        while (piece_index + 1 < len(piece) and t<len(word.split()[word_index])):
            mapping[word_index].append(piece_index)
            piece_index += 1
            if piece[piece_index] != '[UNK]':
                mid = piece[piece_index].strip('Ġ').strip('▁').strip('##')
                mid = mid.replace('</w>', '')
                t += len(mid)
            else:
                t += 1
        try:
            assert(t == len(word.split()[word_index]))
        except:
            print(word)
            print(piece)
            import pdb
            pdb.set_trace()
        mapping[word_index].append(piece_index)
        word_index += 1
        piece_index += 1
    return mapping

def convert_logits_to_probs(logits, input_ids):
    """"
    input:
        logits: (1, n_word, n_vocab), GPT2 outputed logits of each word
        input_inds: (1, n_word), the word id in vocab
    output: probs: (1, n_word), the softmax probability of each word
    """

    probs = F.softmax(logits[0], dim=1)
    n_word = input_ids.shape[1]
    res = []
    for i in range(n_word):
        res.append(probs[i, input_ids[0][i]].item())
    return np.array(res).reshape(1, n_word)


if __name__ == '__main__':
    '''
    parameters
    ind1: index for the cue word, starting from 0
    ind2: index for the target word, starting from 0
    inputfile: sentences with cue and target word
    
    '''
    ind1 = -4  # index for the cue word, first word index is 0
    ind2 = -3  # index for the target word

    ind3 = -5
    # ind4 = 0
    # ind5 = 0
#inputfile = 'lovely_blue.txt'
    inputfile = 'bluehat_index.txt'

    model_names = ["xlnet", "distillbert", "bert", "bertlarge", "roberta", "gpt", "gpt2", "gpt2large"]
    #model_names = ["gpt2large"]
    for model_name in model_names:
        fields = ['sent', 'cue', 'target',  'prob_cue', 'prob_target', 'prob_this', 'cosine_distance1', 'cosine_distance2', 'cosine_distance3',
                      'cosine_distance4', 'more']

        print(model_name)
        # if model_name in ["distillbert"]:
        #     fields = ['sent', 'cue', 'target',  'prob_cue', 'prob_target', 'cosine_sim_layer0', 'cosine_sim_layer1', 'cosine_sim_layer2',
        #               'cosine_sim_layer3', 'cosine_sim_layer4', 'cosine_sim_layer5', 'cosine_sim_layer6']

        # elif model_name in ["bert",  "roberta", "xlnet", "gpt", "gpt2"]:
        #     fields = ['sent', 'cue', 'target', 'prob_cue', 'prob_target', 'cosine_sim_layer0', 'cosine_sim_layer1',
        #               'cosine_sim_layer2',
        #               'cosine_sim_layer3', 'cosine_sim_layer4', 'cosine_sim_layer5', 'cosine_sim_layer6',
        #               'cosine_sim_layer7', 'cosine_sim_layer8', 'cosine_sim_layer9', 'cosine_sim_layer10',
        #               'cosine_sim_layer11', 'cosine_sim_layer12']

        # elif model_name in ["bertlarge", "gpt2large"]:
        #     fields = ['sent', 'cue', 'target', 'prob_cue', 'prob_target', 'cosine_sim_layer0', 'cosine_sim_layer1', 'cosine_sim_layer2',
        #               'cosine_sim_layer3','cosine_sim_layer4','cosine_sim_layer5','cosine_sim_layer6',
        #               'cosine_sim_layer7','cosine_sim_layer8','cosine_sim_layer9','cosine_sim_layer10',
        #               'cosine_sim_layer11', 'cosine_sim_layer12', 'cosine_sim_layer13', 'cosine_sim_layer14',
        #               'cosine_sim_layer15','cosine_sim_layer16', 'cosine_sim_layer17','cosine_sim_layer18',
        #               'cosine_sim_layer19','cosine_sim_layer20', 'cosine_sim_layer21', 'cosine_sim_layer22',
        #               'cosine_sim_layer23','cosine_sim_layer24']

        out = [] # cosine1 .. cosine12 prob1 prob2
        for input in open(inputfile):
            input_sent = input.strip()
            ind4 = int(input_sent.split()[-2])
            ind5 = int(input_sent.split()[-1])

            tokenizer, model, model_lm = init_model(model_name)
            input_ids = tokenizer.encode(input_sent, return_tensors = "pt")
            tok_input = tokenizer.convert_ids_to_tokens(input_ids[0])
            print(input_ids)
            print(tok_input)
            if model_name in ["xlnet"]:
                tok_input = tok_input[0:-2]
                input_ids = input_ids[:,0:-2]
            elif model_name in ["distillbert", "bert", "bertlarge", "roberta"]:
                tok_input = tok_input[1:-1]
                input_ids = input_ids[:,1:-1]

            #word1 = tok_input[ind1-1]
            #word2 = tok_input[ind2-1]
            tok_sent = ' '.join(tok_input).replace('Ġ', '').replace('▁', '').replace('##', '').replace('</w>', '')
            word_piece_mapping = match_piece_to_word(tok_input, input_sent)
            # print(word_piece_mapping)
            word1 = tok_input[word_piece_mapping[len(input_sent.split()) + ind1][0]]
            word2 = tok_input[word_piece_mapping[len(input_sent.split()) + ind2][0]]
            word3 = tok_input[word_piece_mapping[len(input_sent.split()) + ind3][0]]

            with torch.no_grad():
                outputs = model(input_ids)
                logits = model_lm(input_ids)[0]
            hidden_states = outputs['hidden_states']
            # print(len(hidden_states))
            prob = convert_logits_to_probs(logits, input_ids)[0]
            # print(len(prob), prob)
            prob1 = 1
            prob2 = 1
            prob3 = 1
            for i in word_piece_mapping[len(input_sent.split())+ind1]:
                prob1 *= prob[i]
            for i in word_piece_mapping[len(input_sent.split())+ind2]:
                prob2 *= prob[i]
            for i in word_piece_mapping[len(input_sent.split())+ind3]:
                    prob3 *= prob[i]
            mid = [tok_sent, word1, word2, prob1, prob2, prob3]

            for j in range(ind4 + 1, ind5 + 1):
                mid1 = 0
                for i in range(len(outputs['hidden_states'])):
                    vec1 = outputs['hidden_states'][i][0, word_piece_mapping[ind4], :].detach().numpy().mean(axis=0)
                    vec2 = outputs['hidden_states'][i][0, word_piece_mapping[j], :].detach().numpy().mean(axis=0)
                    mid1 += 1 - spatial.distance.cosine(vec1, vec2)
                mid.append(mid1/len(outputs['hidden_states']))
            out.append(mid)

        with open('results1/' + 'out_' + inputfile[0:-4] + '_' + model_name + '.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            for row in out:
                csvwriter.writerow(row)


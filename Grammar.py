import math
import itertools
import torch
import pytorch_pretrained_bert
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
# Load pre-trained model (weights)
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

max_scores_cache = []


def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)


def max_score(sentence):
    s_list = sentence.split()
    s_list_set = set(s_list)

    # check if in cache
    for (sentence_set, max_score) in max_scores_cache:
        if s_list_set == sentence_set:
            return max_score

    all_sentences = list(itertools.permutations(s_list))
    # find max score
    max_s = 0
    # max nb of iterations?
    for l in all_sentences:
        s = ' '.join(l)
        r = score(s)
        if r > max_s:
            # print(s)
            max_s = r

    # put in cache
    max_scores_cache.append((s_list_set, max_s))
    return max_s


def relative_score(sentence):
    max_s = max_score(sentence)
    return score(sentence) / max_s

# a=['apple and pear company', 'pear and apple  company', 'company apple and pear', 'apple pear and company']
# print([score(i) for i in a])

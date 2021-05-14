# todo rework smart acronym generator
# todo importance per word, higher = more likely to be used (more letters) in acro. use nlp to predict if not given
# todo volgorde woorden niet veranderen (switch on/of)
#todo simple combinations know what word part leeter of acronym

from math import *
from builtins import range
import numpy as np
import nltk
# nltk.download('cmudict')
from pronounceable import Complexity
import time
#import Grammar
import Flair

complexity = Complexity()

try:
    nltk.corpus.words.ensure_loaded()
    nltk.corpus.brown.ensure_loaded()
    nltk.corpus.gutenberg.ensure_loaded()
    nltk.corpus.wordnet.ensure_loaded()
except LookupError:
    print('Initial downloading of word corpus')
    for d in ['words', 'brown', 'gutenberg', 'wordnet']:
        nltk.download(d)
corpus = nltk.corpus.words
corpus_set = set(corpus.words()) # changed to gensim


import gensim.downloader as gensim_api
gensim_corpus = gensim_api.load('text8')
from gensim.models.word2vec import Word2Vec
# print('making/loading gensim model') #todo gensimi still needed?

from gensim.models import KeyedVectors
gensim_model = KeyedVectors.load('vectors.kv')
#gensim_model = Word2Vec(gensim_corpus)
#gensim_model.save('vectors.kv')

gensim_model_set = set(gensim_model.wv.vocab)
for w in ['radar','anova','sonar']: # remove good acronym words
    if w in gensim_model_set: gensim_model_set.remove(w)
#print(gensim_model.wv.most_similar('tree'))
#print(gensim_model.wv.similarity('tree', 'of'))

from gensim.summarization import keywords


#todo global params: max/min letters per word, use syns, change word order or not, etc

global combinations
combinations = []
global smart_combinations
smart_combinations = []
global simple_combinations
simple_combinations = []

global synonym_sets
synonym_sets = []

word_order_variable = False


def get_synonyms(word, threshold=500, similarity=0.3):
    #check if syn set exists
    global synonym_sets
    for set in synonym_sets:
        if word in set:
            #exists already, return set
            syns = set.copy()
            syns.remove(word)
            return syns

    # create new set
    synonyms = {word}
      # todo make parameter?
    for syn in nltk.corpus.wordnet.synsets(word):
        for lem in syn.lemmas():
            w = lem.name()
            if get_document_frequency(w) > threshold:
                if word in gensim_model_set and w in gensim_model_set:
                    if gensim_model.wv.similarity(word, w) > similarity:
                        synonyms.add(w)

    syns = synonyms.copy()
    syns.discard(word)
    synonym_sets.append(synonyms)
    return syns


def next_piece_in_word(wordlist, word, group_list, orig_word=None, prev_choices=[], max_letters_to_use=3, ):
    global combinations
    if orig_word is None:
        orig_word = word
    for j, w in enumerate(wordlist):
        for i in range(np.min([len(w), max_letters_to_use])):
            s = w[0:(i + 1)]
            if s == word[0:(i + 1)]:
                new_choices = prev_choices[:]
                new_choices.append([w, i])
                if len(s) == len(word):
                    combinations.append(new_choices)
                else:
                    new_wordlist = wordlist[group_list != group_list[j]]
                    new_group_list = group_list[group_list != group_list[j]]
                    if len(new_wordlist):
                        new_word = word[(i + 1):]
                        next_piece_in_word(new_wordlist, new_word, new_group_list, orig_word=orig_word, prev_choices=new_choices, max_letters_to_use=max_letters_to_use)


def print_smart_acronyms():
    if len(smart_combinations) > 0:
        for seq in smart_combinations:
            acronym = seq[0]
            exp_acronym = seq[1]
            score = "{:.2f}".format(seq[2])
            print("\t" + (acronym) + ' | ' + (exp_acronym) + ' | score: ' + (score))
    else:
        print("No smart acronyms found.")


def print_simple_acronyms():
    for seq in simple_combinations:
        acronym = seq[0]
        exp_acronym = ' '.join(word for word in seq[1])
        score = "{:.2f}".format(seq[2])
        print("\t" + acronym + ' | ' + exp_acronym + ' | score: ' + score)


def print_acronyms():
    print("\t" + 'Smart Acronyms: ')
    print_smart_acronyms()
    print("\t" + 'Simple Acronyms: ')
    print_simple_acronyms()
    #todo only print n first acros


def generate_smart_acronyms(keywordlist):
    # Init grouping
    group_list = list(range(len(keywordlist)))
    # Look for grouped words
    temp_keywordlist = keywordlist[:]
    temp_group_list = group_list[:]
    for i, w in enumerate(keywordlist):
        if (len(w.split(',')) > 1):
            temp_keywordlist.remove(w)
            groupid = group_list[i]
            temp_group_list.remove(groupid)
            for w2 in w.split(','):
                temp_keywordlist.append(w2)
                temp_group_list.append(groupid)
    keywordlist = temp_keywordlist[:];
    group_list = temp_group_list[:]
    # Look for words that allow synonyms
    use_syns = True
    if use_syns:
        use_synonyms = np.full(len(keywordlist), True)
    else:
        use_synonyms = np.full(len(keywordlist), False)
        for i, w in enumerate(keywordlist):
            if (w[0] == '*'):
                use_synonyms[i] = True
                keywordlist[i] = w[1:]

    min_acronymlength = 3
    max_letters_to_use = 6
    global combinations
    combinations = []
    # Add synonyms if needed, note that this will add a LOT of new results many using synonyms of the same word several times
    if np.sum(use_synonyms):
        temp_keywordlist = keywordlist[:]
        temp_group_list = group_list[:]
        for i, w in enumerate(keywordlist):
            if use_synonyms[i]:
                syns = get_synonyms(w)
                for s in syns:
                    if (s.isalpha()):
                        temp_keywordlist.append(s.lower())
                        temp_group_list.append(group_list[i])
        temp_keywordlist, uniq_idx = np.unique(temp_keywordlist, return_index=True)
        keywordlist = temp_keywordlist[:]
        group_list = np.array(temp_group_list)[uniq_idx]
    keywordlist = np.array(keywordlist)
    group_list = np.array(group_list)
    key_chars = []
    for w in keywordlist:
        for c in w[0:max_letters_to_use]:
            key_chars.append(c)
    key_chars = np.unique(key_chars)
    word_list = np.unique([w.lower() for w in corpus.words() if
                           w.isalpha() and (len(w) >= min_acronymlength) and set(list(w)).issubset(set(key_chars))])
    for w in word_list:
        next_piece_in_word(keywordlist, w, group_list, max_letters_to_use=max_letters_to_use)


def reformat_smart_acronyms():
    global smart_combinations
    for seq in combinations:
        acronym = ''
        exp_acronym = ''
        for piece in seq:
            capital_part = (piece[0][0:(piece[1] + 1)]).upper()
            noncapital_part = (piece[0][(piece[1] + 1):]).lower()
            acronym += capital_part
            exp_acronym += capital_part + noncapital_part # + ' '
        smart_combinations.append([acronym, exp_acronym])

#todo capitalise relevant letters in exp

# TODO also with synonyms etc


# generate acr part for each keyword
def generate_naive_acronyms(keywordlist):
    acrs_parts = []
    for keyword in keywordlist:
        keyword_acrs = generate_acr_part(keyword.upper())
        acrs_parts.append(keyword_acrs)

    # generate all possible naive acrs from multiple keyword acrs
    acr_lst = recursive_exhaust([], 0, acrs_parts)

    global simple_combinations
    for i in range(0, len(acr_lst)):
        acr_exp = keywordlist
        simple_combinations.append([acr_lst[i].upper(), acr_exp])


# return a list of the first n letters of the word, to be used in an acronym
def generate_acr_part(word):
    lst = []
    acr = ""
    for i in word:
        acr += i
        lst.append(acr)
    return lst


def recursive_exhaust(lst, current_word_index, matrix):
    if current_word_index == 0:
        lst = matrix[0]

    if len(matrix) == current_word_index + 1:
        return matrix[current_word_index]

    else:
        newlst = []
        for acr1 in lst:
            for acr2 in recursive_exhaust(matrix[current_word_index + 1], current_word_index + 1, matrix):
                newlst.append(acr1 + acr2)
        lst = newlst
        return lst


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    if denominator == 0:
        return 0
    return round(numerator / float(denominator), 3)


def get_document_frequency(word):
    f = open('enwiki-20190320-words-frequency.txt', encoding='utf-8')
    freq = 0
    for line in f:
        line = line.strip().split()
        #print(line)
        if line[0] == word:
            freq = int(line[1])
            break

    f.close()
    return freq


def get_idf(word):
    freq = max(get_document_frequency(word), 1.1)
    return 1.0 / log10(freq)


# return the score of a seq
def compute_max_similarity(acro, exp):
    max_similarity = 0
    if acro.lower() in gensim_model_set:
        for w in exp.split():
            if w.lower() in gensim_model_set:
                sim = gensim_model.wv.similarity(acro.lower(), w.lower())
                if sim > max_similarity:
                    max_similarity = sim
    return max_similarity


def compute_representation(acro, exp):
    relevant_letters_vec = []
    importances_vec = []

    importances = [(w, get_idf(w.lower())) for w in exp.split()]
    for (keyword, imp) in importances:
        nb_capitals = 0
        for w in exp.split():
            if w.lower() == keyword.lower():
                nb_capitals = sum(1 for c in w if c.isupper())
        relevant_letters_vec.append(nb_capitals)
        importances_vec.append(imp)

    rep = cosine_similarity(relevant_letters_vec, importances_vec)
    return rep


def compute_grammar(exp):
    s = Flair.perplexity(exp)
    return s


acronym_evaluation_functions = []  # list of functions for evaluation of acro

def pronouncability(acro, exp): return complexity.complexity(acro.lower())
def extra_length(acro, exp): return max(0, len(acro) - len(exp.split()))
def is_word(acro, exp): return 1 if acro.lower() in gensim_model_set else 0
def max_similarity(acro, exp): return compute_max_similarity(acro, exp)
def representation(acro, exp): return compute_representation(acro, exp)
def grammar(acro, exp): return compute_grammar(exp.lower())


# acronym_evaluation_functions.extend([pronouncability, extra_length, is_word, max_similarity, representation, grammar])
# weights = [-0.26, -0.11, 1.08, 20.76, 26.50]
acronym_evaluation_functions.extend([pronouncability, extra_length, is_word, max_similarity, representation])
acronym_evaluation_functions.append(grammar)
weights = [-0.0305103, -0.00705881, -0.14422545, 0.71355269, 0.46824187]
# grammar_weights = [-0.13241045, -0.01163915, -0.20622822,  1.05071494,  0.16005865,  -0.13950423] # todo grammar w should be neg ?
grammar_weights = [-0.14636966, -0.04194246, -0.14581107,  0.85065471,  0.31059875,  -0.17286973]


# weights_no_word = [-0.26, -0.11, 0, 0, 26.50]
# new_weights = [-0.0305103, -0.00705881, -0.14422545, 0.71355269, 0.46824187]
# weights.append(-1.06)
# new_weights.append(-0.09843551)


def score_seq(acro, exp, variable_word_order=False):
    s = 0.0
    for i in range(len(acronym_evaluation_functions) - 1):
        f = acronym_evaluation_functions[i]
        s += grammar_weights[i] * f(acro, exp)
    if variable_word_order:
        f = acronym_evaluation_functions[-1]
        s += grammar_weights[-1] * f(acro, exp)
    return s


def score_seq_no_word(acro, exp, variable_word_order=False):
    s = 0.0
    for i in range(len(acronym_evaluation_functions)):
        f = acronym_evaluation_functions[i]
        s += weights[i] * f(acro, exp)
    if variable_word_order:
        s += grammar_weights[5] * acronym_evaluation_functions[5](acro, exp)
    return s


def rank_smart_acronyms():
    for seq in smart_combinations:
        rank = score_seq(seq[0], seq[1], word_order_variable)
        seq.append(rank)


def rank_naive_acronyms():
    for seq in simple_combinations:
        rank = score_seq(seq[0], seq[1], word_order_variable)
        seq.append(rank)


def sort_smart_acronyms():
    smart_combinations.sort(key=lambda x: x[2], reverse=True)


def sort_naive_acronyms():
    simple_combinations.sort(key=lambda x: x[2], reverse=True)


def print_info(t1, t2):
    seconds_elapsed = t2 - t1
    nb_acronyms = len(smart_combinations) + len(simple_combinations)
    print("Found " + str(nb_acronyms) + " acronyms in " + "{:.2f}".format(seconds_elapsed) + " seconds.")


def main():
    running = False
    if not running:
        return
    expression = input("Enter keywords: ")
    starting_time = time.time()

    keywordlist = expression.split(" ")
    keywordlist = [w.lower() for w in keywordlist]

    generate_smart_acronyms(keywordlist)
    reformat_smart_acronyms()
    # generate_naive_acronyms(keywordlist)

    ending_time = time.time()

    rank_smart_acronyms()
    # rank_naive_acronyms()

    sort_smart_acronyms()
    # sort_naive_acronyms()

    print_info(starting_time, ending_time)
    print_acronyms()


# execute program
if __name__ == "__main__":
   main()

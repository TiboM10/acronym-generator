import Acronym
import random
import itertools
import functools
import numpy as np

# global params
max_letters_per_word = 3  # standard 3
min_letters_per_word = 0  # standard 0
max_acronym_length = 10  # standard 10
min_acronym_length = 1  # standard 1
use_synonyms = False  # if true allow synonyms
variable_word_order = False  # if true allow changes in word order
variable_wordgroup_order = False  # if true allow changes in wordgroup order
word_order_variable = variable_wordgroup_order or variable_word_order

current_expression = ''

# first acronym


def acronym_from_expression(expression):
    acro = ''
    for w in expression.split():
        for l in w:
            if l.isupper():
                acro += l
    return acro


def generate_initial_acronym(starting_expression):
    first_expression = ' '.join(word[0].upper() + word[1:].lower() for word in starting_expression.split())

    first_acronym = acronym_from_expression(first_expression)
    #print(first_acronym)
    #print(first_expression)
    return (first_acronym, first_expression)


def get_nb_capital_letters(word):
    nb_capitals = 0
    for l in word:
        if l.isupper():
            nb_capitals += 1
    return nb_capitals


def one_letter_difference(previous_expression):
    expressions = []
    previous_acro = acronym_from_expression(previous_expression)
    #print(previous_acro)
    expression_list = previous_expression.split()
    if len(previous_acro) < max_acronym_length:  # can add one letter to acro
        for i in range(len(expression_list)):
            word = expression_list[i]
            nb_capitals = get_nb_capital_letters(word)
            if nb_capitals < max_letters_per_word and nb_capitals < len(word):  # add one capital letter to this word in expr
                new_word = word[:nb_capitals] + word[nb_capitals].upper() + word[nb_capitals + 1:]
                #print('new word: ' + new_word)
                new_expression_list = expression_list[:]
                new_expression_list[i] = new_word
                expressions.append(' '.join(new_expression_list))  # add new expression to expressions
    if len(previous_acro) > min_acronym_length:  # can remove one letter from acro
        for i in range(len(expression_list)):
            word = expression_list[i]
            nb_capitals = get_nb_capital_letters(word)
            if nb_capitals > min_letters_per_word:  # add one capital letter to this word in expr
                new_word = word[:nb_capitals - 1] + word[nb_capitals - 1].lower() + word[nb_capitals:]
                #print('new word: ' + new_word)
                new_expression_list = expression_list[:]
                new_expression_list[i] = new_word
                expressions.append(' '.join(new_expression_list))  # add new expression to expressions
    #print(expression_list)
    return expressions

#print(one_letter_difference('The BIg cat'))


def synonym_swap(previous_expression):
    expressions = []
    expression_list = previous_expression.split()
    for i in range(len(expression_list)):
        word = expression_list[i]
        nb_capitals = get_nb_capital_letters(word)
        for syn in Acronym.get_synonyms(word):
            if len(syn) >= nb_capitals:
                new_word = syn[:nb_capitals].upper() + syn[nb_capitals:].lower()
                # print('new word: ' + new_word)
                new_expression_list = expression_list[:]
                new_expression_list[i] = new_word
                expressions.append(' '.join(new_expression_list))  # add new expression to expressions
    return expressions


#depr
def swap_word_order(previous_expression):
    expressions = []
    expression_list = previous_expression.split()
    if len(expression_list) < 2:
        return expressions
    for i in range(len(expression_list) - 1):
        for j in range(i + 1, len(expression_list)):
            # swap 2 words
            new_expression_list = expression_list[:]
            new_expression_list[i], new_expression_list[j] = new_expression_list[j], new_expression_list[i]
            expressions.append(' '.join(new_expression_list))  # add new expression to expressions
    return expressions


def swap_wordgroup_order(previous_expression):
    exp_list = previous_expression.split()
    neighbours = set()
    if len(exp_list) > 3:
        #return 10 random permutations
        for i in range(10):
            neighbours.add(' '.join(np.random.permutation(exp_list)))
        return list(neighbours)
    iters = list(itertools.permutations(exp_list))
    return [' '.join(i) for i in iters]


#return neighb but not self
def get_exclusively_neighbouring_acronyms(previous_expression):
    neighbour_expressions = []  # does not includes self as neighbour

    neighbour_expressions.extend(one_letter_difference(previous_expression))
    if use_synonyms:
        neighbour_expressions.extend(synonym_swap(previous_expression))
    if variable_word_order:
        neighbour_expressions.extend(swap_word_order(previous_expression))
    if variable_wordgroup_order:
        neighbour_expressions.extend(swap_wordgroup_order(previous_expression))

    # return list of 'çlose' acronyms
    return neighbour_expressions


# different kinds of elementary changes
#@functools.lru_cache(maxsize=1024) #todo useful here?
def get_neighbouring_acronyms(previous_expression):
    neighbour_expressions = get_exclusively_neighbouring_acronyms(previous_expression)  # includes self as neighbour

    # return list of 'çlose' acronyms
    neighbour_expressions.append(previous_expression)
    return neighbour_expressions


# stop when no improvement in rating, return best expr
def find_best_acronyms(prev_expression):
    current_expression = prev_expression
    expressions = [current_expression]
    while True:
        current_acronym = acronym_from_expression(current_expression)
        # print(current_acronym, current_expression)
        current_score = Acronym.score_seq(current_acronym, current_expression, word_order_variable)
        prev_score = current_score

        neighbouring_acros = get_neighbouring_acronyms(current_expression)
        for expr in neighbouring_acros:
            acro = acronym_from_expression(expr)
            score = Acronym.score_seq(acro, expr, word_order_variable)
            if score > current_score:
                current_expression = expr
                current_acronym = acro
                current_score = score
                expressions.append(current_expression)
        # if no progress, return
        if prev_score == current_score:
            return expressions


# stop when no improvement in rating, return best expr
def find_best_acronym(prev_expression):
    current_expression = prev_expression
    while True:
        current_acronym = acronym_from_expression(current_expression)
        # print(current_acronym, current_expression)
        current_score = Acronym.score_seq(current_acronym, current_expression, word_order_variable)
        prev_score = current_score

        neighbouring_acros = get_neighbouring_acronyms(current_expression)
        for expr in neighbouring_acros:
            acro = acronym_from_expression(expr)
            score = Acronym.score_seq(acro, expr, word_order_variable)
            if score > current_score:
                current_expression = expr
                current_acronym = acro
                current_score = score
        # if no progress, return
        if prev_score == current_score:
            return current_expression


def find_distant_neigbours(prev_expression, depth):  # find al neighbours of given expression up to depth n
    if depth < 1:
        return {prev_expression}
    elif depth == 1:
        return set(get_neighbouring_acronyms(prev_expression))  # set
    else:  # depth > 1
        neighbours = find_distant_neigbours(prev_expression, depth - 1)
        new_neighbours = neighbours.copy()
        for e in neighbours:
            for n in get_neighbouring_acronyms(e):
                new_neighbours.add(n)
        return new_neighbours


def find_best_deep_neighbours(exp, depth): # todo some sort of caching

    deep_neighbours = find_distant_neigbours(exp, depth)
    best_acros = set()
    for n in deep_neighbours:
        for e in find_best_acronyms(n):
            best_acros.add(e)
    return best_acros


def stochastic_beam_search_find_deep_neighbours(exp, depth, beta=20, p=0.9): # beta = 20, p = 0.9
    # beta nb of exp that are expanded every step
    best_acros = {exp}
    for i in range(depth):
        # generate new candidates from best_acros
        all_neighbours = set()
        for e in best_acros:
            for n in get_neighbouring_acronyms(e):
                all_neighbours.add(n)
        #in last step: return all neighbours instead of 20 'best'
        if i == depth - 1:  # last iteration
            return all_neighbours

        # select stochastic new best acros from candidates
        # pairwise tournament selection: select best with prob p

        new_best_acros = set()
        while len(new_best_acros) < beta and len(all_neighbours) > 2:  # todo or convergence
            a = all_neighbours.pop()
            b = all_neighbours.pop()
            if Acronym.score_seq(acronym_from_expression(a), a,word_order_variable) < Acronym.score_seq(acronym_from_expression(b), b,word_order_variable):
                a, b = b, a
            # a is now best acronym
            if random.random() < p:
                # add best with prob p
                new_best_acros.add(a)
                all_neighbours.add(b)
            else:
                new_best_acros.add(b)
                all_neighbours.add(a)

        best_acros = new_best_acros.copy()
    return best_acros


def beam_search_find_deep_neighbours(exp, depth, beta=20):
    # beta nb of exp that are expanded every step
    best_acros = {exp}
    for i in range(depth):
        # generate new candidates from best_acros
        all_neighbours = []
        for e in best_acros:
            for n in get_neighbouring_acronyms(e):
                all_neighbours.append(n)
        # select stochastic new best acros from candidates
        # pairwise tournament selection: select best with prob p
        sorted(all_neighbours, key=lambda e: Acronym.score_seq(acronym_from_expression(e), e, word_order_variable), reverse=True)

        best_acros = set(all_neighbours[:beta])
    return best_acros


# return best acros, ranked on score
def generate_acronyms(exp, depth):
    # search_algo = find_best_deep_neighbours
    search_algo = stochastic_beam_search_find_deep_neighbours
    # search_algo = beam_search_find_deep_neighbours
    sorted_exps = sort_acros(search_algo(exp, depth))
    return sorted_exps


def sort_acros(exps):
    return sorted(exps, key=lambda e: Acronym.score_seq(acronym_from_expression(e), e, word_order_variable), reverse=True)


# pretty print list of expressions
def pprint_acros(exps):
    for exp in exps:
        acro = acronym_from_expression(exp)
        print(acro, ": ", exp,':', "{:.4f}".format(Acronym.score_seq(acro, exp, word_order_variable)))


def generate_n_good_acros(starting_expression, n, depth=5):
    (first_acronym, first_expression) = generate_initial_acronym(starting_expression)
    # todo check param constraints for first acro, return if invalid

    exps = generate_acronyms(first_expression, depth)
    return exps[:n]


def random_deep_neighbour(starting_expression, depth=3):
    prev_acro = starting_expression
    new_acro = starting_expression
    for i in range(depth):
        new_acro = random.choice(get_neighbouring_acronyms(prev_acro))
        prev_acro = new_acro
    return new_acro


def main():
    starting_expression = input("Enter expression: ")
    pprint_acros(generate_n_good_acros(starting_expression, 10))


# execute program
if __name__ == "__main__":
   main()

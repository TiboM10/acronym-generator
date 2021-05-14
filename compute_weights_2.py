import Acronym
import Generator
import cvxpy as cp
import numpy as np

expansions = []


def load_acronyms_from_file(file):
    f = open(file, "r")
    global expansions
    for l in f:
        if l != "":
            expansions.append(l)
    f.close()


# setup evaluations etc of acronyms for time efficiency
# return matrix of acr_groups (acro and neighbours) as a list of exp, acro, and func evaluations
def setup():
    all_exps = []  # matrix with first elem of every list original exp
    for exp in expansions:
        l = [exp]
        neigbour_exps = Generator.get_exclusively_neighbouring_acronyms(exp)
        for e in neigbour_exps:
            l.append(e)
        all_exps.append(l)

    setup_matrix = []
    for exp_group in all_exps:
        row = []
        for exp in exp_group:
            l = [exp]
            acro = Generator.acronym_from_expression(exp)
            l.append(acro)
            for f in Acronym.acronym_evaluation_functions:
                l.append(f(acro, exp))
            row.append(l)
        setup_matrix.append(row)

    return setup_matrix


def solve(setup_matrix):
    # Create two constraints.
    global e
    constraints = []
    constraint_matrices = []

    # make variable (weight) for every evaluation function
    weights_len = len(Acronym.acronym_evaluation_functions)
    print('nb weights =', weights_len)
    weights = cp.Variable(weights_len)
    v = weights.value

    # make slack variables
    slack = []
    # slack = [[0,s:e0n1, s:e0n2,...]
    #          [0,s:e1n1, s:e1n2,...]
    #           ...
    #         ]
    for expansion_group in range(0, len(setup_matrix)):
        for neighbour in range(1, len(setup_matrix[expansion_group])):
            # make slack variable for every neighbour exp
            # sl = cp.Variable(nonneg=True)
            # s.append(sl)
            # constraints.append(sl >= 0)

            # # make square slack variable
            # sl2 = cp.Variable(nonneg=True)
            # constraints.append(sl2 == sl ** 2)
            #slack2.append(sl)
            pass

        s = cp.Variable(len(setup_matrix[expansion_group]) - 1)
        slack.append(s)
        constraints.append(s >= 0)


    # Form objective.
    # obj = cp.Minimize(sum(np.square(slack2)))
    obj = cp.Minimize(sum([cp.sum_squares(s) for s in slack]))  # sq_sum of all slacks

    # create constraints
    for e in range(len(setup_matrix)):
        expansion_group = setup_matrix[e]
        C = np.zeros(shape=(weights_len, len(expansion_group) - 1))
        # print(C)
        # exp = expansion_group[0][0]
        # acro = expansion_group[0][1]
        w = np.array(expansion_group[0][2:])
        # print(w)
        for neighbour in range(1, len(expansion_group)):
            wn = np.array(expansion_group[neighbour][2:])
            #  make C
            C[:, neighbour - 1] = w - wn
            # constraints.append(np.dot(wv - wnv, weights) + slack[e][neighbour] >= e-6)
        constraint_matrices.append(C)

    # add min value constr for weights
    minimum = 0.1
    # constraints.append(cp.abs(weights) >= minimum)

    # print(constraint_matrices)

    for i in range(len(constraint_matrices)):  # = len(slack)
        n = len(constraint_matrices[i][0])
        C = constraint_matrices[i]  # 5*n
        s = slack[i]  # 1*n
        eta = np.array([0.000001] * n)  # 1*n

        constraints.append(weights @ C + s >= eta)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    # print(prob.variables)
    prob.solve()  # Returns the optimal value.
    print("status:", prob.status)
    print("sumsq slack =", prob.value)
    print("weights:", weights.value)
    sum_weights = sum(weights.value)
    normalised_weights = weights.value / sum_weights
    print("normalised weights:", normalised_weights)


file = "20good_acronyms.txt"
load_acronyms_from_file(file)
matrix = setup()
solve(matrix)

import Acronym
import Generator
import numpy
from ortools.linear_solver import pywraplp

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
        neigbour_exps = Generator.get_neighbouring_acronyms(exp)
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


def LinearProgramming(setup_matrix):
    # Instantiate a Glop solver, naming it SolveStigler.
    solver = pywraplp.Solver('Solver', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    objective = solver.Objective()

    # x = solver.NumVar(0, solver.infinity(), 'x')
    # y = solver.NumVar(0, solver.infinity(), 'y')

    # make variable (weight) for every evaluation function
    weights = [0] * len(Acronym.acronym_evaluation_functions)
    for i in range(0, len(weights)):
        weights[i] = solver.NumVar(-solver.infinity(), solver.infinity(), 'w' + str(i))

    # make slack variables
    slack = []
    # slack = [[0,s:e0n1, s:e0n2,...]
    #          [0,s:e1n1, s:e1n2,...]
    #           ...
    #         ]
    for expansion_group in range(0, len(setup_matrix)):
        s = [0]
        for neighbour in range(1, len(setup_matrix[expansion_group])):
            # make slack variable for every neighbour exp
            s.append(solver.NumVar(0.0, solver.infinity(), 's:e' + str(expansion_group) + 'n' + str(neighbour)))
            objective.SetCoefficient(s[neighbour], 1)  # objective is sum of slacks
            # make square slack variable
            # todo squares not possible :(
        slack.append(s)

    objective.SetMinimization()

    # print('Number of variables =', solver.NumVariables())

    #create constraints todo equalities instead of inequalities?
    for e in range(len(setup_matrix)):
        expansion_group = setup_matrix[e]
        # exp = expansion_group[0][0]
        # acro = expansion_group[0][1]
        w = expansion_group[0][2:]
        for neighbour in range(1, len(expansion_group)):
            wn = expansion_group[neighbour][2:]
            solver.Add(numpy.dot(w, weights) + slack[e][neighbour] >= numpy.dot(wn, weights) + e-6)

    # print('Number of constraints =', solver.NumConstraints())

    # add min value constr for weights
    # minimum = 0.1
    # m = 10000
    # for w in weights:
    #     b = solver.IntVar(0, 1, 'b')  # b = 0 Or 1
    #     solver.Add(w + m * b >= minimum)
    #     solver.Add(w + m * b <= m - minimum)

    # Solve the system.
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', objective.Value())
        # todo print weights
        for w in weights:
            print(w.solution_value())
    else:
        print('The problem does not have an optimal solution.')

    print('\nAdvanced usage:')
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())


file = "good_acronyms.txt"
load_acronyms_from_file(file)
matrix = setup()
LinearProgramming(matrix)

import Acronym
import Generator

# see if we can find good acro from deep neighbour, and at what depth/rank

exps = [
        'Together Everyone Achieves More',
        'ANalysis Of VAriance',
        'Smart Acronym GEnerator using MAchine Learning',
        'UNiversity of Cambridge Local Examinations Syndicate',
        'PROgram MAnagement System',
        'Disk And Execution MONitor',
        'COmbined release and Radiation Effects Satellite',
        'INTernational English Language testing System',
        'Packet INternet Groper',
        'RAdio Detection And Ranging',
]

weights = [
    [-0.01415706, -0.11343679, 0.0338701,   0.05217203,  1.79660482, -0.7550531],
    [-0.06861414, -0.05074295,  0. ,         0. ,         1.10649523,  0.01286186],
    [-0.70682993,  0.04933734,  0.  ,        0. ,         5.74735644, -4.08986385],
    [9.63862911e-07, -2.03019336e-06, -4.04037123e-01,  1.40392877e+00,   1.13944285e-04, -4.52944798e-06],
    [-0.09957548, -0.06592344,  0. ,         0. ,         1.47203271, -0.30653379],
    [-6.05400058e-03, -8.05017505e-04, -6.37716892e-02,  9.59549120e-01,  1.09844904e-01,  1.23668365e-03],
    [-0.03145942,  0. ,        -0.16244455,  0.43466644,  0.9285106,  -0.16927307],
    [-0.68132117, -0.50492444, -0.23514066, -0.64350895,  5.20419298, -2.13929776],
    [-0.12162889,  0.0012613,   0.16893135,  0.28972986,  0.48787186,  0.17383452],
    [0.39006747,  0.23387718,  0.04653449,  0.76323064, -0.282459,   -0.15125077],
]

# for exp in exps:
#     best_acros = Generator.generate_n_good_acros(exp.lower(), 20)
#     if exp in best_acros:
#         print(best_acros.index(exp), best_acros)
#     else:
#         print('acro not found in', best_acros)

# todo run with stat beam

for e in range(10):

    exp = exps[e]
    print(exp)
    w = weights[e]
    #w[5] = 0
    Acronym.grammar_weights = w

    for i in range(5):
        random_deep_acro = Generator.random_deep_neighbour(exp, i)

        best_acros = Generator.generate_acronyms(random_deep_acro, i)[:20]
        # print(i,random_deep_acro)
        if exp in best_acros:
            print(best_acros.index(exp))
        else:
            print('/')

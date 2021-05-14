import Acronym
import Generator

# see if we can find good acro from deep neighbour, and at what depth/rank

exps = [
        'INTernational English Language testing System',
        'PROgram MAnagement System',
        'Disk And Execution MONitor',
        'COmbined release and Radiation Effects Satellite',
        'Packet INternet Groper',
        'RAdio Detection And Ranging',
        'Together Everyone Achieves More',
        'ANalysis Of VAriance',
        'Smart Acronym GEnerator using MAchine Learning',
        'UNiversity of Cambridge Local Examinations Syndicate'
]

# for exp in exps:
#     best_acros = Generator.generate_n_good_acros(exp.lower(), 20)
#     if exp in best_acros:
#         print(best_acros.index(exp), best_acros)
#     else:
#         print('acro not found in', best_acros)

for exp in exps:
    random_deep_acro = Generator.random_deep_neighbour(exp, 3)
    best_acros = Generator.generate_acronyms(random_deep_acro, 3)[:20]

    if exp in best_acros:
        print(best_acros.index(exp), best_acros)
    else:
        print('acro not found in', best_acros)

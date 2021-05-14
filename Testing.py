import Acronym
import Generator
import Flair
import numpy as np

# #print(Generator.beam_search_find_deep_neighbours('Disk and execution monitor',3))
# exp = 'UNiversity of Cambridge Local Examinations Syndicate'
#
# # print(np.random.permutation(exp.split()))
#
# for n in Generator.get_neighbouring_acronyms(exp):
#         print(n, Acronym.compute_grammar(n))
#

# exps = [
#         'Disk And Execution MONitor',
#         'Completely Automated Public turing Test to tell Computers and Humans Apart',
#         'Coronavirus Aid Relief and Economic Security',
#         'SOund Navigation And Ranging',
#         'Together Everyone Achieves More',
#         'ANalysis Of VAriance',
#         'RAdio Detection And Ranging',
#         'Packet INternet Groper',
#         'Light Amplification by Stimulated Emission of Radiation',
#         'Keep It Simple Stupid'
# ]

exps = [
        'Institute for MAgnetic Resonance Imaging',
        'WATerbury Catholic High School',
        'Barcelona international business school',
        'European agency for worldwide development',
        'strategy and business operations associate',
        'Stanford Physics Information Retrieval System',
        'Elastomeric Reusable Surface Insulation',
        'IMprovise ADapt Overcome',
        'Southern ORegon UNiversity',
        'National Transport And Logistics company'
]

for e in exps:
        # print(Generator.acronym_from_expression(e), Acronym.pronouncability(Generator.acronym_from_expression(e), e))
        # print([(exp, Acronym.score_seq(Generator.acronym_from_expression(exp), exp)) for exp in Generator.generate_n_good_acros(e,5)])
        # print(Generator.acronym_from_expression(e),':', e, ',', Acronym.score_seq(Generator.acronym_from_expression(e), e,False))
        # neighbours = Generator.get_exclusively_neighbouring_acronyms(e)
        # best_acros = Generator.generate_n_good_acros(e,10,1)
        # if e in best_acros:
        #     print(1 + best_acros.index(e), best_acros)
        # else:
        #     print('acro not found in', best_acros)
        print(e,Generator.generate_n_good_acros(e,10,4))

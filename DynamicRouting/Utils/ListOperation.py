import numpy as np


def onehot(number_of_state, activating_state):
    code = np.zeros(number_of_state, dtype=bool)
    code[activating_state] = True
    return code

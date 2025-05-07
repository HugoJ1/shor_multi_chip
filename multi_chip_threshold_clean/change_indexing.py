#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaire pour changer l'indexation des qubits afin de pouvoir comparer.

Created on Fri Apr  7 19:07:48 2023
@author: elie
"""
import stim

# Obtenu par symétrie autour de l'axe vertical.
TEST_DICT = {1: 13,
             2: 16,
             3: 14,
             5: 15,
             8: 7,
             9: 10,
             10: 8,
             11: 11,
             12: 9,
             13: 12,
             14: 4,
             15: 1,
             16: 5,
             17: 2,
             18: 6,
             19: 3,
             25: 0
             }


def to_str_dict(dico):
    """Convertie un dico en le même mais avec des str."""
    return {str(key): str(val) for key, val in dico.items()}


def change_indexing(circuit, mapping):
    """Change l'indexation des qubits du circuit."""
    mapping = to_str_dict(mapping)
    res = ''
    for line in repr(circuit).splitlines():
        if line.startswith('stim.Circuit'):
            continue
        elif line.startswith("''')"):
            continue
        elif line.lstrip().startswith('REPEAT'):
            res += line
        else:
            res += (' '*(len(line) - len(line.lstrip())) +
                    ' '.join(mapping.get(word, word) for word in line.split()))
        res += '\n'
    return stim.Circuit(res)

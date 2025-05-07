#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimation du seuil d'un code de répétition avec télémesure.

Ce code est construit en trois temps:
    1. les stabilisateurs sont définis simplement en disant quel qubit joue
    quel rôle ;
    2. le circuit est synthétisé, uniquement à partir de l'information "quel
       qubit est où ?" (le cycle du code de surface est toujours le même et
       seules les positions des qubits importent) ;
    3. Du circuit, sinter fait les calculs pour obtenir le seuil.
"""
from collections import namedtuple
import scipy
from tqdm import tqdm
import os
import stim
import sinter
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, minimize, report_fit, report_ci, conf_interval, Minimizer, fit_report
from IPython.display import clear_output
import matplotlib.ticker as mticker
from Coordinates import Coord, SPLIT_DIST
from noise_v2 import NoiseModel, Probas
from fitting import *

SPLIT = Coord(0, SPLIT_DIST)
SPLIT_tel_CNOT_X = Coord(0, 0.5)
SPLIT_tel_CNOT_Y = Coord(0.5, 0)


# %% Tools functions
def multiply_namedtuple(namedtuple_instance, multiplier):
    """Multiply a namedtuple by a float."""
    return namedtuple_instance._replace(
        **{field: getattr(namedtuple_instance, field) * multiplier for field in namedtuple_instance._fields})


def merge_dicts(dict1, dict2):
    """Merge of dict appropriate for dict of stabs or qubits coordinates."""
    result = {}
    # Loop through keys in both dictionaries
    for key in dict1.keys() | dict2.keys():
        # Concatenate lists if key exists in both dictionaries
        result[key] = dict1.get(key, []) + dict2.get(key, [])

    return result


def flatten(iterable_of_lists):
    """Flatten a list of lists (or any iterable of lists)."""
    return sum(iterable_of_lists, [])


def flatten_dict(dico):
    """Make a list out of the values of a dictionnary that should be lists."""
    return sum((v for v in dico.values()), [])


def plot_stabs(data_qubits, x_stabs, z_stabs, x_tel=None, z_tel=None,
               convention='ij'):
    """Representation of where are the data qubits and stabilizers.

    The input format is the output from surf_qubits_stabs (and other functions).
    Normal convention is 'ij', but as crumble uses the 'xy' convention, it is
    allowed to use it (but note that I changed the coordinates of the qubits in
    stim so that crumble also shows as following the 'ij' convention).
    """
    if convention not in ('ij', 'xy'):
        raise ValueError("Convention must be 'ij' or 'xy'!")
    _, ax = plt.subplots()
    data = np.array(data_qubits)
    x = np.array(flatten_dict(x_stabs))
    z = np.array(flatten_dict(z_stabs))
    x_t = np.empty((0, 2)) if x_tel is None else np.array(flatten_dict(x_tel))
    z_t = np.empty((0, 2)) if z_tel is None else np.array(flatten_dict(z_tel))
    if convention == 'ij':  # invert x and y axis
        data = np.flip(data, axis=1)
        x = np.flip(x, axis=1)
        z = np.flip(z, axis=1)
        x_t = np.flip(x_t, axis=1)
        z_t = np.flip(z_t, axis=1)
    ax.scatter(data[:, 0], data[:, 1], color='red', edgecolors='black',
               label="data")
    ax.scatter(x[:, 0], x[:, 1], color='grey', edgecolors='black', label="x")
    ax.scatter(z[:, 0], z[:, 1], color='white', edgecolors='black', label="z")
    if x_t.size > 0:
        ax.scatter(x_t[:, 0], x_t[:, 1], color='grey', edgecolors='blue',
                   label="tele_x")
    if z_t.size > 0:
        ax.scatter(z_t[:, 0], z_t[:, 1], color='white', edgecolors='b',
                   label="tele_z")
    if convention == 'ij':
        ax.invert_yaxis()
        ax.xaxis.tick_top()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_aspect('equal', 'box')
    ax.grid(linestyle=':')
    ax.set_axisbelow(True)
    plt.show()


def nbr_cycle(d):
    """Return the number of cycle in one simulation."""
    return (3*d)


# %% Stabilizer description


def surf_qubits_stabs(dist_i, dist_j=None, shift=(0, 0)):
    """Génère les qubits et stabilisateur d'un qubit logique."""
    if dist_j is None:
        dist_j = dist_i
    data_qubits = [Coord(2*i, 2*j)
                   for i in range(1, dist_i+1) for j in range(1, dist_j+1)]
    x_stabs = {
        'top': [Coord(1, 1+2*j) for j in range(1, dist_j) if j % 2 == 0],
        'bottom': [Coord(1+2*dist_i, 1+2*j) for j in range(1, dist_j)
                   if (j+dist_i) % 2 == 0],
        'bulk': [Coord(1+2*i, 1+2*j) for i in range(1, dist_i)
                 for j in range(1, dist_j) if (i+j) % 2 == 0]
    }
    z_stabs = {
        'left': [Coord(1+2*i, 1) for i in range(1, dist_i) if i % 2 == 1],
        'right': [Coord(1+2*i, 1+2*dist_j) for i in range(1, dist_i)
                  if (i+dist_j) % 2 == 1],
        'bulk': [Coord(1+2*i, 1+2*j) for i in range(1, dist_i)
                 for j in range(1, dist_j) if (i+j) % 2 == 1]
    }
    # Check that only (even, even) and (odd, odd) coordinates are possible
    assert all((x[0] + x[1]) % 2 == 0 for x in data_qubits)
    assert all((x[0] + x[1]) % 2 == 0
               for s in {**x_stabs, **z_stabs}.values() for x in s)
    # Shift the logical qubit
    data_qubits = [x + shift for x in data_qubits]
    x_stabs = {key: [x + shift for x in val] for key, val in x_stabs.items()}
    z_stabs = {key: [x + shift for x in val] for key, val in z_stabs.items()}
    return data_qubits, x_stabs, z_stabs


def split_surf_code(data_qubits, x_stabs, z_stabs, split_col, x_tel={}, z_tel={}, naive_seam=False):
    """Split the surface code and introduce measurement teleportation.

    If split_col is a list it will create several seam
    """
    if isinstance(split_col, list):
        x_tel = {}
        z_tel = {}
        for col in split_col:
            # puts the teleported stab in _tel_col and remove them from _stabs
            data_qubits, x_stabs, z_stabs, x_tel_col, z_tel_col = split_surf_code(
                data_qubits, x_stabs, z_stabs, col, x_tel, z_tel)
            x_tel = merge_dicts(x_tel, x_tel_col)
            z_tel = merge_dicts(z_tel, z_tel_col)
    else:
        if split_col % 2 == 1 and not naive_seam:
            if any(point.j == split_col for point in data_qubits):
                raise ValueError(
                    "Only measurements can be splitted, check positions.")
            x_tel = {key: [p for p in val if p.j == split_col]
                     for key, val in x_stabs.items()}
            z_tel = {key: [p for p in val if p.j == split_col]
                     for key, val in z_stabs.items()}
            x_stabs = {key: [p for p in val if p not in x_tel[key]]
                       for key, val in x_stabs.items()}
            z_stabs = {key: [p for p in val if p not in z_tel[key]]
                       for key, val in z_stabs.items()}
        elif split_col % 2 == 1 and naive_seam:
            if any(point.j == split_col for point in data_qubits):
                raise ValueError(
                    "Only measurements can be splitted, check positions.")
            x_tel = {key: [p for p in val if p.j == split_col]
                     for key, val in x_stabs.items()}
            z_tel = {key: [p for p in val if p.j == split_col]
                     for key, val in z_stabs.items()}
            x_stabs = {key: [p for p in val if p not in x_tel[key]]
                       for key, val in x_stabs.items()}
            z_stabs = {key: [p for p in val if p not in z_tel[key]]
                       for key, val in z_stabs.items()}
        elif split_col % 2 == 0:
            for key, val in x_stabs.items():
                if any(point.j == split_col for point in x_stabs[key]):
                    raise ValueError(
                        "Only data can be splitted in v2, check positions.")
            x_tel = {key: [p for p in val if (p.j == split_col+1 or p.j == split_col-1)]
                     for key, val in x_stabs.items()}
            z_tel = {key: [p for p in val if (p.j == split_col+1 or p.j == split_col-1)]
                     for key, val in z_stabs.items()}
            x_stabs = {key: [p for p in val if p not in x_tel[key]]
                       for key, val in x_stabs.items()}
            z_stabs = {key: [p for p in val if p not in z_tel[key]]
                       for key, val in z_stabs.items()}
    return data_qubits, x_stabs, z_stabs, x_tel, z_tel


def _double_points(points):
    """Split each of the points."""
    return sum([[p-SPLIT, p+SPLIT] for p in points], [])


def _bell_pair_for_CNOT(points):
    """Create the coordinates of the center of Bell pairs used for teleported CNOT."""
    return sum([[p+SPLIT_tel_CNOT_X+SPLIT_tel_CNOT_Y, p+SPLIT_tel_CNOT_X-SPLIT_tel_CNOT_Y] for p in points], [])


def _prepare_ids(data_qubits, x_stabs, z_stabs, x_tel=None, z_tel=None):
    """Prepare metadata for easier handling of indexes.

    qubits_dict and qubits_id_dict are guaranted to be in the same order.
    """
    # Joints lists of stabilizers
    x_tel_joint = [] if x_tel is None else flatten_dict(x_tel)
    z_tel_joint = [] if z_tel is None else flatten_dict(z_tel)
    x_tel_joint_split = _double_points(x_tel_joint)
    z_tel_joint_split = _double_points(z_tel_joint)
    qubits_dict = {'data': data_qubits,
                   'x': flatten_dict(x_stabs), 'z': flatten_dict(z_stabs),
                   'x_tel_split': x_tel_joint_split,
                   'z_tel_split': z_tel_joint_split}
    # All the qubits and the indexes
    qubits_list = sorted(flatten_dict(qubits_dict))
    qubits_index = {q: i for i, q in enumerate(qubits_list)}
    # Add for convenience the set of all stabilizers qubits ; main interest is
    # that it's lenght allow to easily compute where the results are in the
    # measurement reg.
    qubits_dict['stabs'] = sum(
        (qubits_dict[i] for i in ('x', 'z', 'x_tel_split', 'z_tel_split')), [])
    # Indexes for each type of qubits.
    qubits_id_dict = {key: [qubits_index[q] for q in val]
                      for key, val in qubits_dict.items()}
    assert all(qubits_dict[key] == [qubits_list[i] for i in qubits_id_dict[key]]
               for key in qubits_dict.keys()), "Order not coinciding."
    # Add virtual unsplit qubits for teleported measurements
    qubits_dict['x_tel'] = x_tel_joint
    qubits_dict['z_tel'] = z_tel_joint
    qubits_dict['stabs_virtual'] = sum(
        (qubits_dict[i] for i in ('x', 'z', 'x_tel', 'z_tel')), [])
    return qubits_list, qubits_index, qubits_dict, qubits_id_dict


def _prepare_ids_teleport_CNOT(data_qubits, x_stabs, z_stabs, x_tel=None, z_tel=None):
    """Prepare metadata for easier handling of indexes in the case of teleported CNOT scheme.

    qubits_dict and qubits_id_dict are guaranted to be in the same order.
    """
    # Joints lists of stabilizers
    x_tel_joint = [] if x_tel is None else flatten_dict(x_tel)
    z_tel_joint = [] if z_tel is None else flatten_dict(z_tel)
    x_tel_joint_split = _double_points(_bell_pair_for_CNOT(x_tel_joint))
    z_tel_joint_split = _double_points(_bell_pair_for_CNOT(z_tel_joint))
    qubits_dict = {'data': data_qubits,
                   'x': flatten_dict(x_stabs), 'z': flatten_dict(z_stabs),
                   'x_tel_split': x_tel_joint_split,
                   'z_tel_split': z_tel_joint_split}
    # Add virtual unsplit qubits for teleported measurements
    qubits_dict['x_tel'] = x_tel_joint
    qubits_dict['z_tel'] = z_tel_joint
    # All the qubits and the indexes
    qubits_list = sorted(flatten_dict(qubits_dict))
    qubits_index = {q: i for i, q in enumerate(qubits_list)}
    # Add for convenience the set of all stabilizers qubits ; main interest is
    # that it's lenght allow to easily compute where the results are in the
    # measurement reg.
    qubits_dict['stabs'] = sum(
        (qubits_dict[i] for i in ('x', 'z', 'x_tel', 'z_tel')), [])
    # Indexes for each type of qubits.
    qubits_id_dict = {key: [qubits_index[q] for q in val]
                      for key, val in qubits_dict.items()}
    assert all(qubits_dict[key] == [qubits_list[i] for i in qubits_id_dict[key]]
               for key in qubits_dict.keys()), "Order not coinciding."
    qubits_dict['stabs_virtual'] = sum(
        (qubits_dict[i] for i in ('x', 'z', 'x_tel', 'z_tel')), [])
    return qubits_list, qubits_index, qubits_dict, qubits_id_dict
# %% Circuit synthesis


def _initialize_circuit(state, qubits_list, qubits_id):
    """Create the circuit, and initialize it.

    state is either '0' or '+'
    """
    state = state.casefold()
    if state not in ('0', '+'):
        raise ValueError("State should be '0' or '+'!")
    # Create circuit and label qubits
    circuit = stim.Circuit()
    for i, q in enumerate(qubits_list):
        circuit.append('QUBIT_COORDS', i, q.stim_coord())
    # Initialisation in |0>|0>|0>...|0> or |+>|+>|+>...|+>
    # We consider we can't prepare + :
    circuit.append("R", qubits_id['data'])
    # Initialisation in 0 of the stabilisers ancilary qubits
    circuit.append("R", qubits_id['stabs'])
    if state == '+':
        circuit.append("H", qubits_id['data'])
    return circuit


def _extract_with_bell_pair(x, data_qubits, qubits_list, qubits_index, qubits,
                            qubits_id, split_col, dirr_z, dirr_x):
    """Build the circuit of extraction with bell pair for one time step."""
    circuit_mes_temp = stim.Circuit()
    cnot_args = []
    # If split_col is odd => cnot teleportation
    if x in qubits['z_tel'] and x + dirr_z in data_qubits and split_col % 2 == 1:
        if dirr_z == (-1, 1):
            cnot_args.extend([qubits_index[x + dirr_z],
                              qubits_index[x + SPLIT_tel_CNOT_X - SPLIT_tel_CNOT_Y +
                                           SPLIT]])
            cnot_args.extend([qubits_index[x + SPLIT_tel_CNOT_X - SPLIT_tel_CNOT_Y -
                                           SPLIT],
                              qubits_index[x]])
            # measurement of qubits in bell basis
            MZ_args = []
            MX_args = []
            MZ_args.append(qubits_index[x + SPLIT_tel_CNOT_X - SPLIT_tel_CNOT_Y +
                                        SPLIT])
            MX_args.append(qubits_index[x + SPLIT_tel_CNOT_X - SPLIT_tel_CNOT_Y -
                                        SPLIT])
            # classically controlled correction
            circuit_mes_temp.append("M", MZ_args)
            circuit_mes_temp.append("CX", [stim.target_rec(-1), qubits_index[x]])
            circuit_mes_temp.append("MX", MX_args)
            circuit_mes_temp.append("CZ", [stim.target_rec(-1), qubits_index[x + dirr_z]])
        elif dirr_z == (1, 1):
            cnot_args.extend([qubits_index[x + dirr_z],
                              qubits_index[x + SPLIT_tel_CNOT_X + SPLIT_tel_CNOT_Y +
                                           SPLIT]])
            cnot_args.extend([qubits_index[x + SPLIT_tel_CNOT_X + SPLIT_tel_CNOT_Y -
                                           SPLIT],
                              qubits_index[x]])
            # measurement of qubits in bell basis
            MZ_args = []
            MX_args = []
            MZ_args.append(qubits_index[x + SPLIT_tel_CNOT_X + SPLIT_tel_CNOT_Y +
                                        SPLIT])
            MX_args.append(qubits_index[x + SPLIT_tel_CNOT_X + SPLIT_tel_CNOT_Y -
                                        SPLIT])
            # classically controlled correction
            circuit_mes_temp.append("M", MZ_args)
            circuit_mes_temp.append("CX", [stim.target_rec(-1), qubits_index[x]])
            circuit_mes_temp.append("MX", MX_args)
            circuit_mes_temp.append("CZ", [stim.target_rec(-1), qubits_index[x + dirr_z]])
        else:
            cnot_args.extend([qubits_index[x + dirr_z], qubits_index[x]])
    elif x in qubits['x_tel'] and x + dirr_x in data_qubits and split_col % 2 == 1:
        if dirr_x == (-1, 1):
            cnot_args.extend([qubits_index[x + SPLIT_tel_CNOT_X - SPLIT_tel_CNOT_Y +
                                           SPLIT],
                              qubits_index[x + dirr_x]])
            cnot_args.extend([qubits_index[x],
                              qubits_index[x + SPLIT_tel_CNOT_X - SPLIT_tel_CNOT_Y -
                                           SPLIT]])
            # measurement in bell basis
            MZ_args = []
            MX_args = []
            MZ_args.append(qubits_index[x + SPLIT_tel_CNOT_X - SPLIT_tel_CNOT_Y -
                                        SPLIT])
            MX_args.append(qubits_index[x + SPLIT_tel_CNOT_X - SPLIT_tel_CNOT_Y +
                                        SPLIT])
            # classically controlled correction
            circuit_mes_temp.append("M", MZ_args)
            circuit_mes_temp.append("CX", [stim.target_rec(-1), qubits_index[x + dirr_x]])
            circuit_mes_temp.append("MX", MX_args)
            circuit_mes_temp.append("CZ", [stim.target_rec(-1), qubits_index[x]])
        elif dirr_x == (1, 1):
            cnot_args.extend([qubits_index[x + SPLIT_tel_CNOT_X + SPLIT_tel_CNOT_Y +
                                           SPLIT],
                              qubits_index[x + dirr_x]])
            cnot_args.extend([qubits_index[x],
                              qubits_index[x + SPLIT_tel_CNOT_X + SPLIT_tel_CNOT_Y -
                                           SPLIT]])
            # measurement in bell basis
            MZ_args = []
            MX_args = []
            MZ_args.append(qubits_index[x + SPLIT_tel_CNOT_X + SPLIT_tel_CNOT_Y -
                                        SPLIT])
            MX_args.append(qubits_index[x + SPLIT_tel_CNOT_X + SPLIT_tel_CNOT_Y +
                                        SPLIT])
            # classically controlled correction
            circuit_mes_temp.append("M", MZ_args)
            circuit_mes_temp.append("CX", [stim.target_rec(-1), qubits_index[x + dirr_x]])
            circuit_mes_temp.append("MX", MX_args)
            circuit_mes_temp.append("CZ", [stim.target_rec(-1), qubits_index[x]])
        else:
            cnot_args.extend([qubits_index[x], qubits_index[x + dirr_x]])
    # if split_col is even => measurement teleportation
    elif x in qubits['z_tel'] and x + dirr_z in data_qubits and split_col % 2 == 0 and x[1] == split_col-1:
        if dirr_z == (1, 1):
            cnot_args.extend([qubits_index[x + dirr_z],
                              qubits_index[x + SPLIT]])
        else:
            cnot_args.extend([qubits_index[x + dirr_z],
                              qubits_index[x - SPLIT]])

    elif x in qubits['x_tel'] and x + dirr_x in data_qubits and split_col % 2 == 0 and x[1] == split_col-1:
        if dirr_x == (-1, 1):
            cnot_args.extend(
                [qubits_index[x + SPLIT], qubits_index[x + dirr_x]])
        else:
            cnot_args.extend([qubits_index[x - SPLIT],
                              qubits_index[x + dirr_x]])

    elif x in qubits['z_tel'] and x + dirr_z in data_qubits and split_col % 2 == 0 and x[1] == split_col+1:
        if dirr_z == (1, -1):
            cnot_args.extend([qubits_index[x + dirr_z],
                              qubits_index[x - SPLIT]])
        else:
            cnot_args.extend([qubits_index[x + dirr_z],
                              qubits_index[x + SPLIT]])
    elif x in qubits['x_tel'] and x + dirr_x in data_qubits and split_col % 2 == 0 and x[1] == split_col+1:
        if dirr_x == (-1, -1):
            cnot_args.extend([qubits_index[x - SPLIT],
                              qubits_index[x + dirr_x]])
        else:
            cnot_args.extend([qubits_index[x + SPLIT],
                              qubits_index[x + dirr_x]])
    return (cnot_args, circuit_mes_temp)


def _surface_code_cycle(dist_i, dist_j, data_qubits, x_stabs, z_stabs,
                        x_tel=None, z_tel=None, split_col_list=None, naive_seam=False):
    """Un cycle de correction d'erreur, du H aux mesures/réinitialisation.

    Mais sans les détecteurs (car ils changent entre le premier cycle et les
    autres).

    Adapté aussi bien au code de surface normal qu'au code de surface splitté.
    Si split_col (du moins ses éléments) est pair on a deux lignes de paires de Bell en ancillae.
    Si split_col est impair on teleportes les cnot sur la frontière.
    La teleportation de porte fonctionne seulement avec une
    seule frontière pour le moment.
    """
    if naive_seam and split_col_list % 2 == 1:
        # Create the circuit
        qubits_list, qubits_index, qubits, qubits_id = _prepare_ids(
            data_qubits, x_stabs, z_stabs, x_tel, z_tel)
        circuit = stim.Circuit()
        circuit.append("TICK")
        # Prepare the X ancilla in the correct base.
        circuit.append("H", qubits_id['x'])  # No error as virtual gate
        # Prepare Bell pairs
        circuit.append("H", [qubits_index[q-SPLIT]
                             for q in qubits['x_tel'] + qubits['z_tel']])
        circuit.append("CX",
                       flatten([qubits_index[q-SPLIT], qubits_index[q+SPLIT]]
                               for q in qubits['x_tel'] + qubits['z_tel']))
        circuit.append("TICK")

        for dirr_x, dirr_z in zip([(-1, 1), (-1, -1), (1, 1), (1, -1)],
                                  [(-1, 1), (1, 1), (-1, -1), (1, -1)]):
            cnot_args = []
            # Temporary circuit to identify iddle qubits
            circuit_temp = stim.Circuit()
            for x in qubits['stabs_virtual']:
                if x in qubits['z'] and x + dirr_z in data_qubits:
                    cnot_args.extend([qubits_index[x + dirr_z], qubits_index[x]])

                elif x in qubits['x'] and x + dirr_x in data_qubits:
                    cnot_args.extend([qubits_index[x], qubits_index[x + dirr_x]])

        # naive way of splitting
                elif x in qubits['z_tel'] and x + dirr_z in data_qubits:
                    cnot_args.extend([qubits_index[x + dirr_z],
                                      qubits_index[x + dirr_z[1]*SPLIT]])

                elif x in qubits['x_tel'] and x + dirr_x in data_qubits:
                    cnot_args.extend([qubits_index[x + dirr_x[1]*SPLIT],
                                      qubits_index[x + dirr_x]])
            circuit_temp.append("CX", cnot_args)
            circuit += circuit_temp
            circuit.append("TICK")
        # Rotate basis because we only mesure in Z basis
        circuit.append("H", qubits_id['x'] + qubits_id['x_tel_split'])
        # Do the measures
        circuit.append("TICK")
        circuit.append("MR", qubits_id['stabs'])

    else:
        # Si on a donné une liste on regarde la parité du premier élément pour savoir quel splitting faire
        if isinstance(split_col_list, list):
            split_col = split_col_list[0]
        else:
            split_col = split_col_list
        teleport = x_tel is not None or z_tel is not None
        # Prepare data
        if split_col_list is not None:
            if split_col % 2 == 0:
                qubits_list, qubits_index, qubits, qubits_id = _prepare_ids(
                    data_qubits, x_stabs, z_stabs, x_tel, z_tel)
            elif split_col % 2 == 1:
                qubits_list, qubits_index, qubits, qubits_id = _prepare_ids_teleport_CNOT(
                    data_qubits, x_stabs, z_stabs, x_tel, z_tel)
        else:
            qubits_list, qubits_index, qubits, qubits_id = _prepare_ids(
                data_qubits, x_stabs, z_stabs, x_tel, z_tel)
        # Create the circuit
        circuit = stim.Circuit()

        circuit.append("TICK")
        # Prepare the X ancilla in the correct base.
        circuit.append("H", qubits_id['x'])  # No error as virtual gate
        if split_col_list is not None:
            if split_col % 2 == 1:
                circuit.append("H", qubits_id['x_tel'])
        # Prepare the Bell ancillas (considering they are already initialised at 0)

        if teleport and split_col % 2 == 0:
            circuit.append("H", [qubits_index[q-SPLIT]
                                 for q in qubits['x_tel'] + qubits['z_tel']])

            circuit.append("CX",
                           flatten([qubits_index[q-SPLIT], qubits_index[q+SPLIT]]
                                   for q in qubits['x_tel'] + qubits['z_tel']))
        elif teleport and split_col % 2 == 1:
            if split_col % 2 == 1:
                circuit.append("R", qubits_id['x_tel_split'] + qubits_id['z_tel_split'])
            circuit.append("H", [qubits_index[q+SPLIT_tel_CNOT_X-SPLIT_tel_CNOT_Y-SPLIT]
                                 for q in qubits['x_tel'] + qubits['z_tel']])
            circuit.append("H", [qubits_index[q+SPLIT_tel_CNOT_X+SPLIT_tel_CNOT_Y-SPLIT]
                                 for q in qubits['x_tel'] + qubits['z_tel']])
            circuit.append("CX",
                           flatten([qubits_index[q+SPLIT_tel_CNOT_X-SPLIT_tel_CNOT_Y-SPLIT],
                                    qubits_index[q+SPLIT_tel_CNOT_X-SPLIT_tel_CNOT_Y+SPLIT]]
                                   for q in qubits['x_tel'] + qubits['z_tel']))
            circuit.append("CX",
                           flatten([qubits_index[q+SPLIT_tel_CNOT_X+SPLIT_tel_CNOT_Y-SPLIT],
                                    qubits_index[q+SPLIT_tel_CNOT_X+SPLIT_tel_CNOT_Y+SPLIT]]
                                   for q in qubits['x_tel'] + qubits['z_tel']))
        circuit.append("TICK")

        for dirr_x, dirr_z in zip([(-1, 1), (-1, -1), (1, 1), (1, -1)],
                                  [(-1, 1), (1, 1), (-1, -1), (1, -1)]):
            cnot_args = []
            # Temporary circuit to identify idle qubits
            circuit_temp = stim.Circuit()
            circuit_mes_temp = stim.Circuit()
            for x in qubits['stabs_virtual']:
                if x in qubits['z'] and x + dirr_z in data_qubits:
                    cnot_args.extend([qubits_index[x + dirr_z], qubits_index[x]])

                elif x in qubits['x'] and x + dirr_x in data_qubits:
                    cnot_args.extend([qubits_index[x], qubits_index[x + dirr_x]])
            # extraction on the seam
                else:
                    if isinstance(split_col_list, list):
                        for split_col in split_col_list:
                            cnot_args_bell, circuit_mes_temp_bell = \
                                _extract_with_bell_pair(x, data_qubits, qubits_list, qubits_index,
                                                        qubits, qubits_id,
                                                        split_col, dirr_z, dirr_x)
                            cnot_args.extend(cnot_args_bell)
                            circuit_mes_temp += circuit_mes_temp_bell
                    else:
                        cnot_args_bell, circuit_mes_temp_bell = \
                            _extract_with_bell_pair(x, data_qubits, qubits_list, qubits_index,
                                                    qubits, qubits_id,
                                                    split_col, dirr_z, dirr_x)
                        cnot_args.extend(cnot_args_bell)
                        circuit_mes_temp += circuit_mes_temp_bell
            circuit_temp.append("CX", cnot_args)
            circuit += circuit_temp
            circuit.append("TICK")
            circuit += circuit_mes_temp
        # Rotate basis because we only mesure in Z basis
        if split_col_list is not None:
            if split_col % 2 == 0:
                circuit.append("H", qubits_id['x'] + qubits_id['x_tel_split'])
            if split_col % 2 == 1:
                circuit.append("H", qubits_id['x'] + qubits_id['x_tel'])
        else:
            circuit.append("H", qubits_id['x'])
        # Do the measures
        circuit.append("TICK")
        circuit.append("MR", qubits_id['stabs'])
    return circuit


def _add_surface_code_detectors(bloc, qubits, nb_stabs, x=True, z=True,
                                first=False, time=0, split_col=None, naive_seam=False):
    """Add detectors of the surface code.

    Assumes that the stabilizers where measured in the orders of qubits['stabs']
    """
    if isinstance(split_col, list):
        split_col_parity = split_col[0]
    else:
        split_col_parity = split_col
    if split_col is not None and not naive_seam:
        if split_col_parity % 2 == 1:
            for s in (qubits['x'] if x else []) + (qubits['z'] if z else []):
                mes = [stim.target_rec(qubits['stabs'].index(s) - nb_stabs)]
                if not first:  # Comparison with last turn.
                    # Teleported cnot only works for one seam for the moment
                    nb_bell_measurement = len(qubits['z_tel_split']) + \
                        len(qubits['x_tel_split']) - 2
                    mes += [stim.target_rec(qubits['stabs'].index(s) - 2 *
                                            nb_stabs-nb_bell_measurement)]
                bloc.append("DETECTOR", mes, s.stim_coord(time))
            for s in (qubits['x_tel'] if x else []) + (qubits['z_tel'] if z else []):
                mes = [stim.target_rec(qubits['stabs'].index(s) - nb_stabs)]
                if not first:  # Comparison with last turn
                    # Teleported cnot only works for one seam for the moment
                    nb_bell_measurement = len(qubits['z_tel_split']) + \
                        len(qubits['x_tel_split']) - 2
                    mes += [stim.target_rec(qubits['stabs'].index(s)-2 *
                                            nb_stabs-nb_bell_measurement)]
                bloc.append("DETECTOR", mes, s.stim_coord(time))
        else:
            for s in (qubits['x'] if x else []) + (qubits['z'] if z else []):
                mes = [stim.target_rec(qubits['stabs'].index(s) - nb_stabs)]
                if not first:  # Comparison with last turn.
                    mes += [stim.target_rec(qubits['stabs'].index(s) - 2*nb_stabs)]
                bloc.append("DETECTOR", mes, s.stim_coord(time))
            for s in (qubits['x_tel'] if x else []) + (qubits['z_tel'] if z else []):
                mes = [stim.target_rec(qubits['stabs'].index(s-SPLIT) - nb_stabs),
                       stim.target_rec(qubits['stabs'].index(s+SPLIT) - nb_stabs)]
                if not first:  # Comparison with last turn
                    mes += [stim.target_rec(qubits['stabs'].index(s-SPLIT)-2*nb_stabs),
                            stim.target_rec(qubits['stabs'].index(s+SPLIT)-2*nb_stabs)]
                bloc.append("DETECTOR", mes, s.stim_coord(time))
    else:
        for s in (qubits['x'] if x else []) + (qubits['z'] if z else []):
            mes = [stim.target_rec(qubits['stabs'].index(s) - nb_stabs)]
            if not first:  # Comparison with last turn.
                mes += [stim.target_rec(qubits['stabs'].index(s) - 2*nb_stabs)]
            bloc.append("DETECTOR", mes, s.stim_coord(time))
        for s in (qubits['x_tel'] if x else []) + (qubits['z_tel'] if z else []):
            mes = [stim.target_rec(qubits['stabs'].index(s-SPLIT) - nb_stabs),
                   stim.target_rec(qubits['stabs'].index(s+SPLIT) - nb_stabs)]
            if not first:  # Comparison with last turn
                mes += [stim.target_rec(qubits['stabs'].index(s-SPLIT)-2*nb_stabs),
                        stim.target_rec(qubits['stabs'].index(s+SPLIT)-2*nb_stabs)]
            bloc.append("DETECTOR", mes, s.stim_coord(time))


def _add_final_measure(circuit, kind, nb_data, nb_stabs, qubits, qubits_id,
                       time=1, split_col=None, naive_seam=False):
    """Add the final measurements and corresponding detectors."""
    kind = kind.casefold()
    if kind not in ('x', 'z'):
        raise ValueError("'kind' must be 'x' or 'z'!")
    circuit.append("TICK")
    circuit.append({'x': "MX", 'z': 'M'}[kind], qubits_id['data'])
    # Vérification des stabiliseurs à la main après mesure
    if isinstance(split_col, list):
        split_col_parity = split_col[0]
    else:
        split_col_parity = split_col
    if split_col is not None and not naive_seam:
        if split_col_parity % 2 == 1:
            for s in qubits[kind]:
                circuit.append(
                    "DETECTOR",
                    [stim.target_rec(qubits['data'].index(s + dirr) - nb_data)
                     for dirr in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                     if s + dirr in qubits['data']] +
                    [stim.target_rec(qubits['stabs'].index(s)-nb_stabs-nb_data)],
                    s.stim_coord(time))
            for s in qubits[kind+'_tel']:
                circuit.append(
                    "DETECTOR",
                    [stim.target_rec(qubits['data'].index(s + dirr) - nb_data)
                     for dirr in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                     if s + dirr in qubits['data']] +
                    [stim.target_rec(qubits['stabs'].index(s)-nb_stabs-nb_data)],
                    s.stim_coord(time))
        else:
            for s in qubits[kind]:
                circuit.append(
                    "DETECTOR",
                    [stim.target_rec(qubits['data'].index(s + dirr) - nb_data)
                     for dirr in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                     if s + dirr in qubits['data']] +
                    [stim.target_rec(qubits['stabs'].index(s) - nb_stabs - nb_data)],
                    s.stim_coord(time))
            for s in qubits[kind+'_tel']:
                circuit.append(
                    "DETECTOR",
                    [stim.target_rec(qubits['data'].index(s + dirr) - nb_data)
                     for dirr in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                     if s + dirr in qubits['data']] +
                    [stim.target_rec(qubits['stabs'].index(s-SPLIT)-nb_stabs-nb_data),
                     stim.target_rec(qubits['stabs'].index(s+SPLIT)-nb_stabs-nb_data)],
                    s.stim_coord(time))
    else:
        for s in qubits[kind]:
            circuit.append(
                "DETECTOR",
                [stim.target_rec(qubits['data'].index(s + dirr) - nb_data)
                 for dirr in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                 if s + dirr in qubits['data']] +
                [stim.target_rec(qubits['stabs'].index(s) - nb_stabs - nb_data)],
                s.stim_coord(time))
        for s in qubits[kind+'_tel']:
            circuit.append(
                "DETECTOR",
                [stim.target_rec(qubits['data'].index(s + dirr) - nb_data)
                 for dirr in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                 if s + dirr in qubits['data']] +
                [stim.target_rec(qubits['stabs'].index(s-SPLIT)-nb_stabs-nb_data),
                 stim.target_rec(qubits['stabs'].index(s+SPLIT)-nb_stabs-nb_data)],
                s.stim_coord(time))


def gen_memory(dist_i, dist_j, repeat, kind='x', split_col=None,
               probas=Probas(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), plot=False, naive_seam=False):
    """Génère le circuit pour une mémoire où on stoque |+> ou |0>.

    kind : 'x' or 'z'
    """
    kind = kind.casefold()
    if kind not in ('x', 'z'):
        raise ValueError("'kind' must be 'x' or 'z'!")
    if repeat is None:
        repeat = dist_i
    # Prepare qubits and stabilizers
    data_qubits, x_stabs, z_stabs = surf_qubits_stabs(dist_i, dist_j)
    x_tel = z_tel = None
    # Choose between old version and new splitting
    if split_col is not None:
        data_qubits, x_stabs, z_stabs, x_tel, z_tel = split_surf_code(
            data_qubits, x_stabs, z_stabs, split_col, naive_seam=naive_seam)
        # Just to get the parity of the columns
        if isinstance(split_col, list):
            split_col_parity = split_col[0]
        else:
            split_col_parity = split_col
        if split_col_parity % 2 == 0:
            qubits_list, qubits_index, qubits, qubits_id = _prepare_ids(
                data_qubits, x_stabs, z_stabs, x_tel, z_tel)
        elif split_col_parity % 2 == 1 and not naive_seam:
            qubits_list, qubits_index, qubits, qubits_id = _prepare_ids_teleport_CNOT(
                data_qubits, x_stabs, z_stabs, x_tel, z_tel)
        elif split_col_parity % 2 == 1 and naive_seam:
            qubits_list, qubits_index, qubits, qubits_id = _prepare_ids(
                data_qubits, x_stabs, z_stabs, x_tel, z_tel)
    else:
        qubits_list, qubits_index, qubits, qubits_id = _prepare_ids(
            data_qubits, x_stabs, z_stabs, x_tel, z_tel)
    if plot:
        plot_stabs(data_qubits, x_stabs, z_stabs, x_tel, z_tel,
                   convention='ij')

    # Useful for computing indexes in measurement record.
    nb_data, nb_stabs = len(qubits['data']), len(qubits['stabs'])
    # Initialize circuit and qubits
    circuit = _initialize_circuit({'x': '+', 'z': '0'}[kind],
                                  qubits_list, qubits_id)
    # First cycle
    circuit += _surface_code_cycle(dist_i, dist_j, data_qubits, x_stabs, z_stabs, x_tel, z_tel,
                                   split_col, naive_seam)
    _add_surface_code_detectors(circuit, qubits, nb_stabs, x=(kind == 'x'), z=(kind == 'z'),
                                first=True, split_col=split_col, naive_seam=naive_seam)
    # Generic cycle
    bloc = _surface_code_cycle(dist_i, dist_j, data_qubits, x_stabs, z_stabs, x_tel, z_tel,
                               split_col, naive_seam)
    bloc.append("SHIFT_COORDS", [], (0, 0, 1))
    _add_surface_code_detectors(bloc, qubits, nb_stabs, split_col=split_col, naive_seam=naive_seam)
    circuit.append(stim.CircuitRepeatBlock(repeat, bloc))
    # Mesure finale et assertion résultat
    _add_final_measure(circuit, kind, nb_data, nb_stabs, qubits, qubits_id,
                       time=1, split_col=split_col, naive_seam=naive_seam)
    # Observable
    if kind == 'x':
        logic_pauli = [Coord(2*i, 2) for i in range(1, dist_i+1)]
    elif kind == 'z':
        logic_pauli = [Coord(2, 2*j) for j in range(1, dist_j+1)]
    else:
        raise RuntimeError("Nothing to do here")
    circuit.append("OBSERVABLE_INCLUDE",
                   [stim.target_rec(qubits['data'].index(x) - nb_data)
                    for x in logic_pauli], 0)
    if dist_i == 3 and dist_j == 3:
        with open('circuits/diagram_without_error.svg', 'w') as f:
            print(circuit.diagram("timeline-svg"), file=f)
            # print(repr(circuit))
    # Prepare the list of qubits involved in bell pairs
    if split_col is not None:
        if split_col_parity % 2 == 0 or naive_seam:
            Bell_pairs_list = flatten([qubits_index[q-SPLIT], qubits_index[q+SPLIT]]
                                      for q in qubits['x_tel'] + qubits['z_tel'])
        elif split_col_parity % 2 == 1 and not naive_seam:
            Bell_pairs_list = flatten([qubits_index[q+SPLIT_tel_CNOT_X-SPLIT_tel_CNOT_Y-SPLIT],
                                       qubits_index[q+SPLIT_tel_CNOT_X-SPLIT_tel_CNOT_Y+SPLIT],
                                       qubits_index[q+SPLIT_tel_CNOT_X+SPLIT_tel_CNOT_Y-SPLIT],
                                       qubits_index[q+SPLIT_tel_CNOT_X+SPLIT_tel_CNOT_Y+SPLIT]]
                                      for q in qubits['x_tel'] + qubits['z_tel'])
    else:
        Bell_pairs_list = []
    noisy_circuit = NoiseModel.Standard(probas).noisy_circuit(circuit,
                                                              auto_push_prep=False,
                                                              auto_pull_mes=False,
                                                              auto_push_prep_bell=True,
                                                              auto_pull_mes_bell=True,
                                                              bell_pairs=Bell_pairs_list
                                                              )
    if dist_i == 5:
        with open("circuits/circuit.html", 'w') as file:
            print(circuit.diagram("interactive"), file=file)
        with open('circuits/diagram_with_error.svg', 'w') as f:
            print(noisy_circuit.diagram("timeline-svg"), file=f)
    return noisy_circuit
# %% Helper functions for collecting and plotting


def _collect_and_print(tasks, show_table=True,
                       fits=['general with error', 'general without error',
                             'slopes', 'slope+alpha', 'crossing asymptots'],
                       data_type='no_split', method='leastsq',
                       file='data_simu/last_simu.csv',
                       read_file=None,
                       **kwargs):
    """Collect and print tasks, kwargs is passed to sinter.collect.

    Also do the fit accordingly to the argument fits,data_type and method.
    """
    nb_shots = 3_000_000
    if read_file is not None:
        if os.path.exists(read_file):
            print('This data file already exist')
            kwargs = dict(num_workers=8, save_resume_filepath=read_file, max_shots=nb_shots,
                          max_errors=3000, tasks=tasks, decoders=['pymatching'], print_progress=True) | kwargs
        else:
            print('Create a new data file')
            kwargs = dict(num_workers=8, save_resume_filepath=read_file, max_shots=nb_shots, max_errors=3000,
                          tasks=tasks, decoders=['pymatching'], print_progress=True) | kwargs
    else:
        print('Create a new data file')
        kwargs = dict(num_workers=8, save_resume_filepath=file, max_shots=nb_shots, max_errors=3000,
                      tasks=tasks, decoders=['pymatching'], print_progress=True) | kwargs
    # Collect the samples (takes a few minutes).
    samples = sinter.collect(**kwargs)

    if show_table:
        # Print samples as CSV data.
        print(sinter.CSV_HEADER)
        for sample in samples:
            print(f"p_l={sample.errors/sample.shots:e},\t" +
                  ',\t'.join(f"{k}={v}" for k, v in sorted(
                      sample.json_metadata.items())))

    data_tofit, data = prepare_data(samples, data_type, fits)
    fit(nb_shots, data_tofit, data, fits, data_type, method)

    return samples


def _plot(samples: list[sinter.TaskStats],
          x_axis='p', x_label="Physical error rate", label="d={d}",
          title="Surface code", ylim=None, filename=None, filtered=False):
    """Plot the error rate.

    When filtered = True it only plot data with uncertainty less than quarter the value.
    label is deduced from stat.json_metadata
    """
    # Render a matplotlib plot of the data.
    fig, ax = plt.subplots(1, 1)
    if filtered:
        sinter.plot_error_rate(
            ax=ax,
            stats=samples,
            group_func=lambda stat: label.format_map(stat.json_metadata),
            filter_func=lambda stat: stat_error(
                stat.errors/stat.shots, stat.shots) < stat.errors/stat.shots/4,
            x_func=lambda stat: stat.json_metadata[x_axis],
            # highlight_max_likelihood_factor = 1
        )
    else:
        sinter.plot_error_rate(
            ax=ax,
            stats=samples,
            group_func=lambda stat: label.format_map(stat.json_metadata),
            x_func=lambda stat: stat.json_metadata[x_axis],
            # highlight_max_likelihood_factor = 1−1)z1 ZL1Z
        )
    ax.loglog()
    ax.set_ylim(ylim)
    ax.grid()
    ax.set_title(title)
    ax.set_ylabel("Logical error probability")
    ax.set_xlabel(x_label)
    ax.legend()
    # Save to file and also open in a window.
    if filename is not None:
        fig.savefig('figures/'+filename)
    plt.show()
    return fig, ax


def _plot_per_round(samples: list[sinter.TaskStats],
                    x_axis='p', x_label="Bulk error rate", label="d={d}",
                    title="Surface code", ylim=None, filename=None, filtered=False):
    """Plot the error rate per round.

    When filtered = True it only plot data with uncertainty less than quarter the value.
    label is deduced from stat.json_metadata
    """
    # Render a matplotlib plot of the data.
    fig, ax = plt.subplots(1, 1)
    if filtered:
        sinter.plot_error_rate(
            ax=ax,
            stats=samples,
            group_func=lambda stat: label.format_map(stat.json_metadata),
            filter_func=lambda stat: (stat_error(stat.errors/stat.shots, stat.shots,
                                                 rep=stat.json_metadata['k']) < to_error_per_round(
                stat.errors/stat.shots, stat.json_metadata['k'])/4 and stat.json_metadata['p'] < 3e-2),

            x_func=lambda stat: stat.json_metadata[x_axis],
            failure_units_per_shot_func=lambda stats: stats.json_metadata['k']
            # highlight_max_likelihood_factor = 1
        )
    else:
        sinter.plot_error_rate(
            ax=ax,
            stats=samples,
            group_func=lambda stat: label.format_map(stat.json_metadata),
            x_func=lambda stat: stat.json_metadata[x_axis],
            failure_units_per_shot_func=lambda stats: stats.json_metadata['k']
            # highlight_max_likelihood_factor = 1
        )
    ax.loglog()
    ax.set_ylim(ylim)
    ax.grid()
    ax.set_title(title)
    ax.set_ylabel("Logical error probability")
    ax.set_xlabel(x_label)
    ax.legend()
    # Save to file and also open in a window.
    if filename is not None:
        fig.savefig('figures/'+filename)
    plt.show()
    return fig, ax


def _collect(tasks, file, **kwargs):
    """Collect  tasks, kwargs is passed to sinter.collect."""
    nb_shots = 3_000_000
    if os.path.exists(file):
        print('This data file already exist')
        kwargs = dict(num_workers=8, save_resume_filepath=file, max_shots=nb_shots, max_errors=1000,
                      tasks=tasks, decoders=['pymatching'], print_progress=True) | kwargs
    else:
        print('Create a new data file')
        kwargs = dict(num_workers=8, save_resume_filepath=file, max_shots=nb_shots, max_errors=1000,
                      tasks=tasks, decoders=['pymatching'], print_progress=True) | kwargs
    # Collect the samples (takes a few minutes).
    samples = sinter.collect(**kwargs)
    return (samples)

# %% Threshold estimation Regular surface code


def generate_example_tasks(kind, rep, probas):
    """Generate surface code circuit without splitting tasks using Stim's circuit generation.

    For regular surface code."""
    # As a reminder, Probas=['Hadam','idle_data','idle_bell','depol', 'prep', 'mes', 'bell']
    for p in tqdm([0.0001, 0.0003, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003,
                   0.004, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3], desc='Outer loop'):
        for d in tqdm([3, 5, 7, 9, 11], desc='Inner loop', leave=False):
            yield sinter.Task(
                circuit=gen_memory(d, d, rep(d), kind, None,
                                   multiply_namedtuple(probas, p)),
                json_metadata={
                    'p': p,
                    'd': d,
                    'k': rep(d)
                },
            )


def generate_example_tasks_critical(kind, rep, probas):
    """Generate surface code circuit without splitting tasks using Stim's circuit generation.

    Made for the critical order ansatz valid around threshold for O(d) cycle.
    """
    for p in tqdm([4.6e-3, 4.7e-3, 4.8e-3, 4.9e-3, 4.55e-3, 4.65e-3, 4.75e-3, 4.85e-3, 4.95e-3,
                   0.0050, 0.0051, 0.0052, 5.3e-3, 0.0054, 0.00505, 0.00515, 0.00525, 5.35e-3,
                   0.00545, 5.5e-3, 0.0056, 0.0057, 0.0058, 0.0059, 0.0060], desc='Outer loop'):
        for d in tqdm([7, 9, 11, 13, 15], desc='Inner loop', leave=False):
            yield sinter.Task(
                circuit=gen_memory(d, d, rep(d), kind, None,
                                   multiply_namedtuple(probas, p)),
                json_metadata={
                    'p': p,
                    'd': d,
                    'k': rep(d)
                },
            )


def surface_code_threshold(kind='x', rep=nbr_cycle, probas=Probas(1, 1, 1, 1, 1, 1, 0), filtered=True, read_file=None):
    """Regular surface code."""
    samples = _collect_and_print(generate_example_tasks(kind, rep, probas), fits=['general with error'
                                                                                  ], data_type='no split',
                                 read_file=read_file)
    # Plot the logical error rate per rep cycle
    _plot(samples, title=" ",
          filename="surface_code_threshold_all_cycle" + kind + ".pdf", filtered=filtered)
    # Plot the logical error rate per cycle
    _plot_per_round(samples, title=r"  ",
                    filename="surface_code_threshold_per_round" + kind + ".pdf", filtered=filtered)


def surface_code_threshold_critical(kind='x', rep=nbr_cycle, probas=Probas(1, 1, 1, 1, 1, 1, 0)):
    """Regular surface code, critical order ansatz."""
    samples = _collect_and_print(generate_example_tasks_critical(kind, rep, probas), fits=[
        'critical exponent'], data_type='no split')
    # Plot the logical error rate per rep cycle
    _plot(samples, title="Logical error per d round",
          filename="surface_code_threshold_all_cycle" + kind + ".pdf", filtered=False)
    # Plot the logical error rate per cycle
    _plot_per_round(samples, title="Logical error per round ",
                    filename="surface_code_threshold_per_round" + kind + ".pdf", filtered=False)

# %% Threshold estimation of single seam Splitted surface code

# %%% Fixed noise on Bell pair


def generate_telemesures_tasks(kind, p_bell, version, rep):
    """Generate the circuit for splitted surface code at fixed p_bell.

    If version is 1 it uses teleported cnot, if version is 2 it uses measurement teleportation.
    Version 2 is the default version.
    """
    for p in tqdm([1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 1e-2,
                   5e-2, 7e-2, 9e-2, 1e-1], desc='Outer loop'):
        if version == 1:
            for d in tqdm([3, 5, 7, 9, 11], desc='Inner loop', leave=False):
                yield sinter.Task(
                    circuit=gen_memory(d, d, rep(d), kind, d+2,
                                       Probas(p, p, p, p, p, p, p_bell)),
                    json_metadata={
                        'p': p,
                        'd': d,
                        'p_bell': p_bell,
                        'k': rep(d)
                    },
                )
        if version == 2:
            for d in tqdm([3, 5, 7, 9, 11], desc='Inner loop', leave=False):
                yield sinter.Task(
                    circuit=gen_memory(d, d, rep(d), kind, d+1,
                                       Probas(p, p, p, p, p, p, p_bell)),
                    json_metadata={
                        'p': p,
                        'd': d,
                        'p_bell': p_bell,
                        'k': rep(d)
                    },
                )


def splitted_surface_code_pseudothreshold(kind='z', p_bell=1e-1, version=2, rep=nbr_cycle, fits=[
        'correlated contributions'], read_file=None):
    """Sample circuits of splitted surface code at fixed p_bell."""
    if read_file is None:
        samples = _collect_and_print(generate_telemesures_tasks(
            kind, p_bell, version, rep), fits=fits, data_type='p_bell fixed')
    else:
        samples = _collect_and_print(generate_telemesures_tasks(
            kind, p_bell, version, rep), fits=fits, data_type='p_bell fixed', read_file=read_file)
    _plot(samples, x_axis='p', x_label="Bulk error rate",
          title=f"Splitted Surface Code ; kind={kind} ; p_bell={p_bell}; rep={rep}",
          filename="splitted_surface_code_p_bell_fixed_" + kind + "_version_" + str(version) + ".pdf")
    # Plot the logical error rate per cycle
    _plot_per_round(samples, title=r"$p_{\text{Bell}}$="+f"{p_bell}",
                    filename="splitted_surface_code_per_round_pb_bell_fixed" + kind + ".pdf", filtered=True)


# %%% bell pair noise proportional to physical noise p_bell = ratio * p_bulk

def generate_telemesures_tasks2(kind, ratio, version, rep):
    """Do the circuits for splitted surface code with p_bell=ratio* p_bulk."""
    # As a reminder, Probas=['Hadam','idle_data','idle_bell','depol', 'prep', 'mes', 'bell']
    for p in tqdm([5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2,
                   5e-2, 7e-2, 9e-2, 1e-1], desc='Outer loop'):
        if ratio * p >= 1:
            continue
        if version == 1:
            for d in tqdm([3, 5, 7, 9, 11], desc='Inner loop', leave=False):
                yield sinter.Task(
                    circuit=gen_memory(d, d, rep(d), kind, d+2,
                                       Probas(p, p, p, p, p, p, ratio*p)),
                    json_metadata={
                        'p': p,
                        'd': d,
                        'p_bell': ratio*p,
                        'k': rep(d)
                    },
                )
        if version == 2:
            for d in tqdm([3, 5, 7, 9, 11], desc='Inner loop', leave=False):
                yield sinter.Task(
                    circuit=gen_memory(d, d, rep(d), kind, d+1,
                                       Probas(p, p, p, p, p, p, ratio*p)),
                    json_metadata={
                        'p': p,
                        'd': d,
                        'p_bell': ratio*p,
                        'k': rep(d)
                    },
                )


def splitted_surface_code_pseudothreshold2(kind='z', ratio=10, version=2, rep=nbr_cycle):
    """Sample the logical error rate of a splitted surface code.

    with a ratio between p_bell and p.
    """
    samples = _collect_and_print(generate_telemesures_tasks2(
        kind, ratio, version, rep), fits=['general with error'], data_type='ratio')
    _plot(samples, x_label="Physical error p ; p_bell=ratio*p",
          title=f"Splitted surface code ; kind={kind} ; ratio={ratio}",
          filename=f"splitted_surface_code_{kind}_ratio_{ratio}.pdf")
    # Plot the logical error rate per cycle
    _plot_per_round(samples, title=f"Surface code error per round, {kind}",
                    filename=f"splitted_surface_code_per_round_ratio_{ratio}.pdf")

# %%% Fixed noise in the Bulk


def generate_telemesures_tasks3(kind, p, version, rep):
    """Do the circuit for splitted surface code with fixed p_bulk."""
    # As a reminder, Probas=['Hadam','idle_data','idle_bell','depol', 'prep', 'mes', 'bell']
    p_bell_list = np.logspace(-3, -1, 15)
    for p_bell in tqdm(p_bell_list, desc='Outer loop'):
        if p_bell >= 1:
            continue
        if version == 1:
            for d in tqdm([3, 5, 7, 9, 11], desc='Inner loop', leave=False):
                yield sinter.Task(
                    circuit=gen_memory(d, d, rep(d), kind, d,
                                       Probas(p, p, p, p, p, p, p_bell)),
                    json_metadata={
                        'd': d,
                        'p_bell': p_bell,
                        'p': p,
                        'k': rep(d)
                    },
                )
        if version == 2:
            for d in tqdm([3, 5, 7, 9, 11, 13], desc='Inner loop', leave=False):
                yield sinter.Task(
                    circuit=gen_memory(d, d, rep(d), kind, d+1,
                                       Probas(p, p, p, p, p, p, p_bell)),
                    json_metadata={
                        'd': d,
                        'p_bell': p_bell,
                        'p': p,
                        'k': rep(d)
                    },
                )


def splitted_surface_code_pseudothreshold3(kind='z', p=1e-3, version=2, rep=nbr_cycle, fits=[
        'correlated contributions'], read_file=None):
    """Sample the logical error rate of splitted surface code with fixed p_bulk."""
    if read_file is None:
        samples = _collect_and_print(generate_telemesures_tasks3(kind, p, version, rep), fits=fits,
                                     data_type='p fixed')
    else:
        samples = _collect_and_print(generate_telemesures_tasks3(kind, p, version, rep), fits=fits,
                                     data_type='p fixed', read_file=read_file)
    _plot(samples, x_axis='p_bell', x_label="Bell pairs preparation error rate",
          title=" ",
          filename=f"splitted_surface_code_p_fixed_all_cycle_{kind}_{p}.pdf", filtered=True)
    _plot_per_round(samples, x_axis='p_bell', x_label="Bell pairs preparation error rate",
                    title=" ",
                    filename=f"splitted_surface_code_p_fixed_{kind}_{p}_per_round.pdf", filtered=True)


def generate_telemesures_tasks_naive_split(kind, p, version=1, rep=nbr_cycle):
    """Do the circuit for splitted surface code with fixed p_bulk."""
    # As a reminder, Probas=['Hadam','idle_data','idle_bell','depol', 'prep', 'mes', 'bell']
    p_bell_list = np.logspace(-3, -1, 15)
    for p_bell in tqdm(p_bell_list, desc='Outer loop'):
        if p_bell >= 1:
            continue
        if version == 1:
            for d in tqdm([3, 5, 7, 9, 11], desc='Inner loop', leave=False):
                yield sinter.Task(
                    circuit=gen_memory(d, d, rep(d), kind, d+2,
                                       Probas(p, p, p, p, p, p, p_bell), naive_seam=True),
                    json_metadata={
                        'd': d,
                        'p_bell': p_bell,
                        'p': p,
                        'k': rep(d)
                    },
                )
        if version == 2:
            print('naive split only has version 1 !')


def splitted_surface_code_pseudothreshold_naive_split(kind='z', p=1e-3,
                                                      fits=['slopes'], read_file=None):
    """Sample the logical error rate of splitted surface code with fixed p_bulk."""
    if read_file is None:
        samples = _collect_and_print(generate_telemesures_tasks_naive_split(kind, p),
                                     fits=fits, data_type='p fixed')
    else:
        samples = _collect_and_print(generate_telemesures_tasks_naive_split(kind, p),
                                     fits=fits, data_type='p fixed', read_file=read_file)
    _plot_per_round(samples, x_axis='p_bell', x_label="Bell pairs preparation error rate",
                    title=" ",
                    filename=f"splitted_surface_code_naive_{kind}_{p}_per_round.pdf", filtered=True)


# %%% Circuit for 3D fit, free p_bulk and p_bell.
def generate_telemesures_tasks_3d(P_bulk, P_seam, kind, rep):
    """Genere les circuits pour faire le fit 3d."""
    for p in tqdm(P_bulk, desc='Loop on p_bulk'):
        for p_bell in tqdm(P_seam, desc='Inner loop on p_bell', leave=False):
            for d in [3, 5, 7, 9, 11, 13]:
                yield sinter.Task(
                    circuit=gen_memory(d, d, rep(d), kind, d+1,
                                       Probas(p, p, p, p, p, p, p_bell)),
                    json_metadata={
                        'd': d,
                        'p_bell': p_bell,
                        'p': p,
                        'k': rep(d)
                    },
                )

# %%Several seams threshold estimation


def generate_telemesures_tasks_multiple_seam(kind, p_bell, rep):
    """Do the circuits for fixed p_bell and two seam rectangular splitted surface code.

    Reproduce the spacing of the architecture layout for two seam.
    [2*d+2, 6*d+6] is the list of index of interfaces (even numbers).
    """
    for p in tqdm([1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 1e-2,
                   5e-2, 7e-2, 9e-2, 1e-1], desc='Outer loop'):
        for d in tqdm([3, 5, 7, 9], desc='Inner loop', leave=False):
            yield sinter.Task(
                circuit=gen_memory(d, 4*d+3, rep(d), kind, [2*d+2, 6*d+6],
                                   Probas(p, p, p, p, p, p, p_bell)),
                json_metadata={
                    'p': p,
                    'd': d,
                    'p_bell': p_bell,
                    'k': rep(d)
                },
            )


def generate_telemesures_tasks_multiple_seam2(kind, p, rep, visual_check=False):
    """Do the circuits for fixed p_bulk and two seam rectangular splitted surface code.

    Reproduce the spacing of the architecture layout for two seam.
    [2*d+2, 6*d+6] is the list of index of interfaces (even numbers).
    If visual_check is true, it shows the stabilizers to check visually the layout.
    """
    if visual_check:
        # First do a few plots of the stabilizers
        for d in [3, 5, 7, 9, 11]:
            seam_cols = [2*d+2, 6*d+6]
            length = 4*d+3
            data_qubits, x_stabs, z_stabs = surf_qubits_stabs(d, length)
            data_qubits2, x_stabs2, z_stabs2, x_tel2, z_tel2 = split_surf_code(
                data_qubits, x_stabs, z_stabs, seam_cols)
            qubits_list2, qubits_index2, qubits_dict2, qubits_id_dict2 = _prepare_ids(
                data_qubits2, x_stabs2, z_stabs2, x_tel2, z_tel2)
            plot_stabs(data_qubits2, x_stabs2, z_stabs2, x_tel2, z_tel2, convention='ij')
    for p_bell in tqdm([3e-3, 5e-3, 7e-3, 9e-3, 1e-2, 1.5e-2,
                       2e-2, 3e-2, 4e-2, 5e-2, 0.1], desc='Outer loop'):
        for d in tqdm([3, 5, 7, 9, 11], desc='Inner loop', leave=False):
            yield sinter.Task(
                circuit=gen_memory(d, 4*d+3, rep(d), kind, [2*d+2, 6*d+6],
                                   Probas(p, p, p, p, p, p, p_bell)),
                json_metadata={
                    'p': p,
                    'd': d,
                    'p_bell': p_bell,
                    'k': rep(d)
                },
            )


def multiple_seam_surface_code_pseudothreshold(kind='z', p_bell=None, p=None, rep=nbr_cycle,
                                               read_file=None):
    """Sample the logical error rate for splitted surface code with two seams."""
    # When p_bell is fixed
    if p is None:
        if p_bell is not None:
            samples = _collect_and_print(generate_telemesures_tasks_multiple_seam(kind, p_bell, rep), fits=[
            ], data_type='p_bell fixed', read_file=read_file)
            _plot_per_round(samples, x_axis='p', x_label="Physical bulk error rate",
                            title=" ",
                            filename=f"Multiple_split_surface_code_per_round_p_bell_{p_bell}.pdf", filtered=True,
                            )

        else:
            print('Either Bulk or seam error rate should be fixed !')
    # When p is fixed
    if p_bell is None:
        if p is not None:
            if read_file is None:
                samples = _collect_and_print(generate_telemesures_tasks_multiple_seam2(kind, p, rep), fits=[
                ], data_type='p fixed')
            if read_file is not None:
                samples = sinter.read_stats_from_csv_files(read_file)
            _plot_per_round(samples, x_axis='p_bell', x_label="Bell pair error rate",
                            title=f"Two seam patch of size d x 4d+3 with seam in d+1, 3d+3 and p={p} ",
                            filename=f"Multiple_split_surface_code_per_round_p_{p}.pdf", filtered=True,
                            )
            data_to_plot, _ = prepare_data(samples, data_type='p fixed')
            plot_data_vs_model(data_to_plot, data_type='p fixed')
        else:
            print('Either Bulk or seam error rate should be fixed !')

# %% Fonction to call


def surface_code_pseudotreshold(kind, data_type, version=2,
                                ratio=10, p_fixed=1e-3, p_bell_fixed=1e-1,
                                probas=Probas(1, 1, 1, 1, 1, 1, 0), fits=[
                                    'correlated contributions'],
                                read_file=None):
    """Single function to call to do a simulation."""
    if data_type == 'no split':
        surface_code_threshold(kind=kind, probas=probas)
    if data_type == 'ratio':
        splitted_surface_code_pseudothreshold2(kind=kind, ratio=ratio, version=version)
    if data_type == 'p fixed':
        splitted_surface_code_pseudothreshold3(
            kind=kind, p=p_fixed, version=version, fits=fits, read_file=read_file)
    if data_type == 'naive split p fixed':
        splitted_surface_code_pseudothreshold_naive_split(
            kind=kind, p=p_fixed, read_file=read_file)
    if data_type == 'p_bell fixed':
        splitted_surface_code_pseudothreshold(kind=kind, p_bell=p_bell_fixed, version=version,
                                              fits=fits, read_file=read_file)


def plot_error_model(kind, d):
    """Plot the matching graph of the code."""
    circuit = gen_memory(d, d, d, kind, d+1)
    circuit2 = gen_memory(d, d, d, kind, 0)
    circuit3 = gen_memory(d, d, d, kind, d+2)
    circuit.to_file("Circuit_test_base.txt")
    circuit.diagram("matchgraph-3d")
    dem = circuit.detector_error_model()
    with open('matching_graph/matching_graph_w_bell.svg', 'w') as f:
        print(circuit.diagram("matchgraph-svg"), file=f)
        circuit.diagram("matchgraph-3d-html")
    with open('matching_graph/matching_graph_w_bell_tel_cnot.svg', 'w') as f:
        print(circuit3.diagram("matchgraph-svg"), file=f)
        circuit3.diagram("matchgraph-3d-html")
    with open('matching_graph/matching_graph_no_bell.svg', 'w') as f:
        print(circuit2.diagram("matchgraph-svg"), file=f)
    with open('matching_graph/matching_graph_dem_with_bell.gltf', 'w') as f:
        print(dem.diagram("matchgraph-3d"), file=f)
    with open('matching_graph/matching_graph_circ_with_bell.gltf', 'w') as f:
        print(circuit.diagram("matchgraph-3d"), file=f)

# %% Fit 3D


def _3d_fit_sample(p_bulk_min, p_bulk_max, p_seam_min, p_seam_max, kind='z', lattice_size=10,
                   rep=nbr_cycle,
                   file_path='data_simu/last_simu.csv',
                   read_file=None,
                   fit_interval=(0., 1., 0., 1., 100.),
                   fit_3D_type=['naive']):
    """Perform a 3d fit of the naive ansatz over the 3d space.

    Every parameters of the ansatz are fit
    With logical error, bulk error and seam errors as axis.
    fit_interval est (pmin,pmax,psmin,psmax, ratio_std_err) definissant sur quelles datas le fit
    est fait.
    """
    # # Creates the space of points to be covered
    # P_bulk = np.linspace(p_bulk_min, p_bulk_max, lattice_size)
    # P_seam = np.linspace(p_seam_min, p_seam_max, lattice_size)
    # Generate log-scaled array
    P_bulk = np.logspace(np.log10(p_bulk_min), np.log10(p_bulk_max), num=lattice_size, base=10)
    P_seam = np.logspace(np.log10(p_seam_min), np.log10(p_seam_max), num=lattice_size, base=10)
    # collect the result from circuit sampling with sinter
    if read_file is None:
        samples = _collect(generate_telemesures_tasks_3d(P_bulk, P_seam, kind, rep),
                           file=file_path)
    else:
        samples = sinter.read_stats_from_csv_files(read_file)
    # Prepare the data to be fit
    data_tofit, data = prepare_data(samples, data_type='3d', fits=[])
    data_tofit, out = fit_3D(data_tofit, data, P_bulk, P_seam,  fit_interval, fit_3D_type)
    return data_tofit, out, P_bulk, P_seam

# %% Storing of the last simulation
# For convenience we always store the last simulation in a temporary file, here we delete it
# before the next simulation


def del_last_simu():
    """Delete the data of last simu."""
    file_path = 'data_simu/last_simu.csv'
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'{file_path} has been deleted.')
    else:
        print(f'{file_path} does not exist.')


# %% Partie exécutable
# NOTE: This is actually necessary! If the code inside 'main()' was at the
# module level, the multiprocessing children spawned by sinter.collect would
# also attempt to run that code.
if __name__ == '__main__':

    del_last_simu()
    # Uncomment to run a simulation

    # Regular rotated surface code simulation

    # surface_code_threshold(kind='z', rep=lambda x: 3*x,
    #                        probas=Probas(1, 1, 1, 1, 1, 1, 1), filtered=True,
    #                        read_file='data_simu/regular_surface_code.csv')

    # Regular rotated surface code simulation and fit of a critical order ansatz

    # surface_code_threshold_critical(kind='z', rep=nbr_cycle, probas=Probas(1, 1, 1, 1, 1, 1, 0))

    # Naive splitting with halved distance

    surface_code_pseudotreshold(kind='z', data_type='naive split p fixed', version=1, p_fixed=0e-3,
                                read_file='data_simu/naive_seam_halved_d.csv')
    # Splitted surface code with one interface (as fig.2) with bell pair noise fixed

    # surface_code_pseudotreshold(kind='z', data_type='p_bell fixed',
    #                             version=2, p_bell_fixe=0.)

    # Splitted surface code with one interface (as fig.2) with p or p_bell fixed

    # surface_code_pseudotreshold(kind='z', data_type='p fixed', version=2, p_fixed=1e-3, fits=[
    #     'correlated contributions full'],
    #     read_file='data_simu/dx2d_one_seam_d=3-15.txt')

    # surface_code_pseudotreshold(kind='z', data_type='p_bell fixed', version=2, p_bell_fixed=1e-2)

    # Full 3D fit on several values of p and p_bell with the naive ansatz

    # data_tofit, out, P_bulk, P_seam = _3d_fit_sample(
    #     1e-4, 1e-3, 1e-3, 1e-1, lattice_size=22,
    #     read_file='data_simu/sampling_3d_v15.csv',
    #     fit_interval=(5e-4, 1e-3, 0., 5e-2, 0.5),
    #     fit_3D_type=['naive'])

    # Full 3D fit on several values of p and p_bell with the full ansatz

    # data_tofit, out, P_bulk, P_seam = _3d_fit_sample(
    #     1e-4, 1e-3, 1e-3, 1e-1, lattice_size=22,
    #     read_file='data_simu/sampling_3d_v15.csv',
    #     fit_interval=(5e-4, 1e-3, 0., 5e-2, 0.5),
    #     fit_3D_type=['full'])

    # Simulation of a 2 interface rectangular patch reproducing the layout spacing
    # verification of the validity of the logical error rate fitted above in the multi-seam setting.

    # multiple_seam_surface_code_pseudothreshold(kind='z', p=1e-3,
    #                                            read_file='data_simu/simu_two_seam_4d+3_d_3d.csv',
    #                                            rep=nbr_cycle)

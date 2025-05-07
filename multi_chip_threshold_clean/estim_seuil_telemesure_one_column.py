#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimation du seuil d'un code de répétition avec télémesure.

Je suis parti d'une copie de l'exemple de sinter.

NOTE: le modèle de bruit actuel est que le bruit apparait sur les portes.
Il faudra voir si on veut pas aussi prendre en compte un bruit qui apparait
également sur les qubits inactifs. Dans le cœur du code de surface normalement
il n'y en a pas, mais au bord ça pourrait changer quelque-chose qu'on fasse en
sorte de ne pas laisser trainer la mesure des stabilisateurs impliquant juste 2
qubits. Avec le schéma de télémesure, je crains qu'on ait aussi les qubits
auxiliaires pas tout le temps occupés car on est limité par la disponibilité
des qubits de données.

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
from lmfit import Parameters, minimize, report_fit
from IPython.display import clear_output
from scipy.interpolate import griddata
import matplotlib.ticker as mticker
from Coordinates import Coord, SPLIT_DIST
from noise_v2 import NoiseModel, Probas
from fitting import fit, prepare_data, stat_error, getPairs, coupled_error_model, coupled_error_model_v2, to_error_per_round


SPLIT = Coord(0, SPLIT_DIST)


def multiply_namedtuple(namedtuple_instance, multiplier):
    """Multiply a namedtuple by a float."""
    return namedtuple_instance._replace(
        **{field: getattr(namedtuple_instance, field) * multiplier for field in namedtuple_instance._fields})


def sum_namedtuples(*tuples):
    """Sum several named tuples."""
    if not tuples:
        return None

    TupleType = type(tuples[0])
    fields = tuples[0]._fields

    summed_values = [sum(getattr(t, field) for t in tuples) for field in fields]

    return TupleType(*summed_values)


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


def myfunction(progress):
    """For the progress bar."""
    clear_output(wait=True)
    print(progress.status_message)


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


def split_surf_code(data_qubits, x_stabs, z_stabs, split_col, x_tel={}, z_tel={}):
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
        if split_col % 2 == 1:
            if any(point.j == split_col for point in data_qubits):
                raise ValueError(
                    "Only measurements can be splitted, check positions.")
            x_tel = {key: [p for p in val if p.j == split_col]
                     for key, val in x_stabs.items()}
            z_tel = {key: [p for p in val if p.j == split_col]
                     for key, val in z_stabs.items()}
            # HINT: if I come to performance issues, I better store points in sets
            # rather than in lists, but I keep this for "if needed" (especially as
            # order is important latter). Also doing only one iteration and testing on
            # p.j == split_col would be faster (but less reliable if latter I change
            # to strange shapes).
            x_stabs = {key: [p for p in val if p not in x_tel[key]]
                       for key, val in x_stabs.items()}
            z_stabs = {key: [p for p in val if p not in z_tel[key]]
                       for key, val in z_stabs.items()}
        if split_col % 2 == 0:
            for key, val in x_stabs.items():
                if any(point.j == split_col for point in x_stabs[key]):
                    raise ValueError(
                        "Only data can be splitted in v2, check positions.")
            x_tel = {key: [p for p in val if (p.j == split_col+1 or p.j == split_col-1)]
                     for key, val in x_stabs.items()}
            z_tel = {key: [p for p in val if (p.j == split_col+1 or p.j == split_col-1)]
                     for key, val in z_stabs.items()}
            # HINT: if I come to performance issues, I better store points in sets
            # rather than in lists, but I keep this for "if needed" (especially as
            # order is important latter). Also doing only one iteration and testing on
            # p.j == split_col would be faster (but less reliable if latter I change
            # to strange shapes).
            x_stabs = {key: [p for p in val if p not in x_tel[key]]
                       for key, val in x_stabs.items()}
            z_stabs = {key: [p for p in val if p not in z_tel[key]]
                       for key, val in z_stabs.items()}
    return data_qubits, x_stabs, z_stabs, x_tel, z_tel


def _double_points(points):
    """Split each of the points."""
    return sum([[p-SPLIT, p+SPLIT] for p in points], [])


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
    # circuit.append({'0': "R", '+': "RX"}[state], qubits_id['data'])
    # We consider we can't prepare + :
    circuit.append("R", qubits_id['data'])
    # Initialisation in 0 of the stabilisers ancilary qubits
    circuit.append("R", qubits_id['stabs'])
    if state == '+':
        circuit.append("H", qubits_id['data'])
    return circuit


def _surface_code_cycle(dist_i, dist_j, data_qubits, x_stabs, z_stabs,
                        x_tel=None, z_tel=None, split_col_list=None):
    """Un cycle de correction d'erreur, du H aux mesures/réinitialisation.

    Mais sans les détecteurs (car ils changent entre le premier cycle et les
    autres).

    Adapté aussi bien au code de surface normal qu'au code de surface splitté.
    """
    # Si on a donné une liste on regarde la parité du premier élément pour savoir quel splitting faire
    if isinstance(split_col_list, list):
        split_col = split_col_list[0]
    else:
        split_col = split_col_list
    teleport = x_tel is not None or z_tel is not None
    # Prepare data
    qubits_list, qubits_index, qubits, qubits_id = _prepare_ids(
        data_qubits, x_stabs, z_stabs, x_tel, z_tel)
    # Create the circuit
    circuit = stim.Circuit()

    circuit.append("TICK")
    # Prepare the X ancilla in the correct base.
    circuit.append("H", qubits_id['x'])  # No error as virtual gate
    # Prepare the Bell ancillas (considering they are already initialised at 0)
    # To be check

    if teleport:
        circuit.append("H", [qubits_index[q-SPLIT]
                             for q in qubits['x_tel'] + qubits['z_tel']])

        circuit.append("CX",
                       flatten([qubits_index[q-SPLIT], qubits_index[q+SPLIT]]
                               for q in qubits['x_tel'] + qubits['z_tel']))
    circuit.append("TICK")
    # Note : we implement here https://arxiv.org/abs/1404.3747

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

    # First way of splitting
            elif x in qubits['z_tel'] and x + dirr_z in data_qubits and split_col % 2 == 1:
                cnot_args.extend([qubits_index[x + dirr_z],
                                  qubits_index[x + dirr_z[1]*SPLIT]])

            elif x in qubits['x_tel'] and x + dirr_x in data_qubits and split_col % 2 == 1:
                cnot_args.extend([qubits_index[x + dirr_x[1]*SPLIT],
                                  qubits_index[x + dirr_x]])

    # Nicolas's way of splitting
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
        circuit_temp.append("CX", cnot_args)
        circuit += circuit_temp
        # if dirr_x == (-1, 1) or dirr_z == (-1, 1) or dirr_x == (1, 1) or dirr_z == (-1, -1) or dirr_x == (1, -1) or dirr_z == (1, -1):
        #     circuit.append("DEPOLARIZE2", cnot_args, 0)
        circuit.append("TICK")
    # Rotate basis because we only mesure in Z basis
    circuit.append("H", qubits_id['x'] + qubits_id['x_tel_split'])
    # Do the measures
    circuit.append("TICK")
    circuit.append("MR", qubits_id['stabs'])
    return circuit


def _add_surface_code_detectors(bloc, qubits, nb_stabs, x=True, z=True,
                                first=False, time=0):
    """Add detectors of the surface code.

    Assumes that the stabilizers where measured in the orders of qubits['stabs']
    """
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
                       time=1):
    """Add the final measurements and corresponding detectors."""
    kind = kind.casefold()
    if kind not in ('x', 'z'):
        raise ValueError("'kind' must be 'x' or 'z'!")
    circuit.append("TICK")
    circuit.append({'x': "MX", 'z': 'M'}[kind], qubits_id['data'])
    # Vérification des stabiliseurs à la main après mesure
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
               probas=Probas(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), plot=False):
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
            data_qubits, x_stabs, z_stabs, split_col)
    if plot:
        plot_stabs(data_qubits, x_stabs, z_stabs, x_tel, z_tel,
                   convention='ij')
    qubits_list, qubits_index, qubits, qubits_id = _prepare_ids(
        data_qubits, x_stabs, z_stabs, x_tel, z_tel)
    # Useful for computing indexes in measurement record.
    nb_data, nb_stabs = len(qubits['data']), len(qubits['stabs'])
    # Initialize circuit and qubits
    circuit = _initialize_circuit({'x': '+', 'z': '0'}[kind],
                                  qubits_list, qubits_id)
    # First cycle
    circuit += _surface_code_cycle(dist_i, dist_j, data_qubits, x_stabs, z_stabs, x_tel, z_tel,
                                   split_col)
    _add_surface_code_detectors(circuit, qubits, nb_stabs,
                                x=(kind == 'x'), z=(kind == 'z'), first=True)
    # Generic cycle
    bloc = _surface_code_cycle(dist_i, dist_j, data_qubits, x_stabs, z_stabs, x_tel, z_tel,
                               split_col)
    bloc.append("SHIFT_COORDS", [], (0, 0, 1))
    _add_surface_code_detectors(bloc, qubits, nb_stabs)
    circuit.append(stim.CircuitRepeatBlock(repeat, bloc))
    # Mesure finale et assertion résultat
    _add_final_measure(circuit, kind, nb_data, nb_stabs, qubits, qubits_id,
                       time=1)
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
    with open("circuits/circuit.html", 'w') as file:
        print(circuit.diagram("interactive"), file=file)
    if dist_i == 3 and dist_j == 3:
        with open('circuits/diagram_without_error.svg', 'w') as f:
            print(circuit.diagram("timeline-svg"), file=f)
            # print(repr(circuit))
    # Prepare the list of qubits involved in bell pairs
    Bell_pairs_list = flatten([qubits_index[q-SPLIT], qubits_index[q+SPLIT]]
                              for q in qubits['x_tel'] + qubits['z_tel'])
    noisy_circuit = NoiseModel.Standard(probas).noisy_circuit(circuit,
                                                              auto_push_prep=False,
                                                              auto_pull_mes=False,
                                                              auto_push_prep_bell=True,
                                                              auto_pull_mes_bell=True,
                                                              bell_pairs=Bell_pairs_list)
    # with open('circuits/diagram_debug.svg', 'w') as f:
    #    print(circuit.diagram("timeline-svg"), file=f)
    if dist_i == 3 and dist_j == 3:
        with open('circuits/diagram_with_error.svg', 'w') as f:
            print(noisy_circuit.diagram("timeline-svg"), file=f)
    return noisy_circuit
# %% Helper functions for collecting and plotting


def _collect_and_print(tasks, show_table=True,
                       fits=['general with error', 'general without error',
                             'slopes', 'slope+alpha', 'crossing asymptots'],
                       data_type='no_split', method='leastsq',
                       file='/home/hjacinto/multi_chip_threshold/data_simu/last_simu.txt',
                       **kwargs):
    """Collect and print tasks, kwargs is passed to sinter.collect."""
    nb_shots = 3_000_000
    kwargs = dict(num_workers=8, save_resume_filepath=file, max_shots=nb_shots, max_errors=1000,
                  tasks=tasks, decoders=['pymatching'], print_progress=True) | kwargs
    # ,progress_callback=myfunction can be added
    # Collect the samples (takes a few minutes).
    samples = sinter.collect(**kwargs)

    if show_table:
        # Print samples as CSV data.
        print(sinter.CSV_HEADER)
        for sample in samples:
            print(f"p_l={sample.errors/sample.shots:e},\t" +
                  ',\t'.join(f"{k}={v}" for k, v in sorted(
                      sample.json_metadata.items())))

        # Print summary for Nicolas
        print('')

    data_tofit, data = prepare_data(samples, data_type, fits)
    fit(nb_shots, data_tofit, data, fits, data_type, method)

    return samples


def _plot(samples: list[sinter.TaskStats],
          x_axis='p', x_label="Physical error rate", label="d={d}",
          title="Surface code", ylim=None, filename=None, filtered=False):
    """Fait le travail de dessiner.

    label est formaté à partir du dictionnaire stat.json_metadata
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
                    title="Surface code", ylim=None, filename=None, filtered=False, rep=nbr_cycle):
    """Fait le travail de dessiner avec un taux d'erreur logique par round.

    label est formaté à partir du dictionnaire stat.json_metadata
    """
    # Render a matplotlib plot of the data.
    fig, ax = plt.subplots(1, 1)
    if filtered:
        sinter.plot_error_rate(
            ax=ax,
            stats=samples,
            group_func=lambda stat: label.format_map(stat.json_metadata),
            filter_func=lambda stat: (stat_error(stat.errors/stat.shots, stat.shots, rep=stat.json_metadata['k']) < to_error_per_round(
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


def _plot_error_per_nb_round(samples: list[sinter.TaskStats],
                             x_axis='k', x_label="Number of round", label="d={d}",
                             title="Surface code error per round", ylim=None, filename=None):
    """Fait le travail de dessiner pour le plot erreur en fonction du nombre de round.

    label est formaté à partir du dictionnaire stat.json_metadata
    """
    # Render a matplotlib plot of the data.
    fig, ax = plt.subplots(1, 1)
    p_L = []
    for stat in samples:
        if stat.json_metadata['k'] == 2:
            p_L += [[stat.errors/stat.shots, stat.json_metadata['d']]]
    for elem in p_L:
        x = range(2, 2*elem[1], 2)
        y = []
        for k in x:
            y += [elem[0]*(k/2)]
        ax.plot(x, y, color='black')
    sinter.plot_error_rate(
        ax=ax,
        stats=samples,
        group_func=lambda stat: label.format_map(stat.json_metadata),
        x_func=lambda stat: stat.json_metadata[x_axis],
        # highlight_max_likelihood_factor = 1
    )
    ax.semilogy()
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
        kwargs = dict(num_workers=4, existing_data_filepaths=[file], max_shots=nb_shots, max_errors=1000,
                      tasks=tasks, decoders=['pymatching']) | kwargs
    else:
        print('Create a new data file')
        kwargs = dict(num_workers=4, save_resume_filepath=file, max_shots=nb_shots, max_errors=1000,
                      tasks=tasks, decoders=['pymatching']) | kwargs
    # ,progress_callback=myfunction can be added
    # Collect the samples (takes a few minutes).
    samples = sinter.collect(**kwargs)
    return (samples)


def plot_analytical_error_rate(pmin, pmax, f):
    """Plot the analytical model f between pmin and pmax in a log log scale."""
    x = np.linspace(pmin, pmax, 1000)
    plt.yscale("log")
    plt.xscale("log")
    for d in [3, 5, 7, 9, 11, 13]:
        plt.plot(x, f(x, d))
    plt.legend()
    plt.title("analytical error model")
    plt.show()


def _plot_2curves(samples_1: list[sinter.TaskStats], samples_2: list[sinter.TaskStats],
                  x_axis='p', x_label="Physical error rate", label="d={d}",
                  title="Surface code", ylim=None, filename=None):
    """Fait le travail de dessiner.

    label est formaté à partir du dictionnaire stat.json_metadata
    """
    # Render a matplotlib plot of the data.
    fig, ax = plt.subplots(1, 1)

    def custom_plot_args():
        return {'linestyle': ':'}
    sinter.plot_error_rate(
        ax=ax,
        stats=samples_1,
        group_func=lambda stat: label.format_map(stat.json_metadata),
        x_func=lambda stat: stat.json_metadata[x_axis],
        # highlight_max_likelihood_factor = 1
    )
    sinter.plot_error_rate(
        ax=ax,
        stats=samples_2,
        group_func=lambda stat: label.format_map(stat.json_metadata),
        x_func=lambda stat: stat.json_metadata[x_axis],
        plot_args_func=lambda index, curve_id: {'linestyle': ':', 'label': 'without idle'}
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

# %% Threshold estimation


def generate_error_per_round_tasks(kind, p, data_type):
    """Generate surface code circuit tasks using Stim's circuit generation for different number of rounds."""
    if data_type == 'no split':
        # As a reminder, Probas=['Hadam','idle_data','idle_bell','depol', 'prep',
        # 'mes', 'bell']
        for d in [5, 7, 9]:
            for k in range(1, 2*d, 2):
                yield sinter.Task(
                    circuit=gen_memory(d, d, k, kind, None,
                                       Probas(p, p, p, p, p, p, 0)),
                    json_metadata={
                        'p': p,
                        'd': d,
                        'k': k
                    },
                )


def error_per_nb_round(kind, p, data_type):
    """Génère un plot pour un taux d'erreur physique p de P_L en fonction du nombre de round."""
    samples = _collect(generate_error_per_round_tasks(kind, p, data_type),
                       file='/home/hjacinto/multi_chip_threshold/data_simu/sampling_per_round5.txt')
    _plot_error_per_nb_round(samples, title=f"Surface code error per round, {kind}",
                             filename="surface_code_error_per_nb_round_" + kind + ".pdf")


def comparison_idle_nico(kind, idle_ratio, rep):
    """Generate surface code circuit tasks using Stim's circuit generation to compare logical error with and without idle."""
    # As a reminder, Probas=['Hadam','idle_data','idle_bell','depol', 'prep', 'mes', 'bell']

    for p in [0.0001, 0.0003, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004,
              0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]:
        for d in [3, 5, 7, 9, 11]:
            yield sinter.Task(
                circuit=gen_memory(d, d, rep(d), kind, None,
                                   Probas(p, idle_ratio*p, idle_ratio*p, p, p, p, 0)),
                json_metadata={
                    'p': p,
                    'd': d,
                    'idle ratio': idle_ratio,
                    'k': rep(d)
                },
            )


def plot_idle_nico(kind, rep=nbr_cycle):
    """Comparaison with and without idle."""
    samples_1 = _collect_and_print(comparison_idle_nico(
        kind, 1, rep), fits=[], data_type='no split')
    samples_2 = _collect_and_print(comparison_idle_nico(
        kind, 0, rep), fits=[], data_type='no split')
    _plot_2curves(samples_1, samples_2, title=f"Surface code, {kind}",
                  filename="surface_code_comparison_idle" + kind + ".pdf")


def generate_example_tasks(kind, rep, probas):
    """Generate surface code circuit without splitting tasks using Stim's circuit generation."""
    # As a reminder, Probas=['Hadam','idle_data','idle_bell','depol', 'prep', 'mes', 'bell']
    # p_fix_hadam = 1e-4
    # p_fix = 1e-3
    for p in tqdm([0.0001, 0.0003, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003,
                   0.004, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3], desc='Outer loop'):
        for d in tqdm([3, 5, 7, 9, 11], desc='Inner loop', leave=False):
            yield sinter.Task(
                circuit=gen_memory(d, d, rep(d), kind, None,
                                   multiply_namedtuple(probas, p)),
                # sum_namedtuples(multiply_namedtuple(probas, p), Probas(p_fix_hadam, 0., 0., 0., p_fix, p_fix_hadam, 0., 0.))),
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


def surface_code_threshold(kind='x', rep=nbr_cycle, probas=Probas(1, 1, 1, 1, 1, 1, 0), filtered=True):
    """Code de surface normal."""
    samples = _collect_and_print(generate_example_tasks(kind, rep, probas), fits=['general with error'
                                                                                  ], data_type='no split')
    # Plot the logical error rate per rep cycle
    _plot(samples, title=" ",
          filename="surface_code_threshold_all_cycle" + kind + ".pdf", filtered=filtered)
    # Plot the logical error rate per cycle
    _plot_per_round(samples, title=r"  ",
                    filename="surface_code_threshold_per_round" + kind + ".pdf", filtered=filtered)


def surface_code_threshold_gidney(kind='x', rep=nbr_cycle, probas=Probas(1, 1, 1, 1, 1, 1, 0),
                                  filtered=True):
    """Reproduce Gidney's figure of logical error per d rounds while simulating 3d rounds."""
    samples = _collect_and_print(generate_example_tasks(kind, rep, probas), fits=['general with error'
                                                                                  ], data_type='no split')
    # Plot the logical error rate per d cycle
    x_axis = 'p'
    x_label = " "
    label = "d={d}"
    title = " "
    ylim = None
    filename = "Gidney_figure"
    fig, ax = plt.subplots(1, 1)
    # filter the part where uncertainty is too high
    if filtered:
        sinter.plot_error_rate(
            ax=ax,
            stats=samples,
            group_func=lambda stat: label.format_map(stat.json_metadata),
            filter_func=lambda stat: (stat_error(stat.errors/stat.shots, stat.shots,
                                                 rep=stat.json_metadata['k']) < to_error_per_round(
                stat.errors/stat.shots, stat.json_metadata['k'])/4 and stat.json_metadata['p'] < 3e-2),
            x_func=lambda stat: stat.json_metadata[x_axis],
            failure_units_per_shot_func=lambda stats: stats.json_metadata['k'] /
            stats.json_metadata['d']
            # highlight_max_likelihood_factor = 1
        )
    else:
        sinter.plot_error_rate(
            ax=ax,
            stats=samples,
            group_func=lambda stat: label.format_map(stat.json_metadata),
            x_func=lambda stat: stat.json_metadata[x_axis],
            failure_units_per_shot_func=lambda stats: stats.json_metadata['k'] /
            stats.json_metadata['d']
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


def surface_code_threshold_critical(kind='x', rep=nbr_cycle, probas=Probas(1, 1, 1, 1, 1, 1, 0)):
    """Code de surface normal."""
    samples = _collect_and_print(generate_example_tasks_critical(kind, rep, probas), fits=[
        'critical exponent'], data_type='no split')
    # Plot the logical error rate per rep cycle
    _plot(samples, title="Logical error per d round",
          filename="surface_code_threshold_all_cycle" + kind + ".pdf", filtered=False)
    # Plot the logical error rate per cycle
    _plot_per_round(samples, title="Logical error per round ",
                    filename="surface_code_threshold_per_round" + kind + ".pdf", filtered=False)


def generate_telemesures_tasks(kind, p_bell, version, rep):
    """Génère les circuits pour code de surface + télémesure à p_bell fixé."""
    for p in tqdm([1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 1e-2,
                   5e-2, 7e-2, 9e-2, 1e-1], desc='Outer loop'):
        if version == 1:
            for d in tqdm([3, 5, 7, 9, 11], desc='Inner loop', leave=False):
                yield sinter.Task(
                    circuit=gen_memory(d, d, rep(d), kind, d+2,
                                       Probas(0., 0., 0., p, 0., 0., p_bell)),
                    #                   Probas(p, p, p, p, p, p, p_bell)),
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


def splitted_surface_code_pseudothreshold(kind='z', p_bell=1e-1, version=2, rep=nbr_cycle):
    """Code de surface découpé, avec erreur sur les paires de bell fixée."""
    samples = _collect_and_print(generate_telemesures_tasks(kind, p_bell, version, rep), fits=[
                                 'correlated contributions'], data_type='p_bell fixed')
    _plot(samples, x_axis='p', x_label="Bulk error rate",
          title=f"Splitted Surface Code ; kind={kind} ; p_bell={p_bell}; rep={rep}",
          filename="splitted_surface_code_p_bell_fixed_" + kind + "_version_" + str(version) + ".pdf")
    # Plot the logical error rate per cycle
    _plot_per_round(samples, title=r"$p_{\text{Bell}}$="+f"{p_bell}",
                    filename="splitted_surface_code_per_round_pb_bell_fixed" + kind + ".pdf", filtered=True)


def generate_telemesures_tasks2(kind, ratio, version, rep):
    """Génère les circuits pour code de surface + télémesure avec p_bell=ratio* p_bulk."""
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
    """Exécute le calcul de l'erreur pour un code de surface découpé.

    avec un ratio fixe entre les erreurs bulk et les erreurs
    sur les paires de bell.
    """
    samples = _collect_and_print(generate_telemesures_tasks2(
        kind, ratio, version, rep), fits=['general with error'], data_type='ratio')
    _plot(samples, x_label="Physical error p ; p_bell=ratio*p",
          title=f"Splitted surface code ; kind={kind} ; ratio={ratio}",
          filename=f"splitted_surface_code_{kind}_ratio_{ratio}.pdf")
    # Plot the logical error rate per cycle
    _plot_per_round(samples, title=f"Surface code error per round, {kind}",
                    filename=f"splitted_surface_code_per_round_ratio_{ratio}.pdf")


def generate_telemesures_tasks3(kind, p, version, rep):
    """Génère les circuits pour code de surface + télémesure avec p_bulk fixé."""
    # As a reminder, Probas=['Hadam','idle_data','idle_bell','depol', 'prep', 'mes', 'bell']

    for p_bell in tqdm([5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 9e-3, 1e-2,
                       2e-2, 5e-2, 0.1, 0.2, 0.5, 0.6], desc='Outer loop'):
        if p_bell >= 1:
            continue
        if version == 1:
            for d in tqdm([3, 5, 7, 9, 11], desc='Inner loop', leave=False):
                yield sinter.Task(
                    circuit=gen_memory(d, d, rep(d), kind, d+2,
                                       Probas(p, p, p, p, p, p, p_bell)),
                    json_metadata={
                        'd': d,
                        'p_bell': p_bell,
                        'p': p,
                        'k': rep(d)
                    },
                )
        if version == 2:
            for d in tqdm([3, 5, 7, 9, 11], desc='Inner loop', leave=False):
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


def splitted_surface_code_pseudothreshold3(kind='z', p=1e-3, version=2, rep=nbr_cycle):
    """Exécute le calcul de l'erreur pour un code de surface découpé avec erreur dans le bulk fixée."""
    samples = _collect_and_print(generate_telemesures_tasks3(kind, p, version, rep), fits=[
                                 'slopes'], data_type='p fixed')
    _plot(samples, x_axis='p_bell', x_label="Bell pairs preparation error rate",
          # title=f"Splitted surface code ; kind={kind} ; p={p}",
          title=" ",
          filename=f"splitted_surface_code_p_fixed_all_cycle_{kind}_{p}.pdf", filtered=True)
    _plot_per_round(samples, x_axis='p_bell', x_label="Bell pairs preparation error rate",
                    # title=f" Splitted surface code logical error per round"+r" p_{\text{bulk}}"+f"={p}",
                    title=" ",
                    filename=f"splitted_surface_code_p_fixed_{kind}_{p}_per_round.pdf", filtered=True)


def splitted_surface_code_pseudothreshold4(kind='z', ratio=43, version=2):
    """Genere un plot avec la courbe ratio, la courbe bulk seule et la frontiere seule.

    Reproduit la figure du papier de vuletic.
    """
    samples1 = _collect(generate_telemesures_tasks2(kind, ratio, version),
                        file='/home/hjacinto/multi_chip_threshold/data_simu/sampling_1.txt')
    samples2 = _collect(generate_telemesures_tasks2(kind, 0, version),
                        file='/home/hjacinto/multi_chip_threshold/data_simu/sampling_2.txt')
    samples3 = _collect(generate_telemesures_tasks3(kind, 0, version, ratio),
                        file='/home/hjacinto/multi_chip_threshold/data_simu/sampling_3.txt')

    x_axis = 'p'
    x_label = "Physical error rate"
    label = "d={d}"
    title = f"Surface code superposition of curve, full line is p_bell={ratio}*p,\n dotted lines are"
    +" the same with bulk noise turned off,\n dashed line are the same with bell pair noise turned off"
    ylim = None

    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=samples1,
        group_func=lambda stat: label.format_map(stat.json_metadata),
        x_func=lambda stat: stat.json_metadata[x_axis]

    )
    sinter.plot_error_rate(
        ax=ax,
        stats=samples2,
        group_func=lambda stat: label.format_map(stat.json_metadata),
        x_func=lambda stat: stat.json_metadata[x_axis],
        plot_args_func=lambda index, curve_id: {'linestyle': ':'
                                                },
        highlight_max_likelihood_factor=1
    )
    sinter.plot_error_rate(
        ax=ax,
        stats=samples3,
        group_func=lambda stat: label.format_map(stat.json_metadata),
        x_func=lambda stat: stat.json_metadata['p_bell'],
        plot_args_func=lambda index, curve_id: {'linestyle': '--'
                                                },
        highlight_max_likelihood_factor=1
    )
    ax.loglog()
    ax.set_ylim(ylim)
    ax.grid()
    ax.set_title(title)
    ax.set_ylabel("Logical error probability superpositon")
    ax.set_xlabel(x_label)
    # ax.legend()
    # Save to file and also open in a window.

    plt.show()


def splitted_surface_code_pseudothreshold_w_wo_bpnoise(kind='z', p_bell0=0.0, p_bell1=0.05,
                                                       p_bell2=0.01, version=2, rep=nbr_cycle):
    """Genere un plot avec la courbe sans bruit sur les pairs de bell et avec bruit.

    Simule 3 cycle pour éviter les effets de bord puis se ramène à l'erreur par cycle.'
    Permet de quantifier l'effet du bruit sur la paire de bell et de voir l'effet de seuil.
    """
    samples1 = _collect(generate_example_tasks(kind, rep, probas=Probas(1, 1, 1, 1, 1, 1, 0)),
                        file='/home/hjacinto/multi_chip_threshold/data_simu/sampling_16.txt')
    # samples2 = _collect(generate_telemesures_tasks(kind, p_bell1, version, rep),
    #                     file='/home/hjacinto/multi_chip_threshold/data_simu/sampling_14.txt')
    samples3 = _collect(generate_telemesures_tasks(kind, p_bell2, version, rep),
                        file='/home/hjacinto/multi_chip_threshold/data_simu/sampling_15.txt')

    x_axis = 'p'
    x_label = "Physical error rate"
    label = "d={d}"
    title = "full line are withour split " r"$p_{\text{bell}}=$" f"{p_bell2}"
    ylim = None
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=samples1,
        group_func=lambda stat: label.format_map(stat.json_metadata),
        x_func=lambda stat: stat.json_metadata[x_axis],
        plot_args_func=lambda index, curve_id: {'label': r"$p_{\text{bell}}=$"f"{p_bell0}"}
    )
    sinter.plot_error_rate(
        ax=ax,
        stats=samples3,
        group_func=lambda stat: label.format_map(stat.json_metadata),
        x_func=lambda stat: stat.json_metadata[x_axis],
        plot_args_func=lambda index, curve_id: {'linestyle': ':'
                                                },
        highlight_max_likelihood_factor=1
    )
    # sinter.plot_error_rate(
    #     ax=ax,
    #     stats=samples3,
    #     group_func=lambda stat: label.format_map(stat.json_metadata),
    #     x_func=lambda stat: stat.json_metadata[x_axis],
    #     plot_args_func=lambda index, curve_id: {'linestyle': '--'
    #                                             },
    #     highlight_max_likelihood_factor=1
    # )
    ax.loglog()
    ax.set_ylim(ylim)
    ax.grid()
    ax.set_title(title)
    ax.set_ylabel("Logical error probability")
    ax.set_xlabel(x_label)
    # ax.legend()
    # Save to file and also open in a window.
    plt.show()


def generate_telemesures_tasks_3d(P_bulk, P_seam, kind, rep):
    """Genere les circuits pour faire le fit 3d."""
    for p in tqdm(P_bulk, desc='Loop on p_bulk'):
        for p_bell in tqdm(P_seam, desc='Inner loop on p_bell', leave=False):
            for d in [3, 5, 7, 9, 11]:
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


# %%Function of threshold estimation with several seams

def generate_telemesures_tasks_multiple_seam(kind, p_bell, rep, seam_cols):
    """Genere les circuits pour plusieurs seam à p_bell fixé."""
    for p in tqdm([1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 1e-2,
                   5e-2, 7e-2, 9e-2, 1e-1], desc='Outer loop'):
        for d in tqdm([3, 5, 7, 9], desc='Inner loop', leave=False):
            yield sinter.Task(
                circuit=gen_memory(d, 22, rep(d), kind, seam_cols,
                                   Probas(p, p, p, p, p, p, p_bell)),
                json_metadata={
                    'p': p,
                    'd': d,
                    'p_bell': p_bell,
                    'k': rep(d)
                },
            )


def mulitple_seam_surface_code_pseudothreshold(kind='z', p_bell=1e-2, rep=nbr_cycle, seam_cols=[6, 18]):
    """Exécute le calcul de l'erreur pour un code de surface découpé avec erreur dans le bulk fixée."""
    samples = _collect_and_print(generate_telemesures_tasks_multiple_seam(kind, p_bell, rep, seam_cols), fits=[
    ], data_type='p_bell fixed')
    _plot(samples, x_axis='p', x_label="Physical bulk error rate",
          # title=f"Splitted surface code ; kind={kind} ; p={p}",
          title=" ",
          filename=f"Multiple_split_surface_code_per_d_rounds_p_bell_{p_bell}.pdf", filtered=True)
    _plot_per_round(samples, x_axis='p', x_label="Physical bulk error rate",
                    # title=f" Splitted surface code logical error per round"+r" p_{\text{bulk}}"+f"={p}",
                    title=" ",
                    filename=f"Multiple_split_surface_code_per_round_p_bell_{p_bell}.pdf", filtered=True)

# %% Fonction to call


def surface_code_pseudotreshold(kind, data_type, version=2,
                                ratio=10, p_fixe=1e-3, p_bell_fixe=1e-1, number_points=10,
                                probas=Probas(1, 1, 1, 1, 1, 1, 0)):
    """Fonction à appeler pour faire une simu."""
    if data_type == 'no split':
        surface_code_threshold(kind=kind, probas=probas)
    if data_type == 'ratio':
        splitted_surface_code_pseudothreshold2(kind=kind, ratio=ratio, version=version)
    if data_type == 'p fixed':
        splitted_surface_code_pseudothreshold3(kind=kind, p=p_fixe, version=version)
    if data_type == 'p_bell fixed':
        splitted_surface_code_pseudothreshold(kind=kind, p_bell=p_bell_fixe, version=version)
    if data_type == 'superposition ratio':
        splitted_surface_code_pseudothreshold4(kind=kind, ratio=ratio, version=version)


def _3dfit(p_bulk_min, p_bulk_max, p_seam_min, p_seam_max, kind='z', lattice_size=10, rep=nbr_cycle):
    """Perform a 3d fit of the vuletic formula over the 3d space.

    With logical error, bulk error and seam errors as axis.
    """
    # Creates the space of points to be covered
    P_bulk = np.linspace(p_bulk_min, p_bulk_max, lattice_size)
    P_seam = np.linspace(p_seam_min, p_seam_max, lattice_size)
    # P_bulk = np.logspace(np.log(p_bulk_min), np.log(p_bulk_max), lattice_size)
    # P_seam = np.logspace(np.log(p_seam_min), np.log(p_seam_max), lattice_size)
    # collect the result from circuit sampling with sinter
    samples = _collect(generate_telemesures_tasks_3d(P_bulk, P_seam, kind, rep),
                       file='/home/hjacinto/multi_chip_threshold/data_simu/sampling_3d_v9.csv')
    # Prepare the data to be fit
    data_tofit, data = prepare_data(samples, data_type='3d', fits=[])

    # Declare the parameters of the fit
    fit_params3 = Parameters()
    fit_params3.add('alphac', value=1, min=0.0, max=1000.0)
    fit_params3.add('alpha3', value=1, min=0.0, max=100.0)
    # Fix parameters estimated on the slice p_seam=0 and p_bulk=0
    alpha1 = 0.24
    alpha2 = 0.067
    pbth = 7.3e-3
    psth = 0.296
    # test value corresponding to the fit on slice p_bulk=1e-3
    # alphactest=301
    # alpha3test=0.046

    def objective_werror_coupled_3d(params):
        resid = []
        for d in list(data.keys()):
            # Objective function is of the form (y-f(x))/std_y
            resid = np.concatenate((resid, list((data_tofit[d][2][:]-coupled_error_model_v2(
                data_tofit[d][1][:], data_tofit[d][0][:], alpha1, alpha2, params['alpha3'].value,
                psth, pbth, params['alphac'].value, d))/(data_tofit[d][3][:]))))
        return (resid)
    out3 = minimize(objective_werror_coupled_3d, fit_params3)
    report_fit(out3.params)
    # out3 = minimize(objective_werror_coupled_3d, fit_params3,
    #                method="emcee", is_weighted=True)
    report_fit(out3.params)
    return (data_tofit, out3.params['alpha3'].value, out3.params['alphac'].value,
            out3.params['alpha3'].stderr, out3.params['alphac'].stderr, P_bulk, P_seam)


def _3dplot(data_tofit, alpha3, alphac, std_alpha3, std_alphac, P_bulk, P_seam):
    """Do the plot of the 3d fit where the fitted model display as surfaces and data as points."""
    alpha1 = 0.24
    alpha2 = 0.067
    pbth = 7.3e-3
    psth = 0.3
    # alpha3test = 0.026
    # alphactest = 372
    # alpha3test = 0.011
    # alphactest = 0.47
    # pbmin, pbmax, psmin, psmax, n = P_bulk[0], P_bulk[-1], P_seam[0], P_seam[-1], len(P_bulk)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for d in list(data_tofit.keys())[:]:
        points = []
        for i in range(len(list(data_tofit[d][0][:]))):
            points.append(
                (data_tofit[d][0][i], data_tofit[d][1][i], data_tofit[d][2][i]))
        # Listes contenant des triplets de la forme p_bulk,p_bell,p_logic
        x, y, z = zip(*points)
        grid_x, grid_y = np.mgrid[min(x):max(x):10j, min(y):max(y):10j]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
        # Liste contenant error_model(p_bulk,p_bell)
        fitted_z = coupled_error_model_v2(grid_y, grid_x, alpha1, alpha2,
                                          alpha3, psth, pbth, alphac, d)
        ax.scatter(np.log10(grid_x), np.log10(grid_y), np.log10(grid_z))
        ax.plot_surface(np.log10(grid_x), np.log10(grid_y), np.log10(
            fitted_z), color=f'C{d}', alpha=0.5, label=f'distance d={d}')

    # Do the log scale by hand
    def log_tick_formatter(val, pos=None):
        # remove int() if you don't use MaxNLocator
        return f"$10^{{{int(val)}}}$"
        # return f"{10**val:.2e}"      # e-Notation

    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_zlabel(r"$p_L$", fontsize=20)
    ax.set_xlabel(r'$p_{\text{bulk}}$', fontsize=20)
    ax.set_ylabel(r'$p_{\text{bell}}$', fontsize=20)
    # plt.title(
    #     f"Two contributions fit in 3d. \n fitted paramters are alpha_c={round(alphac,3)} and alpha3={round(alpha3,3)} ")
    plt.title(
        r"$\alpha_c =$"+f"{round(alphac,2)}"+r"$\pm$"+f"{round(std_alphac,2)} and "+r"$\alpha_3 =$" +
        f"{round(alpha3,3)}"+r"$\pm$"+f"{round(std_alpha3,3)}", fontsize=20)
    fig.savefig('3d_fit.pdf')
    # # Test to register as html
    # mpld3.save_html(fig, "plot.html")
    # # Convert the Matplotlib figure to a Plotly figure
    # plotly_fig = ptools.mpl_to_plotly(fig)
    # # Save as an HTML file
    # pio.write_html(plotly_fig, '3d_fit_new_ansatz.html')
    plt.show()


def evaluate_influence(kind, rep=lambda x: 3*x):
    """Fit threshold with different noise model."""
    # Base case
    surface_code_threshold(kind, rep, probas=Probas(1, 1, 1, 1, 1, 1, 0))
    # Without idle noise
    surface_code_threshold(kind, rep, probas=Probas(1, 0, 0, 1, 1, 1, 0))
    # without prep noise
    surface_code_threshold(kind, rep, probas=Probas(1, 1, 1, 1, 0, 1, 0))
    # without mes noise
    surface_code_threshold(kind, rep, probas=Probas(1, 1, 1, 1, 1, 0, 0))
    # Without noise on hadamard
    surface_code_threshold(kind, rep, probas=Probas(0, 1, 1, 1, 1, 1, 0))
    # Without noise on CNOT
    surface_code_threshold(kind, rep, probas=Probas(1, 1, 1, 0, 1, 1, 0))


def plot_for_thesis(kind, rep=lambda x: 3*x):
    """Simulate for different noise models."""
    # Noise model A
    surface_code_threshold(kind, rep, probas=Probas(0, 0, 0, 1, 0, 0, 0))
    # Noise model B
    surface_code_threshold(kind, rep, probas=Probas(0, 0, 0, 1, 1, 1, 0))
    # Noise model C
    surface_code_threshold(kind, rep, probas=Probas(1, 1, 0, 1, 1, 1, 0))


def plot_error_model(kind, d):
    """Plot the matching graph of the code."""
    circuit = gen_memory(d, d, d, kind, d+1)
    dem = circuit.detector_error_model()
    with open('matching_graph/matching_graph.svg', 'w') as f:
        print(circuit.diagram("matchgraph-svg"), file=f)
        # print(circuit.diagram("matchgraph-svg"), file=f)
        circuit.diagram("matchgraph-3d-html")
    with open('matching_graph/matching_graph_dem_with_bell.gltf', 'w') as f:
        print(dem.diagram("matchgraph-3d"), file=f)
    with open('matching_graph/matching_graph_circ_with_bell.gltf', 'w') as f:
        print(circuit.diagram("matchgraph-3d"), file=f)

# %% Storing of the last simulation


def del_last_simu():
    """Delete the data of last simu."""
    file_path = '/home/hjacinto/multi_chip_threshold/data_simu/last_simu.txt'
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
    data_qubits, x_stabs, z_stabs = surf_qubits_stabs(5, 10)
    # plot_stabs(data_qubits, x_stabs, z_stabs)
    data_qubits2, x_stabs2, z_stabs2, x_tel2, z_tel2 = split_surf_code(
        data_qubits, x_stabs, z_stabs, [5, 11])
    # plot_stabs(data_qubits2, x_stabs2, z_stabs2, x_tel2, z_tel2, convention='ij')
    data_qubits2, x_stabs2, z_stabs2, x_tel2, z_tel2 = split_surf_code(
        data_qubits, x_stabs, z_stabs, [6, 12])
    # plot_stabs(data_qubits2, x_stabs2, z_stabs2, x_tel2, z_tel2, convention='ij')
    dist_i, dist_j = 3, 3
    cycle = _surface_code_cycle(dist_i, dist_j, data_qubits2, x_stabs2, z_stabs2,
                                x_tel2, z_tel2, 4)
    circuit = gen_memory(dist_i, dist_j, 3, 'x', 4, Probas(0., 0., 0., 0., 0., 0.))
    with open("circuit1.html", 'w') as file:
        print(circuit.diagram("interactive"), file=file)

    # del_last_simu()
    # The type of figure you want is in data_type

    surface_code_pseudotreshold(kind='z', data_type='p fixed', version=1, p_fixe=0e-3)


# %% Tests


def test_split_without_errors():
    """Test that splited code with perfect Bell ~= normal surface code."""
    d, p = 3, 1e-3
    std = sinter.Task(circuit=gen_memory(d, d, d, 'x', None,
                                         Probas(p, 0, 0, 0)),
                      json_metadata={'name': "standard"})
    splitted = sinter.Task(circuit=gen_memory(d, d, d, 'x', d+2,
                                              Probas(p, 0, 0, 0)),
                           json_metadata={'name': "splited"})
    samples = sinter.collect(num_workers=4,
                             max_shots=10_000_000,
                             max_errors=1000,
                             tasks=(std, splitted),
                             decoders=['pymatching'],
                             )
    # Follow https://fr.wikipedia.org/wiki/Loi_binomiale#Tests
    p1 = samples[0].errors / samples[0].shots
    p2 = samples[1].errors / samples[1].shots
    p = (p1 + p2)/2
    z = abs(p1 - p2) / np.sqrt(p*(1-p)*(1/samples[0].shots+1/samples[1].shots))
    assert z <= 1.96, f"Statistical test failed (might be normal)!\n{samples}"

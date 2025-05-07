from magic_state_factory import MagicStateFactory
from functools import lru_cache
import mpmath
from mpmath import mp
import numpy as np
from scipy import optimize
from definitions import (
    z,
    one,
    projx,
    kron,
    trace,
    apply_rot,
    plog,
    storage_x_5,
    storage_z_5,
    init5qubit,
    ideal15to1,
)
import re
import matplotlib.pyplot as plt
from onelevel15to1 import one_level_15to1_state
from twolevel15to1 import cost_of_two_level_15to1
from twolevel8toCCZ import cost_of_two_level_8toccz
import pickle
from tqdm import tqdm

# To reproduce our results, make sure in definitions.py the plog functions use a threshold of 7.6e-3 instead of the initial 1e-2.

def merge_text_file(file1_path, file2_path, output_file_path):
    """Create the concatenation of both file's text."""
    # Open and read the contents of the first file
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2, open(output_file_path, 'w') as output_file:
        # Write content of the first file to the output file
        output_file.write(file1.read())
        # Optionally, add a newline between the contents
        output_file.write('\n')
        # Write content of the second file to the output file
        output_file.write(file2.read())
    print(f'Files {file1_path} and {file2_path} have been merged into {output_file_path}.')


def creation_tableau_15to1():
    """Fonction qui créer un fichier avec un tableau d'usines possibles."""
    # Appeler la fonction pour obtenir le résultat
    pphys = 1e-3
    for dx in range(11, 16, 2):
        for dz in range(5, 10, 2):
            for dx2 in range(25, 34, 2):
                for dz2 in range(11, 18, 2):
                    for nl1 in [4, 6]:
                        dm = dz
                        dm2 = dz2
                        resultat = cost_of_two_level_15to1(pphys, dx, dz, dm, dx2, dz2, dm2, nl1)
                        with open("usines_litinski_15to1_15to1_test_new_th.txt", "a") as fichier:
                            fichier.write("/n"+repr(resultat))


def creation_tableau_8toCCZ():
    """Fonction qui créer un fichier avec un tableau d'usines possibles."""
    # Appeler la fonction pour obtenir le résultat
    pphys = 1e-3
    # list_dx = range(15, 20, 2)
    # list_dz = range(7, 12, 2)
    # list_dx2 = range(27, 34, 2)
    # list_dz2 = range(15, 22, 2)
    # list_nl1 = [4, 6]
    list_dx = range(17, 24, 2)
    list_dz = range(9, 14, 2)
    list_dx2 = range(29, 36, 2)
    list_dz2 = range(17, 22, 2)
    list_nl1 = [4, 6]
    total_iterations = len(list_dx) * len(list_dz) * len(list_dx2) * len(list_dz2) * len(list_nl1)
    # Barre de progression pour suivre les itérations
    with tqdm(total=total_iterations, desc="Calculs en cours", unit="itérations") as pbar:
        for dx in list_dx:
            for dz in list_dz:
                for dx2 in list_dx2:
                    for dz2 in list_dz2:
                        for nl1 in list_nl1:
                            dm = dz
                            dm2 = dz2
                            resultat = cost_of_two_level_8toccz(
                                pphys, dx, dz, dm, dx2, dz2, dm2, nl1)
                            with open("usines_litinski_15to1_8toCCZ_test_new_th_v2.txt", "a") as fichier:
                                fichier.write("\n" + repr(resultat))
                            print(repr(resultat))
                            pbar.update(1)  # Incrémente la barre de progression


def remove_duplicates(input_list):
    """Remove duplicates."""
    unique_elements = []
    for item in input_list:
        if item not in unique_elements and item is not None:
            unique_elements.append(item)
    return unique_elements


def plot_usines(file_path):
    """
    Genere un plot avec toutes les usines contenues dans le fichier donné.

    Args:
    - fichier txt avec les données d'usine séparés par des \n
    """
    # Expressions régulières pour extraire les valeurs
    output_error_pattern = re.compile(r'Output error: ([\d\.e-]+)')
    qubitcycles_pattern = re.compile(r'Qubitcycles: (\d+)')
    qubits_pattern = re.compile(r'Qubits: (\d+)')
    codecycles_pattern = re.compile(r'Code cycles: (\d+\.\d)')
    dx_pattern = re.compile(r'dx=(\d+)')
    dz_pattern = re.compile(r'dz=(\d+)')
    dm_pattern = re.compile(r'dm=(\d+)')
    dx2_pattern = re.compile(r'dx2=(\d+)')
    dz2_pattern = re.compile(r'dz2=(\d+)')
    dm2_pattern = re.compile(r'dm2=(\d+)')
    nl1_pattern = re.compile(r'nl1=(\d+)')
    # Listes pour stocker les valeurs
    output_errors = []
    qubitcycles = []
    factories = []
    # Lecture et extraction des données
    with open(file_path, 'r') as file:
        content = file.read().split('\n\n')  # Séparer par les marqueurs "/n"
        for entry in content:
            output_error_match = output_error_pattern.search(entry)
            qubitcycles_match = qubitcycles_pattern.search(entry)
            qubits_match = qubits_pattern.search(entry)
            codecycles_match = codecycles_pattern.search(entry)
            dx_match = dx_pattern.search(entry)
            dz_match = dz_pattern.search(entry)
            dm_match = dm_pattern.search(entry)
            dx2_match = dx2_pattern.search(entry)
            dz2_match = dz2_pattern.search(entry)
            dm2_match = dm2_pattern.search(entry)
            nl1_match = nl1_pattern.search(entry)
            # Ajouter les valeurs si elles sont trouvées
            if output_error_match and qubitcycles_match:
                factory = {}
                factory['Output error'] = float(output_error_match.group(1))
                factory['Qubitcycles'] = float(qubitcycles_match.group(1))
                factory['Qubits'] = float(qubits_match.group(1))
                factory['Codecycles'] = float(codecycles_match.group(1))
                factory['dx'] = float(dx_match.group(1))
                factory['dz'] = float(dz_match.group(1))
                factory['dm'] = float(dm_match.group(1))
                factory['dx2'] = float(dx2_match.group(1))
                factory['dz2'] = float(dz2_match.group(1))
                factory['dm2'] = float(dm2_match.group(1))
                factory['nl1'] = float(nl1_match.group(1))
                factories.append(factory)
    # Création du graphique
    factories = remove_duplicates(factories)
    output_errors = [factory['Output error'] for factory in factories]
    qubitcycles = [factory['Qubitcycles'] for factory in factories]
    plt.figure(figsize=(10, 6))
    plt.scatter(output_errors, qubitcycles, alpha=0.7)
    plt.xlabel('Output Error')
    plt.ylabel('Qubitcycles')
    plt.title('Qubitcycles vs. Output Error')
    plt.xscale('log')  # Échelle logarithmique pour mieux voir les variations
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.show()
    return output_errors, qubitcycles, factories


def best_factory(target_output_error, output_errors, qubitcycles, factories, criteria='Qubitcycles'):
    """
    Finds the factory with the lowest qubitcycles value among those with output_error
    values within a factor of 2 of the target_output_error.

    Args:
    - target_output_error (float): The target output error to compare against.
    - output_errors (list of float): List of output error values for each factory.
    - qubitcycles (list of int): List of qubitcycles values for each factory.
    - factories :list containing dict with attributes of each factories
    Returns:
    - (float, int): A tuple containing the output error and qubitcycles of the best factory.
    - None: If no factories meet the criteria.
    """
    # Define acceptable range for output error (factor of 2 from target)
    min_error = min(output_errors)
    max_error = target_output_error

    # Filter factories within the error range
    candidates = [
        factory
        for factory in factories
        if min_error <= factory['Output error'] <= max_error
    ]
    # If no factories match, return None
    if not candidates:
        return None

    # Find the candidate with the minimum qubitcycles
    best_factory = min(candidates, key=lambda x: x[criteria])

    return best_factory


def best_factories_for_targets(target_output_errors, output_errors, qubitcycles, factories, criteria='Qubitcycles'):
    """
    Finds the best factory (with minimum qubitcycles) for each target_output_error
    in the list target_output_errors. Each factory's output_error should be within
    a factor of 2 from the target_output_error.

    Args:
    - target_output_errors (list of float): List of target output errors to compare against.
    - output_errors (list of float): List of output error values for each factory.
    - qubitcycles (list of int): List of qubitcycles values for each factory.

    Returns:
    - list of tuples: Each tuple contains (output_error, qubitcycles) for the best factory
                      for each target_output_error, or None if no factory matches.
    """
    # Find the best factory for each target output error using the best_factory function
    best_factories = [best_factory(target, output_errors, qubitcycles, factories, criteria=criteria)
                      for target in target_output_errors]
    # Remove duplicates and prepare list to plot
    best_factories_no_duplicate = remove_duplicates(best_factories)
    output_errors_best = [factory['Output error'] for factory in best_factories_no_duplicate]
    qubitcycles_best = [factory['Qubitcycles'] for factory in best_factories_no_duplicate]
    qubits_best = [factory['Qubits'] for factory in best_factories_no_duplicate]
    qubits = [factory['Qubits'] for factory in factories]
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(output_errors, qubitcycles, alpha=0.7, label='All Factories')
    # Highlight the chosen best factories in red
    plt.scatter(output_errors_best, qubitcycles_best, color='red', s=50, label='Selected Factories')
    plt.xlabel('Output Error')
    plt.ylabel('Qubitcycles')
    plt.title('Qubitcycles vs. Output Error with Chosen Best 15-to-1 => 8-to-CCZ Factories')
    # plt.title('Qubitcycles vs. Output Error with Chosen Best 15-to-1 => 15-to-1 Factories')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()
    # Plot with Qubits in y axis
    plt.figure(figsize=(10, 6))
    plt.scatter(output_errors, qubits, alpha=0.7)
    # Highlight the chosen best factories in red
    plt.scatter(output_errors_best, qubits_best, color='purple', s=50, label='Selected Factories')
    plt.xlabel('Output Error', fontsize=17)
    plt.ylabel('Qubits', fontsize=17)
    # Modification de la taille de la police des ticks
    plt.tick_params(axis='both', which='both', labelsize=14)
    # plt.title('Qubits vs. Output Error with Chosen Best 15-to-1 => 8-to-CCZ Factories')
    # plt.title('Qubits vs. Output Error with Chosen Best 15-to-1 => 15-to-1 Factories')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=17)
    plt.show()
    return best_factories_no_duplicate


# Créer les usines
# creation_tableau_15to1()
# creation_tableau_8toCCZ()


# Chemin vers le fichier
file_path = "usines_litinski_15to1_8toCCZ_test_new_th_v2.txt"

output_errors, qubit_cycles, factories = plot_usines(file_path)
# for the 8 to CCZ factories :
target_output_errors = np.logspace(-14, -11, num=10)
# for the 15 to 1 :
# target_output_errors = np.logspace(-16, -11, num=50)
best_factories = best_factories_for_targets(
    target_output_errors, output_errors, qubit_cycles, factories, criteria='Qubits')
# Save the dictionary to a file
with open('/home/hjacinto/biblio/code_litinski/magicstates-master/Python/factories_8toCCZ_qubits_10_fact_new_th.pkl', 'wb') as f:
    pickle.dump(best_factories, f)

print(best_factories)
# Open the pickle file in read-binary mode and load the data
# with open('/home/hjacinto/biblio/code_litinski/magicstates-master/Python/factories_8toCCZ_qubits_10_fact.pkl', 'rb') as pkl_file:
#     my_list = pickle.load(pkl_file)

# # Print the loaded list
# print("Loaded list from pickle file:", my_list)

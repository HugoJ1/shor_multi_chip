#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the resources for factoring RSA integer.

For surface code 1 qubits gate are free in our Shor implementation as they always happen to be 
Paulis or Hadamard right after initialization or before measurement.

Created on May 19, 2020 10:28:20
@author: Élie Gouzien
"""
from math import isnan, isinf
from itertools import product
from functools import reduce
import warnings
from warnings import warn
import os
import matplotlib as mpl
import matplotlib.ticker as ticker
from tqdm import tqdm
import numpy as np
from tools import AlgoOpts, LowLevelOpts, Params, PhysicalCost
from error_correction import ErrCorrCode, logical_qubits
import error_correction
import re
try:
    if not __IPYTHON__:
        raise NameError("Artifice")
    IPYTHON = True
except NameError:
    IPYTHON = False
    mpl.use('pdf')
import matplotlib.pyplot as plt

warnings.simplefilter("always", UserWarning)

PB_DEF = True  # Default for progress bars.


# %% Optimization
# Default parameters (adapted to each architecture)
DEF_RANGES = {'surface': dict(d1s=range(3, 30, 2),
                              d2s=(None,),
                              ds=range(3, 51, 2),
                              ns=(None,),
                              wes=range(2, 20),
                              wms=range(2, 10),
                              cs=range(1, 40)),
              'surface_free_CCZ': dict(d1s=(None,),
                                       d2s=(None,),
                                       ds=range(3, 51, 2),
                                       ns=(None,),
                                       wes=range(2, 20),
                                       wms=range(2, 10),
                                       cs=range(1, 40)),
              'surface_small_procs': dict(d1s=range(3, 30, 2),
                                          d2s=(None,),
                                          ds=range(3, 51, 2),
                                          ns=(None,),
                                          wes=range(2, 30),
                                          wms=range(2, 10),
                                          cs=range(1, 40)),
              'surface_small_procs_compact': dict(d1s=range(0, 8),
                                                  d2s=(None,),
                                                  ds=range(21, 51, 2),
                                                  ns=(None,),
                                                  wes=range(2, 30),
                                                  wms=range(2, 10),
                                                  cs=range(1, 40)),
              'surface_small_procs_compact_v2': dict(d1s=range(0, 8),
                                                        d2s=(None,),
                                                        ds=range(27, 55, 2),
                                                        ns=(None,),
                                                        wes=range(2, 30),
                                                        wms=range(2, 10),
                                                        cs=range(1, 40)),
              # Hardcoded version of one proc with the same parameters as optimal for small proc
              'surface_big_procs': dict(d1s=range(0, 5),
                                        d2s=(None,),
                                        ds=range(3, 51, 2),
                                        ns=(None,),
                                        wes=range(2, 30),
                                        wms=range(2, 10),
                                        cs=range(1, 40))}


def _calc_ranges(base_params: Params, **kwargs):
    """Calculate the iteration ranges for optimization.

    Possible kwargs: d1s, d2s, ds, ns, wes, wms, cs
    """
    ranges = DEF_RANGES[base_params.type].copy()
    ranges.update(kwargs)
    if not base_params.algo.windowed:
        if (ranges['wes'] not in (DEF_RANGES[base_params.type]['wes'], (None,))
                or ranges['wms'] not in (DEF_RANGES[base_params.type]['wms'],
                                         (None,))):
            warn("'wes' or 'wms' range explicitly given while non windowed "
                 "arithmetic circuit used. Removing them!")
        ranges['wes'] = (None,)
        ranges['wms'] = (None,)
    return ranges


def iterate(base_params: Params, progress=PB_DEF, **kwargs):
    """Generator for all changeable values.

    progress: show a progress bar?
    Possible kwargs: d1s, d2s, ds, ns, wes, wms, cs
    """
    # pylint: disable=C0103
    ranges = _calc_ranges(base_params, **kwargs)
    iterator = product(ranges['d1s'], ranges['d2s'], ranges['ds'],
                       ranges['ns'], ranges['wes'], ranges['wms'],
                       ranges['cs'])
    if progress:
        nb_iter = reduce(lambda x, y: x*len(y), ranges.values(), 1)
        iterator = tqdm(iterator, total=nb_iter, dynamic_ncols=True)
    for d1, d2, d, n, we, wm, c in iterator:
        # we and wm are interchangeable, limiting the search
        if (base_params.algo.prob == 'rsa'
                and wm is not None and we is not None and wm > we):
            continue
        # Elliptic curves: we must be at least 3 to use the addition trick.
        if (base_params.algo.prob == 'elliptic_log'
                and we is not None and we <= 2):
            continue
        yield base_params._replace(
            algo=base_params.algo._replace(we=we, wm=wm, c=c),
            low_level=base_params.low_level._replace(d1=d1, d2=d2, d=d, n=n))


def metrique(cost: PhysicalCost, qubits, params: Params, biais=1):
    """Quality criterion for a score."""
    n = params.low_level.n or 1  # pylint: disable=C0103
    return cost.exp_t * qubits**biais * n


def prepare_ressources(params: Params):
    """Prepare the cost for given parameters."""
    err_corr = ErrCorrCode(params)
    if params.algo.prob == 'rsa':
        cost = err_corr.factorisation()
    elif params.algo.prob == 'elliptic_log':
        cost = err_corr.elliptic_log_compute()
    else:
        raise ValueError("params.prob must be 'rsa' or 'elliptic_log'!")
    qubits = err_corr.proc_qubits
    return cost, qubits


def find_best_params(base_params: Params, biais=1, progress=PB_DEF, **kwargs):
    """Find the best set of parameters."""
    best = float('inf')
    best_params = None
    for params in iterate(base_params, progress=progress, **kwargs):
        try:
            cost, qubits = prepare_ressources(params)
        except RuntimeError as err:
            if isinstance(err, NotImplementedError):
                raise
            continue
        score = metrique(cost, qubits, params, biais)
        if score < best:
            best = score
            best_params = params
    if best_params is None:
        raise RuntimeError("Optimization did not converge. "
                           "No set of parameters allows the computation.")
    # Detecting boundary hits.
    ranges = _calc_ranges(base_params, **kwargs)
    for var_name, var_type in [('d1s', 'low_level'),
                               ('d2s', 'low_level'),
                               ('ds', 'low_level'),
                               ('ns', 'low_level'),
                               ('wes', 'algo'),
                               ('wms', 'algo'),
                               ('cs', 'algo')]:
        var_range = ranges[var_name]
        var_val = getattr(getattr(best_params, var_type), var_name[:-1])
        if (var_range != (None,) and (var_val == min(var_range)
                                      or var_val == max(var_range))):
            warn(f"Params: {best_params} ; "
                 f"Variable '{var_name[:-1 ]}={var_val}' reached one of its "
                 "extremities!")
    return best_params


# %% Plots and tables
def unit_format(num, unit, unicode=False):
    """Write the number with its unit, either in unicode or latex."""
    space = chr(8239)
    num = str(round(num)) if not isinf(num) else "∞" if unicode else r"\infty"
    if not unicode:
        unit = {"µs": r"\micro\second",
                "ms": r"\milli\second",
                "s": r"\second",
                "min": r"\minute",
                "hours": "hours",
                "days": "days"}[unit]
    if unicode:
        return num + space + unit
    return rf"\SI{{{num}}}{{{unit}}}"


def format_time(time, unicode=False):
    """Display the time with the correct unit."""
    if time is None:
        return repr(None)
    if isnan(time):
        return "nan"
    if time < 1e-3:
        temps, unit = time*1e6, "µs"
    elif time < 1:
        temps, unit = time*1000, "ms"
    elif time < 60:
        temps, unit = time, "s"
    elif time < 3600:
        temps, unit = time/60, "min"
    elif time < 3600*24:
        temps, unit = time/3600, "hours"
    else:
        temps, unit = time/(3600*24), "days"
    return unit_format(temps, unit, unicode)


def full_data_string(best_params, best_err_corr, best_cost, best_qubits):
    """Prepare the string with all the data from a simulation."""
    string = f'\n Params={best_params}\n'
    string += f'factory_cost={ErrCorrCode(best_params)._factory}\n'
    string += f'Best case: {best_cost}, ; ,  {best_qubits}\n'
    if hasattr(best_err_corr, 'nb_procs'):
        string += f'Number of processors: {best_err_corr.nb_procs-2, + 2}\n'
        string += f'Layout parameters: nx={best_err_corr.nx}, ny={best_err_corr.ny}\n'
        string += f'Number of qubits per proc: {best_err_corr.proc_qubits_each}\n'
        string += f'Total number of physical qubits: {best_err_corr.proc_qubits}\n'
    return (string)


def from_fulltxt_to_partialtxt(input_file, output_file):
    """Convert a full .txt file to a partial one.

    For the plot_overhead_comparison function.
    """
    # Regular expressions to match the required values
    #pbell_pattern = re.compile(r"pbell=([\d.]+)")
    pbell_pattern = re.compile(r"pbell=np\.float64\(([\d\.]+)\)")
    #exp_t_pattern = re.compile(r"Meilleur cas : .*exp_t=(\d+) days, (\d+):(\d+):([\d.]+)")
    exp_t_pattern = re.compile(r"Best case: .*exp_t=(\d+) days, (\d+):(\d+):([\d.]+)")
    #qubits_pattern = re.compile(r"Nombre de qubit physique total :,([\d]+)")
    qubits_pattern = re.compile(r"Total number of physical qubits: ([\d]+)")

    # Lists to hold the extracted data
    results = []

    with open(input_file, "r", encoding="utf-8") as file:
        block = []  # Store lines in the current block
        for line in file:
            if line.strip() == "":  # Blank line indicates block separation
                # Process the current block
                block_text = " ".join(block)
                pbell_match = pbell_pattern.search(block_text)
                exp_t_match = exp_t_pattern.search(block_text)
                qubits_match = qubits_pattern.search(block_text)
                # If all three matches are found, add them to the results
                if pbell_match and exp_t_match and qubits_match:
                    pbell = pbell_match.group(1)
                    qubits = qubits_match.group(1)
                    days = int(exp_t_match.group(1))
                    hours = int(exp_t_match.group(2))
                    minutes = int(exp_t_match.group(3))
                    seconds = float(exp_t_match.group(4))
                    # Convert exp_t into float days
                    total_days = days + (hours / 24) + (minutes / 1440) + (seconds / 86400)

                    results.append(
                        f"pbell={pbell}, space_overhead={qubits}, exp_t_cost={total_days}")

                # Reset the block
                block = []
            else:
                block.append(line.strip())

        # Process the last block if the file does not end with a blank line
        if block != []:
            block_text = " ".join(block)
            pbell_match = pbell_pattern.search(block_text)
            exp_t_match = exp_t_pattern.search(block_text)
            qubits_match = qubits_pattern.search(block_text)

            if pbell_match and exp_t_match and qubits_match:
                pbell = pbell_match.group(1)
                qubits = qubits_match.group(1)
                days = int(exp_t_match.group(1))
                hours = int(exp_t_match.group(2))
                minutes = int(exp_t_match.group(3))
                seconds = float(exp_t_match.group(4))
                # Convert exp_t into float days
                total_days = days + (hours / 24) + (minutes / 1440) + (seconds / 86400)
                results.append(
                    f"pbell={pbell}, space_overhead={qubits}, exp_t_cost={total_days}")

    # Write the results to the output file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(results))


def plot_resource_comparison(base_params: Params, pbell_list, t_scale='days', biais=1,
                             files="data_simu/données_complètes_layout_opt_2_h_2_new_params_19_04_sampling_v15.txt",
                             file_out = "resource_estimation.pdf",
                             pbell_max=1., type='partial'):
    """Build curve with space and time cost of running RSA w.r.t pbell.

    If type='partial' only fill the txt with the information necessary for the curve
    and plot the curve.
    if the file given is already (partially or not) filled with partial information
    it will use them and not redo the simulation.
    If type='full' it plot a curve but and fills a txt with all information about
    the resource estimations.

    if pbell_list=None and type='partial' and a file is given, it will just read and do
    the curve with the partial data present in the file.
    """
    space_overhead = []
    exp_t_cost = []
    existing_lines = []
    if os.path.exists(files):
        if type == 'full':
            output_file = "data_simu/temporary_partial_file.txt"
            from_fulltxt_to_partialtxt(files, output_file)
            files = output_file
        with open(files, "r") as fichier:
            existing_lines = fichier.readlines()
    # Extract the pbell already present in the file
    if pbell_list is not None:
        pbell_list = [pbell for pbell in pbell_list if pbell < pbell_max]
        existing_pbells = [float(line.split(',')[0].split('=')[1].strip()) for line in existing_lines
                           if float(line.split(',')[0].split('=')[1].strip()) < pbell_max]
    # Fill the lists based on the pbell
    if pbell_list is not None:
        for pbell in pbell_list:
            # If the pbell already exists in the file, load the corresponding values
            if pbell in existing_pbells:
                # Find the corresponding line
                for line in existing_lines:
                    if f"pbell={pbell}" in line:
                        # Extract the corresponding values for space_overhead and exp_t_cost
                        space_overhead_value = float(line.split(',')[1].split('=')[1].strip())
                        exp_t_cost_value = float(line.split(',')[2].split('=')[1].strip())
                        # Add these values to the lists
                        space_overhead.append(space_overhead_value)
                        exp_t_cost.append(exp_t_cost_value)
                        break
            else:
                # Otherwise, perform the simulation for this pbell
                best_params = find_best_params(base_params._replace(
                    low_level=base_params.low_level._replace(pbell=pbell)), biais=biais)
                best_err_corr = ErrCorrCode(best_params)
                best_cost, best_qubits = prepare_ressources(best_params)
                space_overhead.append(best_err_corr.proc_qubits)
                if t_scale == 'days':
                    exp_t_cost.append(best_cost.exp_t / 3600 / 24)
                # Save the data in the file
                with open(files, "a") as fichier:
                    if type == 'partial':
                        fichier.write(f"pbell={pbell}, space_overhead={space_overhead[-1]},"
                                      "exp_t_cost={exp_t_cost[-1]}\n")
                    if type == 'full':
                        fichier.write(full_data_string(best_params, best_err_corr, best_cost,
                                                       best_qubits))
    else:
        for pbell in existing_pbells:
            # Find the corresponding line
            for line in existing_lines:
                if f"pbell={pbell}" in line:
                    # Extract the corresponding values for space_overhead and exp_t_cost
                    space_overhead_value = float(line.split(',')[1].split('=')[1].strip())
                    exp_t_cost_value = float(line.split(',')[2].split('=')[1].strip())
                    # Add these values to the lists
                    space_overhead.append(space_overhead_value)
                    exp_t_cost.append(exp_t_cost_value)
                    break
        pbell_list = existing_pbells
    # Create the figure and axes
    fig, ax1 = plt.subplots()
    # Plot the first curve on the left axis
    ax1.plot(pbell_list, space_overhead, 'g-', drawstyle='steps-mid')
    ax1.set_xlabel(r'$p_{\text{Bell}}$', fontsize=21)
    ax1.set_ylabel('Qubit cost', color='g', fontsize=16)
    # Create a second vertical axis on the right
    ax2 = ax1.twinx()

    # Plot the second curve on the right axis
    ax2.set_ylabel('Time cost in days', color='b', fontsize=16)
    ax2.plot(pbell_list, exp_t_cost, 'b-', drawstyle='steps-mid')
    ax2.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(bottom=2e7)
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.grid(True, which='major', linestyle="-", linewidth=0.5, alpha=0.5)
    ax2.set_yticks(np.arange(20, 45, 3))  # Adjust step size as needed
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))  # Force integer display
    ax1.tick_params(axis='both', which='both', labelsize=14)
    ax2.tick_params(axis='both', which='both', labelsize=14)
    plt.savefig(file_out)
    plt.close()


def plot_Bell_pair_generation_rate(base_params: Params, pbell_list, t_scale='days', biais=1,
                                   files="data_simu/données_complètes_layout_opt_2_h_2_new_params_19_04_sampling_v15.txt",
                                   file_out = "generation_rate.pdf",
                                   pbell_max=1., type='partial'):
    """Build curve with generation rate of Bell pair w.r.t pbell from a simulation file."""
    file_path = files
    # Regex patterns to extract values
    #pbell_pattern = re.compile(r"pbell=([\d\.e-]+)")
    pbell_pattern = re.compile(r"pbell=np\.float64\(([\d\.]+)\)")
    d_pattern = re.compile(r"d=(\d+)")
    processors_pattern = re.compile(r"Number of processors: \((\d+),")
    # Lists to store extracted values
    pbell_values = []
    d_values = []
    nb_procs = []
    tc = 1e-6
    # Read file and extract values
    with open(file_path, "r") as file:
        content = file.read().strip()  # Read entire content and strip any leading/trailing spaces
        blocks = content.split("\n\n")  # Split by double new lines for blocks

        for block in blocks:
            # Initialize variables for this block
            pbell = None
            d = None
            nb_proc = None

            # Process each line in the block
            for line in block.splitlines():
                pbell_match = pbell_pattern.search(line)
                d_match = d_pattern.search(line)
                nb_proc_match = processors_pattern.search(line)

                if pbell_match:
                    pbell = float(pbell_match.group(1))
                if d_match:
                    d = int(d_match.group(1))
                if nb_proc_match:
                    nb_proc = int(nb_proc_match.group(1)) + 2  # Adding the number of factory

            # If all required values are found, store them
            if pbell is not None and d is not None and nb_proc is not None:
                pbell_values.append(pbell)
                d_values.append(d)
                nb_procs.append(nb_proc)

    number_of_pair_per_cycle = [2 * d_values[i] *
                                nb_procs[i] * 1 / tc for i in range(len(d_values))]
    # Check if values were found
    if not pbell_values or not d_values:
        print("No valid data found in the file.")
    else:
        # Plot the Bell pair generation rate vs pbell
        # Create the figure and axes
        fig, ax1 = plt.subplots()
        # Plot the first curve on the left axis
        ax1.plot(pbell_values, number_of_pair_per_cycle, marker='o',
                 linestyle='-', color='b', drawstyle='steps-mid')
        ax1.set_xlabel(r'$p_{\text{Bell}}$', fontsize=21)
        ax1.set_ylabel('Generation rate in Hz', color='g', fontsize=16)
        ax1.tick_params(axis='y', labelcolor='g')
        ax1.set_yscale('log')
        ax1.grid(True, which='major', linestyle="-", linewidth=0.5, alpha=0.5)
        plt.savefig(file_out)
        plt.close()


def plot_overhead_comparison(base_params: Params, pbell_list, t_scale='days', biais=1,
                             files="data_simu/données_complètes_layout_opt_2_h_2_new_params_19_04_sampling_v15.txt",
                             file_out = "overhead_comparison.pdf",
                             pbell_max=1., type='partial'):
    """Build curve with space and time % overhead compared to monolithic approach of the same parameters.

    If type='partial' only fill the txt with the information necessary for the curve
    and plot the curve.
    If the file given is already (partially or not) filled with partial information,
    it will use them and not redo the simulation.
    If type='full' it will plot a curve and fill a txt with all information about
    the resource estimations.

    If pbell_list=None and type='partial' and a file is given, it will just read and do
    the curve with the partial data present in the file.
    """
    space_overhead = []
    exp_t_cost = []
    existing_lines = []
    # Base case monolithic
    monolithic_params = find_best_params(base_params._replace(
        type='surface_big_procs'), biais=biais)
    monolithic_err_corr = ErrCorrCode(monolithic_params)
    monolithic_cost, monolithic_qubits = prepare_ressources(monolithic_params)
    print("Monolithic parameters:")
    print(monolithic_params)
    print("Monolithic case:", monolithic_cost, ";",  monolithic_qubits)
    print("Factory for monolithic case:", monolithic_err_corr._factory)
    print("Number of processors:", monolithic_err_corr.nb_procs)
    print("Number of qubits per proc:", monolithic_err_corr.proc_qubits_each)
    print("Total number of physical qubits:", monolithic_qubits)

    # Basic Arithmetic
    print("\n"*2)
    # Calculate the overhead
    space_cost_monolithic = monolithic_err_corr.proc_qubits
    if t_scale == 'days':
        exp_t_cost_monolithic = monolithic_cost.exp_t / 3600 / 24

    if os.path.exists(files):
        if type == 'full':
            output_file = "data_simu/temporary_partial_file.txt"
            from_fulltxt_to_partialtxt(files, output_file)
            files = output_file
        with open(files, "r") as fichier:
            existing_lines = fichier.readlines()
    # Extract the pbell already present in the file
    if pbell_list is not None:
        pbell_list = [pbell for pbell in pbell_list if pbell < pbell_max]
        existing_pbells = [float(line.split(',')[0].split('=')[1].strip()) for line in existing_lines
                           if float(line.split(',')[0].split('=')[1].strip()) < pbell_max]
    # Fill the lists based on the pbell
    if pbell_list is not None:
        for pbell in pbell_list:
            # If the pbell already exists in the file, load the corresponding values
            if pbell in existing_pbells:
                # Find the corresponding line
                for line in existing_lines:
                    if f"pbell={pbell}" in line:
                        # Extract the corresponding values for space_overhead and exp_t_cost
                        space_overhead_value = float(line.split(',')[1].split('=')[1].strip())
                        exp_t_cost_value = float(line.split(',')[2].split('=')[1].strip())
                        # Add these values to the lists
                        space_overhead.append(
                            (space_overhead_value / space_cost_monolithic - 1) * 100)
                        exp_t_cost.append((exp_t_cost_value / exp_t_cost_monolithic - 1) * 100)
                        break
            else:
                # Otherwise, perform the simulation for this pbell
                best_params = find_best_params(base_params._replace(
                    low_level=base_params.low_level._replace(pbell=pbell)), biais=biais)
                best_err_corr = ErrCorrCode(best_params)
                best_cost, best_qubits = prepare_ressources(best_params)
                # Calculate the overhead
                space_overhead.append((best_err_corr.proc_qubits / space_cost_monolithic - 1) * 100)
                if t_scale == 'days':
                    exp_t_cost.append((best_cost.exp_t / 3600 / 24 /
                                      exp_t_cost_monolithic - 1) * 100)
                # Save the data in the file
                with open(files, "a") as fichier:
                    if type == 'partial':
                        fichier.write(f"pbell={pbell}, space_overhead={best_err_corr.proc_qubits},"
                                      "exp_t_cost={best_cost.exp_t / 3600 / 24}\n")
                    if type == 'full':
                        fichier.write(full_data_string(best_params, best_err_corr, best_cost,
                                                       best_qubits))
    else:
        for pbell in existing_pbells:
            # Find the corresponding line
            for line in existing_lines:
                if f"pbell={pbell}" in line:
                    # Extract the corresponding values for space_overhead and exp_t_cost
                    space_overhead_value = float(line.split(',')[1].split('=')[1].strip())
                    exp_t_cost_value = float(line.split(',')[2].split('=')[1].strip())
                    # Add these values to the lists
                    space_overhead.append((space_overhead_value / space_cost_monolithic - 1) * 100)
                    exp_t_cost.append((exp_t_cost_value / exp_t_cost_monolithic - 1) * 100)
                    break
        pbell_list = existing_pbells
    # Create the figure and axes
    plt.figure()
    plt.plot(pbell_list, space_overhead, 'g-', drawstyle='steps-mid', label='Space overhead in %')
    plt.xlabel(r'$p_{\text{Bell}}$', fontsize=21)
    plt.ylabel('Overhead', fontsize=21)
    plt.plot(pbell_list, exp_t_cost, 'b-', drawstyle='steps-mid', label='Time overhead in %')
    plt.grid(True, which='major', linestyle="-", linewidth=0.5, alpha=0.5)
    # plt.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))  # Force integer display
    plt.tick_params(axis='both', which='both', labelsize=14)
    plt.legend(fontsize=15)
    plt.savefig(file_out)
    plt.close()


# %% Executable Part
if __name__ == '__main__':

    params = Params('surface_small_procs_compact_v2',
                    AlgoOpts(n=2048, windowed=True, parallel_cnots=True),
                    LowLevelOpts(tr=10e-6, pbell=0.025))
    # the list of values of p_bell to test
    pbell_list = np.linspace(0.1e-2, 6e-2, num=100)

    # # Plot the resource needed to factorize a 2048 RSA bit integer with different
    # # p_bell while p is fixed to 0.1%
    # plot_resource_comparison(params, pbell_list=pbell_list, t_scale='days', biais=1,
    #                          files="data_simu/full_data_v2.txt",
    #                          pbell_max=0.045, type='full')

    # Plot the overhead with respect to the monolithic approach to factorize a 2048 RSA bit integer with different
    # # p_bell and p while fixed to 0.1%
    # plot_overhead_comparison(params, pbell_list=pbell_list, t_scale='days', biais=1,
    #                          files="data_simu/full_data_v2.txt",
    #                         pbell_max=0.045, type='full')

    # # Plot the Bell pair generation rate needed (worst case) to factorize a 2048 RSA bit integer
    # # with different p_bell while p is fixed to 0.1%
    # plot_Bell_pair_generation_rate(params, pbell_list=pbell_list, t_scale='days', biais=1,
    #                                files="data_simu/full_data_v2.txt",
    #                                pbell_max=0.045, type='full')

    # Single simulation : 
    # The first entry of Param may be :
    #  'surface_small_procs' for a layout where cnot takes 2d time step but more overhead
    #  'surface_small_procs_compact' compact layout but injection with 3 cnots
    #  'surface_small_procs_compact_v2' compact layout but injection via lattice surgery primitives directly
    #  This is the one used in the article
    #  'surface_big_procs' compact layout on a single big chip, referenced as the monolithic case

    params = Params('surface_small_procs_compact_v2',
                    AlgoOpts(n=2048, windowed=True, parallel_cnots=True),
                    LowLevelOpts(tr=10e-6, pbell=0.0))
    # Windowed arithmetic
    print("\n"*2)
    print("Windowed Arithmetic")
    print("=====================")
    best_params = find_best_params(params, biais=1)
    best_err_corr = ErrCorrCode(best_params)
    print()
    print(ErrCorrCode(best_params)._factory)
    print()
    best_cost, best_qubits = prepare_ressources(best_params)
    print(best_params)
    print("Best case:", best_cost, ";",  best_qubits)
    if hasattr(best_err_corr, 'nb_procs'):
        print("Number of processors:", best_err_corr.nb_procs-2, "+ 2")
        print("Number of qubits per proc:", best_err_corr.proc_qubits_each)
        print("Total number of physical qubits:", best_qubits)
        print("ny:", best_err_corr.ny)
    # Basic Arithmetic
    print("\n"*2)
    print("Controlled Arithmetic")
    print("======================")
    best_params_basic = find_best_params(
        params._replace(algo=params.algo._replace(windowed=False)), biais=1)
    best_err_corr_basic = ErrCorrCode(best_params_basic)
    best_cost_basic, best_qubits_basic = prepare_ressources(best_params_basic)
    print("Best basic case:", best_cost_basic, ";", best_qubits_basic)
    if hasattr(best_err_corr_basic, 'nb_procs'):
        print("Number of processors:", best_err_corr_basic.nb_procs-2, "+ 2")
        print("Number of qubits per proc:", best_err_corr_basic.proc_qubits_each)
        print("Total number of physical qubits:", best_qubits_basic)
        print("ny:", best_err_corr_basic.ny)
    # Comparison between windowed and controlled arithmetic.
    # HINT: when n -> +inf, w_e and w_m remain constant.
    print('\n')
    print("Time ratio between basic and windowed:", best_cost_basic.t / best_cost.t)
    print("Average time ratio between basic and windowed:", best_cost_basic.exp_t / best_cost.exp_t)

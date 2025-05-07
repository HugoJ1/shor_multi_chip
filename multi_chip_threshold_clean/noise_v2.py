import dataclasses
from typing import Optional, Dict, Tuple, Union
import stim
from Coordinates import Coord
import numpy as np
import copy
from collections import namedtuple
import numbers

# TO DO : Adapt the function to optionally take a noise model that depends on which qubit
# the gate is applied.

ANY_CLIFFORD_1_OPS = {"C_XYZ", "C_ZYX", "H", "H_YZ", "I"}
ANY_CLIFFORD_2_OPS = {"CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ"}
RESET_OPS = {"R", "RX", "RY", "MR"}
MEASURE_OPS = {"M", "MX", "MY", "MR"}
ANNOTATION_OPS = {"OBSERVABLE_INCLUDE", "DETECTOR", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}
NOISE_OPS = {
    "DEPOLARIZE1",   # Single-qubit depolarizing error
    "DEPOLARIZE2",   # Two-qubit depolarizing error
    "X_ERROR",       # Single-qubit bit flip error
    "Y_ERROR",       # Single-qubit Y error
    "Z_ERROR",       # Single-qubit phase flip error
    "CORRELATED_ERROR",  # Correlated error across multiple qubits
    "E",             # Any other specific error channel
    "PAULI_CHANNEL_1",  # Single-qubit Pauli error channel
    "PAULI_CHANNEL_2",  # Two-qubit Pauli error channel
    "AMP_DAMP_ERROR",   # Amplitude damping error
    "PHASE_DAMP_ERROR"  # Phase damping error
}
Probas = namedtuple("Probas", ['onequbitgate', 'idle_data', 'idle_bell', 'twoqubitgate', 'prep', 'mes', 'bell'],
                    defaults=[0, 0, 0, 1e-3, 0, 0, 0])

# Possible argument of Probas if we want Pauli channel noise
p_2noise = namedtuple("p_2noise", ['p_ix', 'p_iy', 'p_iz', 'p_xi', 'p_xx', 'p_xy', 'p_xz', 'p_yi',
                                   'p_yx', 'p_yy', 'p_yz', 'p_zi', 'p_zx', 'p_zy', 'p_zz'],
                      defaults=[0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
p_1noise = namedtuple("p_1noise", ['p_x', 'p_y', 'p_z'], defaults=[0., 0., 0.])


# %% Tools functions
def contains_target_rec(lst):
    """
    Return True if an argument of the cnot is a measurement value.

    Used for teleported CNOT where we have classically controlled Pauli.
    """
    for item in lst:
        # Check if the item is a GateTarget and contains a target_rec
        if isinstance(item, stim.GateTarget) and hasattr(item, "value"):
            # item.value returns the index of the measurement concerned or the index of the qubit
            # Only measurement have negative index
            if item.value < 0:
                return True
    return False


def build_idle_dict(p_idle_reset, p_idle_mes, p_idle1gate, p_idle2gate, p_idle_idle=None):
    """Build the dictionary that for each stim gate associate an idle noise.

    This idle noise will be the one applied on idle qubit during the same tick.
    p_idle_reset, p_idle_mes, p_idle1gate, p_idle2gate can be either float
    (then it will apply depolarizing noise)
    or a namedtuple (then it will apply Pauli channel of the given parameters).
    p_idle_idle is an optional argument to use if you want to add cooldown phase.
    if you want to add a cooldown phase then add an empty TICK in your circuit.
    Typical use is the following :

    p_mes = 0.1
    p_res = 0.1
    p_1 = p_1noise([0.1,0.4,0.])
    p_2 = p_1noise([0.1,0.4,0.])
    build_idle_dict(p_res, p_mes, p_1, p_2)
    """
    idle_dict = {}
    if p_idle_idle is not None:
        idle_dict['cooldown'] = p_idle_idle
    for measurement in MEASURE_OPS:
        idle_dict[measurement] = p_idle_mes
    for reset in RESET_OPS:
        idle_dict[reset] = p_idle_reset
    for gate in ANY_CLIFFORD_1_OPS:
        idle_dict[gate] = p_idle1gate
    for gate in ANY_CLIFFORD_2_OPS:
        idle_dict[gate] = p_idle2gate
    return idle_dict


def unused_qubits(circuit, num_qub):
    """Return unused qubits in the given circuit."""
    # num_qub is in argument to be sure not to forget the last qubit if it is not used.
    used_qubits = []

    for instruction in circuit:
        if isinstance(instruction, stim.Tableau):
            used_qubits.update(range(instruction.num_qubits))
        elif isinstance(instruction, stim.CircuitInstruction):

            targets = instruction.targets_copy()

            used_qubits += [a.qubit_value for a in targets]

    unused_qubits = [qubit for qubit in range(
        num_qub) if qubit not in used_qubits]
    return unused_qubits


def couple_elements(input_list):
    """Return a list of couple based on elements of the input list.

    Couples are formed as (current element, next element) starting from the first element
    and jumping two by two in the list.
    """
    couples_list = []
    for i in range(0, len(input_list) - 1, 2):
        couples_list.append((input_list[i], input_list[i + 1]))
    return couples_list


def flatten_list_of_tuples(tuple_list):
    """Transform a list of tuple in a list."""
    flat_list = []
    for t in tuple_list:
        flat_list.extend(t)
    return flat_list


def _is_Bellprep(op, bell_pairs):
    """Determine wether the given gate is a CNOT used to prepare a Bell pair in the given list or not."""
    Targets = []
    couple_targets = []
    prepared_pairs = []
    status = False
    if (bell_pairs is None) or (bell_pairs == []):
        return False, []
    elif isinstance(op, stim.CircuitInstruction):
        if op.name == "CX":
            Targets = list(map(lambda x: x.value, op.targets_copy()))
            couple_targets = couple_elements(Targets)
            bell_pairs_tuple = couple_elements(bell_pairs)
            for (i, j) in couple_targets:
                if (i, j) in bell_pairs_tuple:
                    prepared_pairs.append(i)
                    prepared_pairs.append(j)
                    status = True
                else:
                    status = status
            return status, prepared_pairs
        else:
            return False, []


def remove_specific_noise(circuit, noise_to_remove, index=[]):
    """Remove ONE noise of type "noise_to_remove" on qubit in the list index."""
    new_circuit = stim.Circuit()
    for instruction in circuit:
        # Check if the instruction correspond to the type of noise we are looking to remove
        if isinstance(instruction, stim.CircuitInstruction) and instruction.name == noise_to_remove:
            targ = list(map(lambda x: x.value, instruction.targets_copy()))
            # Iterate over a copy of index because index will be modified in the loop
            for i in index[:]:
                if i in targ[:]:
                    # We use index.remove to be sure to only remove one H noise in the tick.
                    # If a real H is applied after preparation it will be noisy
                    targ.remove(i)
                    index.remove(i)
                else:
                    continue
            if targ != []:
                new_circuit.append(instruction.name, targ, instruction.gate_args_copy())
        else:
            new_circuit.append_operation(instruction.name, instruction.targets_copy(),
                                         instruction.gate_args_copy())
    return new_circuit


def key_noise_idle(x):
    """key function to compare noises.

    Return the float or the largest p_i in a Pauli noise.
    """
    return x if isinstance(x, numbers.Real) else max(x)


# %% Class NoiseModel
@dataclasses.dataclass(frozen=True)
class NoiseModel:
    """Define the noise model."""

    idle: Union[float, dict]
    noisy_gates: Dict[str, Union[float, p_1noise, p_2noise]]
    any_clifford_1: Optional[Union[float, p_1noise]] = None
    any_clifford_2: Optional[Union[float, p_2noise]] = None
    idle_bell: Optional[Union[float]] = None
    prepa_bell: Optional[Union[float]] = None

    @staticmethod
    def Standard(probas):
        """Build noise model parameters from a namedtuple Probas."""
        return NoiseModel(
            any_clifford_1=probas.onequbitgate,
            any_clifford_2=probas.twoqubitgate,
            idle=probas.idle_data,
            idle_bell=probas.idle_bell,
            prepa_bell=probas.bell,
            noisy_gates={
                "CX": probas.twoqubitgate,
                "R": probas.prep,
                "RX": probas.prep,
                "M": probas.mes,
                "MR": probas.mes,
                "MX": probas.mes,
            },
        )

    def noisy_op(self, op, p, p2=0, prepared_pairs=[], bell_pairs=[]
                 ) -> Tuple[stim.Circuit, stim.Circuit, stim.Circuit]:
        """Put the appropriate noise on a stim operation.

        pre contain noise that we want to be before the gate : e.g for measurement
        mid contain the operation
        post contain noise after the operation : e.g reset, cnot...
        p2 is to deal with the situation when some CNOT during one stim operation
        """
        # Si p2>0 alors on a certaines des cnot qui préparent des paires de Bell
        pre = stim.Circuit()
        mid = stim.Circuit()
        post = stim.Circuit()
        targets = op.targets_copy()
        args = op.gate_args_copy()
        if p2 == 0 and op.name not in ANNOTATION_OPS:
            if op.name in ANY_CLIFFORD_1_OPS:
                if isinstance(p, float):
                    post.append("DEPOLARIZE1", targets, p)
                else:
                    post.append("PAULI_CHANNEL_1", targets, p)
            elif op.name in ANY_CLIFFORD_2_OPS:
                if isinstance(p, float):
                    if not contains_target_rec(targets):
                        post.append("DEPOLARIZE2", targets, p)
                else:
                    post.append("PAULI_CHANNEL_2", targets, p)
            elif op.name in RESET_OPS or op.name in MEASURE_OPS:
                if op.name in RESET_OPS:
                    true_resets = [i.value for i in targets]
                    # Because noise on bell pairs is added later with DEPOL2
                    for i in bell_pairs:
                        if i in true_resets:
                            true_resets.remove(i)
                    post.append("Z_ERROR" if op.name.endswith("X") else "X_ERROR", true_resets, p)
                if op.name in MEASURE_OPS:
                    pre.append_operation("Z_ERROR" if op.name.endswith("X")
                                         else "X_ERROR", targets, p)
            else:
                raise NotImplementedError(repr(op))
        if p2 > 0:
            if op.name in ANY_CLIFFORD_1_OPS:
                if isinstance(p, float):
                    post.append("DEPOLARIZE1", targets, p)
                else:
                    post.append("PAULI_CHANNEL_1", targets, p)
            elif op.name in ANY_CLIFFORD_2_OPS:
                if op.name == "CX":
                    # Spot CNOT that prepare bell pairs and those that are regular cnots
                    index_targets_non_prep = list(map(lambda x: x.value, targets))
                    for i in prepared_pairs:
                        if i in index_targets_non_prep:
                            index_targets_non_prep.remove(i)
                    # index_targets_non_prep contains the list of CNOT target that are not bell pair preparation
                    post.append("DEPOLARIZE2", prepared_pairs, p2)
                    if index_targets_non_prep != []:
                        if isinstance(p, float):
                            post.append("DEPOLARIZE2", index_targets_non_prep, p)
                        else:
                            post.append("PAULI_CHANNEL_2", index_targets_non_prep, p)
                else:
                    if not contains_target_rec(targets):
                        post.append_operation("DEPOLARIZE2", targets, p)
            elif op.name in RESET_OPS or op.name in MEASURE_OPS:
                if op.name in RESET_OPS:
                    post.append_operation("Z_ERROR" if op.name.endswith("X")
                                          else "X_ERROR", targets, p)
                if op.name in MEASURE_OPS:
                    pre.append_operation("Z_ERROR" if op.name.endswith("X")
                                         else "X_ERROR", targets, p)
            else:
                raise NotImplementedError(repr(op))
        mid.append_operation(op.name, targets, args)
        return pre, mid, post

    def table_idle(self, circuit, auto_push_prep=True, auto_pull_mes=True,
                   auto_push_prep_bell=False, auto_pull_mes_bell=False, bell_pairs=[]):
        """Create a table to show true idle qubits.

        The argument of the table are tick and index and return true if the qubit at that index
        is unused during that tick.
        """
        idle_table = []
        num_qub = circuit.num_qubits
        indices = list(range(num_qub))
        bloc = stim.Circuit()
        temporary = []
        couple_bell_pairs = couple_elements(bell_pairs)

        def push_prep():
            # Push virtually the preparation of bell pairs by avoiding useless idling
            # It consider that the idle_table first tick is preparation tick
            nonlocal idle_table
            if auto_push_prep_bell is True:
                for i in indices:
                    # if there is teleported ancillae
                    if bell_pairs != []:
                        if i in bell_pairs and len(idle_table) > 2:
                            # because bell prep takes two time steps
                            j = 2
                            if (i, i+1) in couple_bell_pairs:
                                while (idle_table[j][i] is True) and (idle_table[j][i+1] is True):
                                    idle_table[j][i] = False
                                    idle_table[j][i+1] = False
                                    j += 1
                            elif (i-1, i) in couple_bell_pairs:
                                while (idle_table[j][i-1] is True) and (idle_table[j][i] is True):
                                    idle_table[j][i] = False
                                    idle_table[j][i-1] = False
                                    j += 1
                            else:
                                print(f"Can't find Bell partner of {i}")
                        elif auto_push_prep is True:
                            j = 1
                            while idle_table[j][i] is True:
                                idle_table[j][i] = False
                                # Ensure that we are not going to a tick that doesn't exist,
                                # this happen if you didn't finish your circuit by a measurement for exemple
                                if j < len(idle_table)-1:
                                    j += 1
                                else:
                                    break
                            continue
            elif auto_push_prep is True:
                for i in indices:
                    if i not in bell_pairs:
                        j = 1
                        while idle_table[j][i] is True:
                            idle_table[j][i] = False
                            if j < len(idle_table)-1:
                                j += 1
                            else:
                                break
                        continue

        def pull_mes():
            # Pull virtually the measurement by avoiding useless idling
            # The way it works on bell pairs is very specific to stabilizer measurement.
            # One may completely avoid the bell specific case but it is used for a project,
            # the true way of pulling measurement -even on bell pairs- is by
            # activating auto_pull_mes
            nonlocal idle_table
            if auto_pull_mes_bell is True:
                for i in indices:
                    # if there is teleported ancillae
                    if bell_pairs != []:
                        if i in bell_pairs:
                            # measurement step is not yet in idle_table
                            j = 1
                            if (i, i+1) in couple_bell_pairs and len(idle_table) > 2:
                                while (idle_table[-j][i] is True) and (idle_table[-j][i+1] is True):
                                    idle_table[-j][i] = False
                                    idle_table[-j][i+1] = False
                                    j += 1
                            elif (i-1, i) in couple_bell_pairs and len(idle_table) > 2:
                                while (idle_table[-j][i-1] is True) and (idle_table[-j][i] is True):
                                    idle_table[-j][i] = False
                                    idle_table[-j][i-1] = False
                                    j += 1
                            # to also cover the case were we do a Hadamard before measurement.
                            # works bc tick before measurement in a surface code is always an
                            # hadamard or nothing
                            j = 2
                            if (i, i+1) in couple_bell_pairs and len(idle_table) > 2:
                                while (idle_table[-j][i] is True) and (idle_table[-j][i+1] is True):
                                    idle_table[-j][i] = False
                                    idle_table[-j][i+1] = False
                                    j += 1
                            elif (i-1, i) in couple_bell_pairs and len(idle_table) > 2:
                                while (idle_table[-j][i-1] is True) and (idle_table[-j][i] is True):
                                    idle_table[-j][i] = False
                                    idle_table[-j][i-1] = False
                                    j += 1
                            elif len(idle_table) > 2:
                                print(f"Can't find Bell partner of {i}")
                        elif auto_pull_mes is True:
                            # Uncomment if you also want to push data and ancillae prep at each cycle
                            j = 1
                            while idle_table[-j][i] is True:
                                idle_table[-j][i] = False
                                if j < len(idle_table)-1:
                                    j += 1
                                else:
                                    break
                            continue
            elif auto_pull_mes is True:
                for i in indices:
                    if i not in bell_pairs:
                        # j = 1 because H is a time step before measurement
                        j = 1
                        while idle_table[-j][i] is True:
                            idle_table[-j][i] = False
                            if j < len(idle_table)-1:
                                j += 1
                            else:
                                break
                        continue

        def flush_2():
            # Initialise the table by identifying idle qubits without pushing or pulling prep
            nonlocal idle_table
            unused = unused_qubits(bloc, num_qub)
            for i in indices:
                if i in unused:
                    temporary.append(True)
                else:
                    temporary.append(False)
            idle_table.append(copy.deepcopy(temporary))
            temporary.clear()
            bloc.clear()
        # Loop to separate bloc between tick. idle noise is added at the end of a tick or at the end of the circuit
        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock):
                # If it is a rep block then use a recursive implementation
                flush_2()
                idle_table += self.table_idle(op.body_copy(), auto_push_prep=auto_push_prep,
                                              auto_pull_mes=auto_pull_mes,
                                              auto_push_prep_bell=auto_push_prep_bell,
                                              auto_pull_mes_bell=auto_pull_mes_bell,
                                              bell_pairs=bell_pairs) * op.repeat_count
            elif isinstance(op, stim.CircuitInstruction):
                if op.name == "TICK":
                    # At each tick we identify unused qubits during the last tick
                    flush_2()
                    continue

                elif op.name in self.noisy_gates or op.name in ANY_CLIFFORD_1_OPS or op.name in ANY_CLIFFORD_2_OPS:
                    bloc.append_operation(op)
                    if op.name in MEASURE_OPS and (auto_pull_mes or auto_pull_mes_bell) is True:
                        pull_mes()
                elif op.name in ANNOTATION_OPS:
                    continue
                elif op.name in NOISE_OPS:
                    continue
                else:
                    raise NotImplementedError(repr(op))
            else:
                raise NotImplementedError(repr(op))
        flush_2()
        # Initialisation of idle_table is done, it is a table with one column per tick and one line per qubit,
        # the value of an element is True if the qubit is idle during that tick and false otherwise.
        push_prep()
        return idle_table

    def noisy_circuit_v2(self, circuit, auto_push_prep=True, auto_pull_mes=True,
                         auto_push_prep_bell=False, auto_pull_mes_bell=False,
                         key_idle=key_noise_idle, bell_pairs=[], slow_prep=False) -> (stim.Circuit, int):
        """Add noise on a circuit and return its tick length.

        Useful to add noise recursively on a circuit.
        Tick is useful to know where you are in idle_table
        If there is an error gate in the given circuit the function won't add noise on the target
        of the error gate already existing during the tick where the error gate happen.
        It is a feature to help customize error models to do some performance analysis
        When you want to use this feature, make sure your error gates are the last operations of the given tick.
        You can for exemple add a noise gate with probability 0 if you want to avoid noise on a precise qubit.
        """
        result = stim.Circuit()
        current_moment_pre = stim.Circuit()
        current_moment_mid = stim.Circuit()
        current_moment_post = stim.Circuit()
        idle_table = self.table_idle(circuit, auto_push_prep=auto_push_prep,
                                     auto_pull_mes=auto_pull_mes,
                                     auto_push_prep_bell=auto_push_prep_bell,
                                     auto_pull_mes_bell=auto_pull_mes_bell, bell_pairs=bell_pairs)
        tick = 0
        num_qub = circuit.num_qubits

        def flush():
            # Add idle noise when called
            nonlocal result
            if not current_moment_mid:
                # If cooldown option has not been activated : skip this empty bloc
                if isinstance(self.idle, float):
                    return
                elif 'cooldown' not in self.idle.keys():
                    return
            idles = unused_qubits(current_moment_mid, num_qub)
            for index in idles:
                if (index not in bell_pairs) and (idle_table[tick][index] is True):
                    if isinstance(self.idle, float):
                        if slow_prep:
                            # If bell prep last two time step
                            if any([_is_Bellprep(op, bell_pairs)[0] for op in current_moment_mid]):
                                current_moment_post.append("DEPOLARIZE1", index, self.idle)
                                current_moment_post.append("DEPOLARIZE1", index, self.idle)
                            else:
                                current_moment_post.append("DEPOLARIZE1", index, self.idle)
                        else:
                            current_moment_post.append("DEPOLARIZE1", index, self.idle)
                    elif type(self.idle) == dict:
                        # If the circuit bloc is empty it is a cooldown noise
                        if not current_moment_mid:
                            if isinstance(self.idle['cooldown'], float):
                                current_moment_post.append(
                                    "DEPOLARIZE1", index, self.idle['cooldown'])
                            else:
                                current_moment_post.append(
                                    "PAULI_CHANNEL_1", index, self.idle['cooldown'])
                        else:
                            max_idle = max((self.idle[op.name] for op in current_moment_mid
                                            if op.name in self.idle.keys()),
                                           key=key_idle)
                            # if max_model is a float then we add depol noise otherwise
                            # it is a namedtuple and we apply 1 qubit pauli channel
                            if type(max_idle) == float:
                                current_moment_post.append("DEPOLARIZE1", index, max_idle)
                            else:
                                current_moment_post.append("PAULI_CHANNEL_1", index, max_idle)
                # Not completely general but avoid the case where idle noise is added whereas
                # we can wait to prepare or measure earlier.
                elif (index in bell_pairs) and (idle_table[tick][index] is True):
                    current_moment_post.append("DEPOLARIZE1", index, self.idle_bell)
            # Move current noisy moment into result.
            result += current_moment_pre
            result += current_moment_mid
            result += current_moment_post
            current_moment_pre.clear()
            current_moment_mid.clear()
            current_moment_post.clear()
        # Loop to rebuild each bloc between two tick but now with noise on gate and idle
        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock):
                # If it is a rep block then use a recursive implementation
                flush()
                tick += 1
                repeat_result, tick_repeat = self.noisy_circuit_v2(
                    op.body_copy(), auto_push_prep=auto_push_prep, auto_pull_mes=auto_pull_mes,
                    auto_push_prep_bell=auto_push_prep_bell,
                    auto_pull_mes_bell=auto_pull_mes_bell, bell_pairs=bell_pairs, slow_prep=slow_prep)
                result += repeat_result * op.repeat_count
                tick += tick_repeat * op.repeat_count
            elif isinstance(op, stim.CircuitInstruction):
                if op.name == "TICK":
                    # At each tick we add idle noise on the previous block of circuit
                    flush()
                    tick += 1
                    result.append_operation("TICK", [])
                    continue

                elif op.name in self.noisy_gates:
                    # Check if this CNOT is preparing a Bell pair
                    if op.name == "CX":
                        # Test if some of the CNOT are preparing Bell pairs
                        if _is_Bellprep(op, bell_pairs)[0]:
                            # Remove noise on the H to prepare Bell pairs
                            current_moment_post = remove_specific_noise(
                                current_moment_post, "DEPOLARIZE1", _is_Bellprep(op, bell_pairs)[1])
                            p2 = self.prepa_bell
                            p1 = self.noisy_gates[op.name]
                            # Two arguments for the situation where a bell pairs is prepared during
                            # the same time as other cnot are performed
                            pre, mid, post = self.noisy_op(
                                op, p1, p2, _is_Bellprep(op, bell_pairs)[1])
                        else:
                            p = self.noisy_gates[op.name]
                            pre, mid, post = self.noisy_op(op, p)
                    else:
                        p = self.noisy_gates[op.name]
                        pre, mid, post = self.noisy_op(op, p, bell_pairs=bell_pairs)
                elif self.any_clifford_1 is not None and op.name in ANY_CLIFFORD_1_OPS:
                    p = self.any_clifford_1
                    pre, mid, post = self.noisy_op(op, p)
                elif self.any_clifford_2 is not None and op.name in ANY_CLIFFORD_2_OPS:
                    p = self.any_clifford_2
                    pre, mid, post = self.noisy_op(op, p)
                elif op.name in ANNOTATION_OPS:
                    p = 0
                    pre, mid, post = self.noisy_op(op, p)
                elif op.name in NOISE_OPS:
                    p = 0
                    Targets = list(map(lambda x: x.value, op.targets_copy()))
                    for error in NOISE_OPS:
                        current_moment_post = remove_specific_noise(
                            current_moment_post, error, Targets)
                else:
                    raise NotImplementedError(repr(op))
                current_moment_pre += pre
                current_moment_mid += mid
                current_moment_post += post
                pre.clear()
                mid.clear()
                post.clear()
            else:
                raise NotImplementedError(repr(op))
        # If the circuit is finished, add idle noise on the last bloc
        flush()
        return result, tick+1

    def noisy_circuit(self, circuit, auto_push_prep=False, auto_pull_mes=False, auto_push_prep_bell=False,
                      auto_pull_mes_bell=False, key_idle=key_noise_idle, bell_pairs=[], slow_prep=False) -> stim.Circuit:
        """Take a stim noiseles circuit and make it noisy.

        Before anything else if you don't know about stim circuit and gates, check
        https://github.com/quantumlib/Stim/tree/main/doc
        By default virtually push  preparation and virtually meaning that
        it won't move the prep gate in the circuit but won't apply idle noise until first
        true gate is applied.
        Auto pull will do the same by not adding noise after the last true gate on the qubit.

        The list Bell_pairs is a list of the index of all qubits that are in a bell_pairs.
        Put it empty if you don't use bell pairs.
        if you want to also push prep or pull mes on bell pairs then you need to activate resp.
        auto_push_prep_bell and auto_pull_mes_bell.
        If you want to push/pull bell ONLY (very specific use not really meaningful physically) then
        use auto_push_prep_Bell=True AND auto_push_prep=False (resp. pull)

        To use the fonction proceed as follow :
        import noise_v2
        circuit = stim.Circuit()
        [Build your noiseless circuit, when you want a noiseless gate only on a specific qubit,
         add a DEPOL1 noise gate with probability 0 on the qubit you want to protect at the end of
         the tick during which you want to protect it]

        # Define the namedtuple probas put the noise of idle_bell and bell to p=0. if you don't use bell pairs

        p = 1e-3

        # Build 1 qubit and 2 qubit gate Pauli noise
        p1 = noise_v2.p_1noise(p, 2*p, 3*p)
        p2 = noise_v2.p_2noise(p, 2*p, 3*p, 2*p, 3*p, 2*p, 3*p, 2*p, 3*p, 2*p, 3*p, 2*p, 3*p, 2*p, 3*p)

        # The entry of Probas can be float if you want to apply depolarizing noise on that gate
        # or p_1noise or p_2noise if you want PAULI noise (see right above how to use them).
        # The idle entry can be a dict if you want idle noise to depend on the gate that are 
        performed during the tick
        # [If you want to change what is the "biggest iddle" criteria change the argument key_idle]
        # build_idle_dict is a function to build this dictionnary

        p1gate = noise_v2.p_1noise(p, 2*p, 3*p)
        p2gate = noise_v2.p_1noise(p, 5*p, 3*p)
        p_idle_idle = 9*p
        # Build the dict of different idle noise
        p_idle = noise_v2.build_idle_dict(2*p, p, p1gate, p2gate, p_idle_idle)

        probas = noise_v2.Probas(p1, p_idle, 0, p2, p, p, 0)
        noisy_circuit = noise_v2.NoiseModel.Standard(probas).noisy_circuit(clean_circuit, bell_pairs=bell)
        """
        result, tick = self.noisy_circuit_v2(
            circuit, auto_push_prep=auto_push_prep, auto_pull_mes=auto_pull_mes,
            auto_push_prep_bell=auto_push_prep_bell, auto_pull_mes_bell=auto_pull_mes_bell,
            key_idle=key_idle, bell_pairs=bell_pairs, slow_prep=slow_prep)
        return result

#!/usr/bin/env python3
# coding: utf-8
# pylint: disable=C0103, C0144
"""
Tooling to describe error corrections.

Created on Mon Feb  8 16:52:44 2021
@author: Élie Gouzien
"""

from math import exp, log, floor, ceil, sqrt, pi
# from math import isclose
from abc import ABC, abstractmethod
from warnings import warn
import numpy as np
from tools import Params, PhysicalCost
from functools import lru_cache
import pickle


# %% Abstract classes that contain implementation details
class ErrCorrCode(ABC):
    """
    Abstract class to describe error correction.

    The default implementation assumes that measuring X or Z has the same cost.
    Similarly, CZ is assumed to cost the same as CNOT.
    """

    def __new__(cls, params: Params, *args, **kwargs):
        """Creates an instance using params.type."""
        if cls is ErrCorrCode:
            if params.type == 'surface':
                return SurfaceCode(params, *args, **kwargs)
            elif params.type == 'surface_free_CCZ':
                return SurfaceCodeFreeCCZ(params, *args, **kwargs)
            elif params.type == 'surface_small_procs':
                return SurfaceCodeSmallProcs(params, *args, **kwargs)
            elif params.type == 'surface_small_procs_compact':
                return SurfaceCodeSmallProcs_CompactLayout(params, *args, **kwargs)
            elif params.type == 'surface_small_procs_compact_v2':
                return SurfaceCodeSmallProcs_CompactLayout_V2(params, *args, **kwargs)
            elif params.type == 'surface_big_procs':
                return SurfaceCodeBigProcs(params, *args, **kwargs)
            else:
                raise ValueError("params.type n'est pas correct !")
        return super().__new__(cls)

    @abstractmethod
    def __init__(self, params: Params):
        """Create the instance of the code."""
        self.params = params
        # Cost of the gates
        self.gate1 = None  # In practice, only X or Z by default
        self.cnot = None   # Also cost of CZ
        self.init = None   # Initialization of a logical qubit
        self.mesure = None  # Measurement of the logical qubit
        # Processor properties
        self.correct_time = None  # For 1 logical qubit
        self.proc_qubits = None
        # Number of physical qubits per logical qubit
        self.memory_qubits = None
        self.space_modes = None
        self.time_modes = None
        # Set and check for special parameters
        self._set_measure_based_deand()
        self._check_measure_based_deand()
        self._check_mesure_based_unlookup()
        self._check_shor_no_s()
        # Layout properties for small processors
        self.nx = None
        self.ny = None

    def _check_shor_no_s(self):
        """Check that the 's' option is not used with Shor's algorithm."""
        if self.params.algo.algo == 'Shor' and self.params.algo.s is not None:
            raise ValueError("Shor's algorithm don't have tradeoff s param.")

    @property
    def ne(self):
        """Compute the number of times the elementary algorithm is repeated.

        Corresponds to the size of the register to which the QFT is applied.
        Size of the exponent for factorization or the multiplier for
        elliptic curve discrete logarithm.
        """
        n, s = self.params.algo.n, self.params.algo.s
        if self.params.algo.prob == 'rsa':
            if self.params.algo.algo == 'Shor':
                return 2*n
            elif self.params.algo.algo == 'Ekera':
                m = ceil(n/2) - 1
                # Cas s=1 : voir A.2.1 de
                # Ekeraa2017postprocessingquantum pour les détails.
                delta = 20 if n >= 1024 else 0  # Only for s == 1.
                l = m - delta if s == 1 else ceil(m/s)
                return m + 2*l
            else:
                raise ValueError("'self.params.algo.algo' : 'Shor' or 'Ekera'")
        elif self.params.algo.prob == 'elliptic_log':
            if self.params.algo.algo == 'Shor':
                return 2*n
            elif self.params.algo.algo == 'Ekera':
                return n + 2*ceil(n/s)
            else:
                raise ValueError("'self.params.algo.algo' : 'Shor' or 'Ekera'")
        else:
            raise ValueError("'self.params.algo.prob' must be 'rsa' or "
                             "'elliptic_log'!")

    @property
    def and_gate(self):
        """Cost of computing an auxiliary AND qubit."""
        # |T> = T |+>, preparation of |+> at the same cost as |0>
        return self.init + self.gate1*6 + self.cnot*3

    @property
    def deand(self):
        """Cost of an AND uncomputation, based on measurement."""
        # Hadamard gates are fused with preparations/measurements.
        # Indeed, with a CSS code, measuring X and Z have the same cost.
        return self.mesure + 0.5*self.cnot  # CZ assumed as CNOT

    def _set_measure_based_deand(self, val=True):
        """If 'measure_based_deand' is None, set the correct value."""
        # TODO: potentially implement a system for default values.
        if self.params.algo.mesure_based_deand is None:
            self.params = self.params._replace(algo=self.params.algo._replace(
                mesure_based_deand=val))

    def _check_measure_based_deand(self):
        """Check consistency of the deand method with the parameter."""
        if not self.params.algo.mesure_based_deand:
            raise ValueError("params.algo.mesure_based_deand must be true!")

    @property
    def and_deand(self):
        """Cost of an AND computation/uncomputation."""
        return self.and_gate + self.deand

    @property
    def toffoli(self):
        """Cost of a full Toffoli gate."""
        try:
            return self._toffoli
        except AttributeError:
            return self.and_deand + self.cnot

    @toffoli.setter
    def toffoli(self, value):
        self._toffoli = value

    @toffoli.deleter
    def toffoli(self):
        del self._toffoli

    @property
    def fredkin(self):
        """Controlled swap."""
        return self.toffoli + 2*self.cnot

    def multi_and_deand(self, nb_and):
        """AND gate between nb_and qubits.

        Note that the generalized Toffoli is typically obtained with an
        additional CNOT (last AND + CNOT can be merged in a Toffoli if available).
        """
        return (nb_and-1)*self.and_deand

    @property
    def maj(self):
        """Cost of the MAJ operation, with auxiliary qubit."""
        nb_cnots = 2 if self.params.algo.parallel_cnots else 3
        return self.and_gate + nb_cnots*self.cnot

    @property
    def maj_dag(self):
        """Cost of the conjugate MAJ operation, with auxiliary qubit."""
        nb_cnots = 2 if self.params.algo.parallel_cnots else 3
        return self.deand + nb_cnots*self.cnot

    @property
    def uma(self):
        """Cost of the UMA operation, with auxiliary qubit."""
        # HINT: no parallelization of the last CNOT (not enough qubits)
        return 3*self.cnot + self.deand

    @property
    def uma_dag(self):
        """Cost of the UMA operation, with auxiliary qubit."""
        # HINT: no parallelization of the last CNOT (not enough qubits)
        return 3*self.cnot + self.and_gate

    @property
    def uma_ctrl(self):
        """Cost of the controlled UMA operation.

        The presence of an auxiliary qubit is left to the Toffoli implementation.
        """
        nb_cnots = 2 if self.params.algo.parallel_cnots else 3
        return nb_cnots*self.cnot + self.deand + self.toffoli

    def add(self, n=None):
        """Cost of addition (with auxiliary qubits)."""
        if n is None:  # coset representation
            n = self.params.algo.n + self.params.algo.c
        return (n - 2)*(self.maj + self.uma) + 3*self.cnot + self.and_deand

    def add_nomod(self, n=None):
        """Cost of addition (with auxiliary qubits), with carry-out."""
        if n is None:
            n = self.params.algo.n
        return ((n - 2)*(self.maj + self.uma) + self.maj + 3*self.cnot
                + self.and_deand)

    def add_ctrl(self, n=None):
        """Cost of controlled addition, modulo the power of 2."""
        if n is None:
            n = self.params.algo.n
        return ((n - 2)*(self.maj + self.uma_ctrl) + 2*self.cnot
                + self.and_deand + 2*self.toffoli)

    def add_ctrl_nomod(self, n=None):
        """Cost of controlled addition, with carry-out."""
        if n is None:
            n = self.params.algo.n
        return ((n - 1)*(self.maj + self.uma_ctrl) + self.and_deand
                + self.and_gate + self.toffoli)

    def comparison(self, n=None):
        """Cost of comparison, without usage."""
        # TODO: verify once the scheme is complete
        # Trick: subtraction = addition^\dagger + result < 2^n
        if n is None:
            n = self.params.algo.n
        return (n - 1)*(self.uma + self.uma_dag) + self.and_deand + 4*self.cnot

    @property
    def semi_classical_maj(self):
        """Cost of the semi-classical MAJ operation."""
        return self.and_gate + 0.5*self.cnot + self.gate1

    @property
    def semi_classical_maj_dag(self):
        r"""Cost of the semi-classical MAJ^\dagger operation."""
        return self.deand + 0.5*self.cnot + self.gate1

    @property
    def semi_classical_uma(self):
        """Cost of the semi-classical UMA operation."""
        return self.deand + 1.5*self.cnot + 0.5*self.gate1

    @property
    def semi_classical_ctrl_maj(self):
        """Cost of the semi-classical controlled MAJ operation."""
        return self.and_gate + 1.5*self.cnot

    @property
    def semi_classical_ctrl_uma(self):
        """Cost of the semi-classical controlled UMA operation.

        Version of the controlled semi-classical addition 4.
        Warning: not really a semi-classical controlled UMA gate!
        (not the same inputs).
        """
        return self.deand + 2.5*self.cnot

    @property
    def semi_classical_ctrl_uma_true(self):
        """Cost of the semi-classical controlled UMA operation.

        Version of the controlled semi-classical addition 3.
        This one is truly a controlled semi-classical UMA.
        """
        return self.gate1 + 0.5*self.cnot + self.deand + self.toffoli

    def semi_classical_add(self, n=None):
        """Semi-classical addition."""
        if not n >= 3:
            raise ValueError("For this version of addition, n >= 3.")
        return ((n-3)*(self.semi_classical_maj + self.semi_classical_uma)
                + self.toffoli + 2.5*self.cnot + 2*self.gate1)

    def semi_classical_add_nomod(self, n=None):
        """Semi-classical addition with final carry."""
        if not n >= 2:
            raise ValueError("For this version of addition, n >= 2.")
        return ((n-2)*(self.semi_classical_maj + self.semi_classical_uma)
                + 1.5*self.cnot + 1*self.gate1 + self.and_gate)

    def semi_classical_ctrl_add(self, n=None):
        """Cost of the controlled semi-classical addition."""
        if n is None:  # coset representation
            n = self.params.algo.n + self.params.algo.c
        if not n >= 2:
            raise ValueError("For this version of addition, n >= 2.")
        return ((n-2)*(self.semi_classical_ctrl_maj
                       + self.semi_classical_ctrl_uma)
                + 2*self.cnot + 0.5*self.and_deand)

    def semi_classical_ctrl_ctrl_add(self, n=None):
        """Cost of the double-controlled semi-classical addition."""
        return self.and_deand + self.semi_classical_ctrl_add(n)

    def semi_classical_ctrl_add_nomod(self, n=None):
        """Cost of the controlled semi-classical addition with final carry."""
        if not n >= 2:
            raise ValueError("For this version of addition, n >= 2.")
        return ((n-2)*(self.semi_classical_ctrl_maj
                       + self.semi_classical_ctrl_uma)
                + self.semi_classical_ctrl_maj
                + 2*self.cnot + 0.5*self.and_deand)

    def semi_classical_comparison(self, n=None):
        """Semi-classical comparison, without usage.

        Warning: if the usage is to prepare the result, we can save a bit
        compared to what is presented here.
        """
        if n is None:  # coset representation
            n = self.params.algo.n + self.params.algo.c
        if not n >= 1:
            raise ValueError("For this version of comparison, n >= 1.")
        return ((n-1)*(self.semi_classical_maj
                       + self.semi_classical_maj_dag)
                + self.cnot)

    def semi_classical_neg_mod(self, n=None):
        """Modular negation with the classical modulo number."""
        if n is None:
            n = self.params.algo.n
        # TODO: parallelize?
        return n*self.gate1 + self.semi_classical_add(n)

    def semi_classical_ctrl_neg_mod(self, n=None):
        """Modular negation with the classical modulo number."""
        if n is None:
            n = self.params.algo.n
        nb_cnots = 1 if self.params.algo.parallel_cnots else n
        return nb_cnots*self.cnot + self.semi_classical_ctrl_add(n)

    def modular_reduce(self, n=None):
        """Cost of modular reduction (standard representation).

        Cost of the computation |x> -> |x % p> |x // p>.
        x has n+1 qubits as input, and n as output. p is classically known and
        has n qubits.
        """
        if n is None:
            n = self.params.algo.n
        if not n >= 1:
            raise ValueError("For this version of reduction, n >= 1.")
        return ((n-1)*(self.semi_classical_maj
                       + self.semi_classical_ctrl_uma_true)
                + 1.5*self.gate1 + 2*self.cnot + self.and_gate + self.toffoli)

    def add_mod(self, n=None):
        """Cost of modular addition in standard representation.

        Both numbers are quantum, the modulo is classical.
        Works identically in Montgomery representation.
        """
        if n is None:
            n = self.params.algo.n
        return (self.add_nomod(n) + self.modular_reduce(n)
                + self.comparison(n) + self.cnot)

    def rotate(self, n=None, k=None):
        """Qubit rotation, to implement multiplication or division by 2^k.

        Assumed free as only relabelling.
        """
        return PhysicalCost(0, 0)

    def _defaul_lookup_sizes(self):
        """Provides the default values of 'w' and 'n' for the lookup."""
        # total window input size
        w = self.params.algo.we + self.params.algo.wm
        # Numbers read < N: despite coset representation normal size OK.
        n = self.params.algo.n
        return w, n

    def lookup(self, w=None, n=None):
        """Cost of QROM read, address (target) of sizes w (n).

        Initialization of the states is excluded from this function.
        """
        if w is None and n is None:
            w, n = self._defaul_lookup_sizes()
        nb_cnots = 2**w - 2
        nb_cnots += 2**w if self.params.algo.parallel_cnots else 2**w * n/2
        return 2*self.gate1 + nb_cnots*self.cnot + (2**w - 2)*self.and_deand

    def unary_ununairy(self, size=None):
        """Cost of unary iteration followed by its uncomputation.

        Includes initialization and destruction of qubits in the
        unary representation (via self.and_deand).
        """
        # The initial NOT is not counted as it is handled with the initialization of the qubit.
        if size is None:
            size = floor((self.params.algo.we + self.params.algo.wm)/2)
        if not size >= 1:
            raise ValueError("Unary iteration, size >= 1.")
        return self.init + 2*(size-1)*self.cnot + (size-1)*self.and_deand

    def unlookup(self, w=None, n=None):
        """Cost of QROM uncomputation."""
        # Hadamard gates are fused with preparations/measurements.
        if w is None and n is None:
            w, n = self._defaul_lookup_sizes()
        return (n*self.mesure
                + self.unary_ununairy(floor(w/2))
                # + 2*floor(w/2)*self.gate1  # CZ same cost as CNOT
                + self.lookup(w=ceil(w/2), n=floor(w/2)))

    def look_unlookup(self, w=None, n=None):
        """Computation/uncomputation of QROM."""
        if w is None and n is None:
            w, n = self._defaul_lookup_sizes()
        # No initialization because we recycle the target from previous steps.
        # (first initialization neglected).
        # Ancillary qubits are initialized within the subfunctions.
        return self.lookup(w, n) + self.unlookup(w, n)

    def _check_mesure_based_unlookup(self):
        """Check consistency of the deand method with the parameter."""
        if not self.params.algo.mesure_based_unlookup:
            raise ValueError("params.algo.mesure_based_deand must be true!")

    def initialize_coset_reg(self):
        """Coset representation register initialization."""
        # Hadamard gates are fused with preparations/measurements.
        n, m = self.params.algo.n, self.params.algo.c
        return (m*(self.init + self.mesure)
                + m*self.semi_classical_ctrl_add(n+m)
                + 0.5*m*(self.semi_classical_comparison(n+m) + self.gate1))

    def modular_exp_windowed(self):
        """Total cost of windowed modular exponentiation."""
        _, _, _, n, we, wm, c, _, _, _, _ = self.params.algo
        # Factor 2: see Gidney2019Windowedquantumarithmetic figure 6.
        nb = 2 * (self.ne/we) * (n + c)/wm
        classical_error = PhysicalCost(2**(-c), 0)
        return (nb*(self.add() + self.look_unlookup() + classical_error)
                + 2*self.initialize_coset_reg())

    def modular_exp_controlled(self):
        """Total cost of basic controlled modular exponentiation."""
        _, _, _, n, _, _, c, _, _, _, _ = self.params.algo
        nb = 2 * self.ne * (n + c)
        classical_error = PhysicalCost(2**(-c), 0)
        return (nb*(self.semi_classical_ctrl_ctrl_add() + classical_error)
                + 2 * self.initialize_coset_reg()
                + self.ne*(n + c)*(2*self.cnot + self.toffoli))

    def modular_exp(self):
        """Modular exponentiation as requested by the parameters."""
        if self.params.algo.windowed:
            return self.modular_exp_windowed()
        return self.modular_exp_controlled()

    def factorisation(self):
        """Resources for factorization.

        The Fourier transform is neglected. Algorithmic options are used.
        """
        if self.params.algo.algo == 'Shor':
            classical_error = PhysicalCost(1/4, 0)
            return self.modular_exp() + classical_error
        elif self.params.algo.algo == 'Ekera':
            # HINT: by multiplying by s, I consider that no error is tolerated
            # in the quantum processor.
            return self.params.algo.s * self.modular_exp()
        else:
            raise ValueError("'self.params.algo.algo' : 'Shor' or 'Ekera'")

# %% Concrete classes mainly containing the initialization


class SurfaceCode(ErrCorrCode):
    """Surface code with memory."""

    def __init__(self, params: Params):
        """Create an instance of the surface code with memory."""
        super().__init__(params)
        # See Gidney2019Lowoverheadquantum for the explanation of the
        # time for a CNOT.
        d = params.low_level.d  # pylint: disable=C0103
        if not d % 2 == 1:
            raise ValueError("The distance must be odd.")
        self._init_ccz_factory()
        err_1 = self._surf_topological_error_intern(params.low_level.pp, d, 1)
        err_2 = self._surf_topological_error_intern(params.low_level.pp, d, 2)
        self.correct_time = d * params.low_level.tc  # Repeated d times
        self.gate1 = PhysicalCost(0, 0)  # OK if Clifford or H at beginning or end.
        self.cnot = PhysicalCost(err_2, 2 * self.correct_time)  # same for CZ
        self.init = PhysicalCost(err_1, self.correct_time)
        self.mesure = PhysicalCost(err_1, params.low_level.tr)
        # L-shaped layout; factor 2 for measurement qubits.
        self.proc_qubits = 2 * (3 * d ** 2 + 2 * d) + self._factory_qubits
        self.memory_qubits = d ** 2
        self.space_modes = d ** 2
        self.time_modes = 1
        if not self.params.algo.mesure_based_deand:
            raise ValueError("params.algo.mesure_based_deand must be 'True'!")

    @staticmethod
    def _surf_topological_error_intern(proba, distance, nb=1):
        """Error probability due to decoherence."""
        err = 0.1 * (proba / (7.65e-3)) ** ((distance + 1) / 2)
        return 1 - (1 - err) ** nb

    @staticmethod
    def _surf_topological_error_aux(p, p_bell, distance, nb_seam, nx, nb=1):
        """Error probability due to decoherence on a patch with interfaces."""
        alpha1 = 0.25
        alpha2 = 0.062
        pbth = 7.22e-3
        psth = 0.294
        alpha3 = 0.0129
        alphac = 0.245
        alpha1_tot = alpha1 * nb_seam
        alpha3_tot = alpha3 * nb_seam
        # Length of the routing qubit crossing all processors in units of d
        # precisely 3+1 would be (3d + 3 + d) / d
        alpha2_tot = alpha2 * (nx * 3 + 1) * (nb_seam + 1)
        # Errors for X_L
        errx = coupled_error_model_v2(p_bell, p, alpha1_tot, alpha2_tot,
                                      alpha3_tot, psth, pbth, alphac, distance)
        # Errors for Z_L
        distance_z = (nx * 3 + 1) * (nb_seam + 1) * distance
        errz = 0.0537 * (p / pbth) ** ((distance_z + 1) / 2)
        return 1 - (1 - errx - errz) ** nb

    def _init_ccz_factory(self):
        """Compute the parameters of the factories."""
        # Be careful d1 is not a distance, it's a variable that explores the table of
        # referenced factories
        _, d1, _, d, _, tc, _, pp, _ = self.params.low_level
        with open('factories_8toCCZ_qubits_10_fact_new_th.pkl', 'rb') as pkl_file:
            factories = pickle.load(pkl_file)
        L2_total_CCZ = factories[d1]['Output error']
        time = factories[d1]['Codecycles'] * tc
        dx2 = factories[d1]['dx2']
        dz2 = factories[d1]['dz2']
        dm2 = factories[d1]['dm2']
        self._factory = PhysicalCost(L2_total_CCZ, time)
        # Ajoute un espace de routing supplémentaire pour l'injection de l'état CCZ
        self._factory_qubits = factories[d1]['Qubits'] + 2*((d-dx2)*(3*dx2+dz2+dm2))+d+d/2

    @property
    def ccz_interact(self):
        """Interaction between the magic state and the target.

        I consider here that no parallelization is possible.
        """
        # Note: I do not take into account decoherence of fixup qubits
        # during the application of the CNOTs.
        return 3 * self.cnot

    @property
    def ccz_fixup(self):
        """Characteristics of the 'fixup' step."""
        # See Fowler2019Flexiblelayoutsurface Fig.4 for the approximate explanation
        _, _, _, d, _, tc, tr, pp, _ = self.params.low_level
        return PhysicalCost(self._surf_topological_error_intern(pp, d, 6 * tr / tc), 2 * tr)

    @property
    def ccz_interact_fixup(self):
        """Characteristics of the interaction step followed by 'fixup'."""
        # Warning, we are not using "time optimal quantum computing" from Fowler;
        # for that, it would be necessary to propagate the carry in space and
        # not in time, as explained in Fowler2019Flexiblelayoutsurface.
        # Anyway, with only one factory, we are certainly limited by
        # the generation of magic states.
        # In general, the fixup can be done in parallel with the next operation.
        return self.ccz_interact + self.ccz_fixup

    @property
    def and_gate(self):
        """Cost of computing an auxiliary AND qubit.

        Warning: the preparation of the magic state can generally be parallelized.
        """
        return self._factory + self.ccz_interact_fixup

    # For the following operations, we consider the parallelization of the
    # preparation of the magic state with the CNOT gates; the fixup being done
    # in parallel with the CNOTs of the next block (not exact at the edges).
    # Additionally, we potentially use schemas where the depth is reduced
    # at the cost of more gates.
    # The Toffolis required for controlled gates are not simplified, however.
    # Complete circuits are not detailed again even if there would be a
    # possibility of marginal improvement by parallelizing part of the ANDs
    # at the beginning and end with what surrounds them.
    @property
    def maj(self):
        """Cost of the MAJ operation."""
        nb_cnots = 2 if self.params.algo.parallel_cnots else 3
        return ((self._factory | (nb_cnots * self.cnot) | self.ccz_fixup)
                + self.ccz_interact)

    @property
    def uma_dag(self):
        """Cost of the UMA operation with an auxiliary qubit."""
        # HINT: no parallelization of the last CNOT (not enough qubits)
        return ((self._factory | (3 * self.cnot) | self.ccz_fixup)
                + self.ccz_interact)

    @property
    def semi_classical_maj(self):
        """Cost of the semi-classical MAJ operation."""
        # The NOT gates (X) are not applied, they are followed classically
        return ((self._factory | (0.5 * self.cnot + self.gate1) | self.ccz_fixup)
                + self.ccz_interact)

    @property
    def semi_classical_ctrl_maj(self):
        """Cost of the controlled semi-classical MAJ operation."""
        return ((self._factory | (1.5 * self.cnot) | self.ccz_fixup)
                + self.ccz_interact)


class SurfaceCodeFreeCCZ(SurfaceCode):
    """Surface code with CCZ that costs the same as a CNOT.

    The idea is to evaluate what would happen without distillation.
    """

    @property
    def _init_ccz_factory(self):
        """Calculates the characteristics of the distillation."""
        self._factory = PhysicalCost(0, 0)
        self._factory_qubits = 0

    @property
    def ccz_interact(self):
        """Interaction between the magic state and the target."""
        return self.cnot + PhysicalCost(self._surf_topological_error_intern(
            self.params.low_level.pp, self.params.low_level.d, 1), 0)

    @property
    def ccz_fixup(self):
        """Characteristics of the 'fixup' step."""
        return PhysicalCost(0, 0)


class SurfaceCodeSmallProcs(SurfaceCode):
    """Surface code, with a set of small processors.

    Uses Litinski factories.
    First non-optimized Layout.
    """

    def __init__(self, params: Params):
        """Create the instance of the surface code."""
        super(__class__.__bases__[0], self).__init__(params)
        _, _, _, d, _, tc, tr, pp, pbell = self.params.low_level
        # See Gidney2019Lowoverheadquantum for the time of a CNOT.
        if not d % 2 == 1:
            raise ValueError("The distance must be odd.")
        self._init_ccz_factory()
        log_qubits = logical_qubits(params, verb=False)

        # Test to use brute force to find optimal parameters
        nx, ny, nb_log_qbit_per_proc = find_max_product(
            self._factory_qubits, lambda nx, ny: proc_qubits_each(nx, ny, d))

        # Number of processors required to handle the number of logical qubits
        self.nb_procs = ceil(log_qubits / nb_log_qbit_per_proc) + 1
        self.proc_qubits_each = proc_qubits_each(nx, ny, d)
        self.proc_qubits = (self.proc_qubits_each * (self.nb_procs - 1)
                            + self._factory_qubits)
        self.memory_qubits = None
        self.space_modes = None
        self.time_modes = None
        self.nx = nx
        self.ny = ny

        # Errors accumulate on unused qubits as well
        # Definitions of the different error rates per cycle here:
        # Error rate of a giant logical routing patch for an auxiliary qubit in lattice surgery.
        # nb_seam = nb_procs - the factory - 1
        nb_link_between_procs = ceil(log_qubits / nb_log_qbit_per_proc) - 1

        # Error estimations:
        err_aux = self._surf_topological_error_aux(pp, pbell, d, nb_link_between_procs, nx, 1)
        err_internal = self._surf_topological_error_intern(pp, d, log_qubits)
        err_internal_aux = self._surf_topological_error_intern(pp, d, 1)

        # Correction time (repeated `d` times)
        self.correct_time = d * tc

        # Gate cost estimations
        self.gate1 = PhysicalCost(0, 0)  # OK if Clifford or H at the beginning or end.

        # Same for CZ, in parallel noise on the auxiliary during d time steps, noise on all
        # qubits of the processor during 2d time steps, and noise on the internal auxiliary during d
        # time steps
        self.cnot = (d * PhysicalCost(err_aux, tc)) | (2 * d * PhysicalCost(err_internal, tc)) | (
            d * PhysicalCost(err_internal_aux, tc))

        # We only initialize logical qubits of size d² internal to a processor outside the context
        # of lattice surgery, which intervenes for the CNOT
        self.init = d * PhysicalCost(err_internal, tc)

        # We only measure logical qubits of size d² internal to a processor outside the context
        # of lattice surgery, which intervenes for the CNOT
        self.mesure = tr / tc * PhysicalCost(err_internal, tc)

        if not self.params.algo.mesure_based_deand:
            raise ValueError("params.algo.mesure_based_deand must be 'True'!")


class SurfaceCodeSmallProcs_CompactLayout(SurfaceCode):
    """Surface code, with a set of small processors.

    Uses Litinski factories and the new layout with qubits on the sides.
    Here we use two distillation factories to allow parallel preparation.
    Direct Toffoli injection.
    With 4 d time steps for the CNOT.
    """

    def __init__(self, params: Params):
        """Create the instance of the surface code."""
        # We don't call SurfaceCode.__init__() but from its parent.
        super(__class__.__bases__[0], self).__init__(params)
        # See Gidney2019Lowoverheadquantum for the time of a CNOT.
        _, _, _, d, _, tc, tr, pp, pbell = self.params.low_level
        if not d % 2 == 1:
            raise ValueError("The distance must be odd.")
        self._init_ccz_factory()
        log_qubits = logical_qubits(params, verb=False)

        # Determine the layout size and the number of logical qubits per processor
        ny, nb_log_qbit_per_proc = find_size_layout_v2(
            self._factory_qubits, lambda ny: proc_qubits_each_v2(ny, d)
        )

        # Calculate the number of processors and the required qubits
        self.nb_procs = ceil(log_qubits / nb_log_qbit_per_proc) + 2
        self.proc_qubits_each = proc_qubits_each_v2(ny, d)
        self.proc_qubits = (self.proc_qubits_each * (self.nb_procs - 2)
                            + 2 * self._factory_qubits)
        self.memory_qubits = None
        self.space_modes = None
        self.time_modes = None
        self.ny = ny

        # Errors also accumulate on unused qubits
        # Error rate of a giant logical routing patch for an auxiliary qubit in lattice surgery.
        # nb_seam = nb_procs - 1
        nb_link_between_procs = ceil(log_qubits / nb_log_qbit_per_proc) - 1

        # Compute error rates
        err_aux = self._surf_topological_error_aux(pp, pbell, d,
                                                   nb_link_between_procs, ny, 1)
        err_internal = self._surf_topological_error_intern(pp, d, log_qubits)
        err_internal_aux = self._surf_topological_error_inner_aux(pp, pbell, d,
                                                                  nb_link_between_procs, ny, 1)

        # Repetition `d` times
        self.correct_time = d * tc

        self.gate1 = PhysicalCost(0, 0)  # OK if Clifford or H at the beginning or end.

        # Same for CZ
        self.cnot = (d * PhysicalCost(err_aux, tc)) | (4 * d * PhysicalCost(err_internal, tc)) | (
            d * PhysicalCost(err_internal_aux, tc)
        )

        # We only initialize logical qubits of size d² internal to a processor outside the context
        # of lattice surgery which intervenes for the CNOT
        self.init = d * PhysicalCost(err_internal, tc)

        # We only measure logical qubits of size d² internal to a processor outside the context
        # of lattice surgery which intervenes for the CNOT
        self.mesure = tr / tc * PhysicalCost(err_internal, tc)

    @staticmethod
    def _surf_topological_error_aux(p, p_bell, distance, nb_seam, ny, nb=1):
        """Error probability due to decoherence during a CNOT in layout 2.
        this is the error model for the patch crossing all processors
        in the compact layout."""
        alpha3 = 0.053
        alphac = 0.21
        alpha1 = 0.099
        alpha2 = 0.045
        pbth = 7.2e-3
        psth = 0.298
        alpha1_tot = alpha1 * nb_seam
        alpha3_tot = alpha3 * nb_seam

        # Length of the routing qubit crossing all processors in units of d
        length_aux = (2 * distance + 2) / distance * (nb_seam + 1) + ny * (distance + 1) / distance
        alpha2_tot = alpha2 * length_aux

        # Errors for X_L
        errx = coupled_error_model_v2(p_bell, p, alpha1_tot, alpha2_tot,
                                      alpha3_tot, psth, pbth, alphac, distance)

        # Errors for Z_L
        distance_z = distance
        pbth_reg = 7.43e-3
        errz = 0.05 * (p / pbth_reg) ** ((distance_z + 1) / 2)
        return 1 - (1 - errx - errz) ** nb

    @staticmethod
    def _surf_topological_error_inner_aux(p, distance, ny, nb=1):
        """Error probability due to decoherence during a CNOT in layout 2.
        This is the error model for the merged patch inside a processor."""
        pbth = 7.43e-3
        # Length of the routing qubit spanning the entire processor
        length_aux = (ny + 1) * (distance + 1) / distance

        # Errors for X_L
        errx = 0.05 * (p / pbth) ** ((distance + 1) / 2)

        # Errors for Z_L
        errz = 0.05 * length_aux * (p / pbth) ** ((distance + 1) / 2)
        return 1 - (1 - errx - errz) ** nb

    @property
    def ccz_interact(self):
        """Interaction between the magic state and the target.

        I assume here that no parallelization is possible.
        """
        return 3 * self.cnot

    @property
    def ccz_interact_lattice_surgery(self):
        """Interaction between the magic state and the target via lattice surgery.

        I assume here that no parallelization is possible.
        """
        _, _, _, d, _, tc, tr, pp, pbell = self.params.low_level
        log_qubits = logical_qubits(self.params, verb=False)
        nb_link_between_procs = self.nb_procs - 1
        ny = self.ny
         # Compute error rates
        # Error rate of a giant logical routing patch for an 
        # auxiliary qubit in lattice surgery.
        err_aux = self._surf_topological_error_aux(pp, pbell, d,
                                                   nb_link_between_procs, ny, 1)
        # Errors accumulate on unused qubits as well
        err_internal = self._surf_topological_error_intern(pp, d, log_qubits)
        interaction_error_model = (3*d * PhysicalCost(err_aux, tc)) | (
            7 * d * PhysicalCost(err_internal, tc))
        return interaction_error_model
    
    @property
    def ccz_fixup(self):
        """Characteristics of the 'fixup' step.

        1.5 because there is always 50% probability that the correction is required.
        """
        _, _, _, d, _, tc, tr, pp, _ = self.params.low_level
        return self.mesure + 1.5 * self.cnot

    @property
    def ccz_interact_fixup(self):
        """Characteristics of the interaction step followed by 'fixup'."""
        return self.ccz_interact + self.ccz_fixup

    @property
    def and_gate(self):
        """Cost of computing an auxiliary AND qubit.

        Warning: the preparation of the magic state can generally be parallelized.
        """
        return self._factory | self.ccz_interact_fixup

    @property
    def maj(self):
        """Cost of the MAJ operation."""
        nb_cnots = 2 if self.params.algo.parallel_cnots else 3
        return (self._factory | ((nb_cnots * self.cnot) + self.ccz_interact + self.ccz_fixup))

    @property
    def uma_dag(self):
        """Cost of the UMA operation with an auxiliary qubit."""
        return (self._factory | (3 * self.cnot + self.ccz_fixup + self.ccz_interact))

    @property
    def semi_classical_maj(self):
        """Cost of the semi-classical MAJ operation."""
        return (self._factory | (0.5 * self.cnot + self.gate1 + self.ccz_fixup + self.ccz_interact))

    @property
    def semi_classical_ctrl_maj(self):
        """Cost of the controlled semi-classical MAJ operation."""
        return (self._factory | (1.5 * self.cnot + self.ccz_fixup + self.ccz_interact))


class SurfaceCodeSmallProcs_CompactLayout_V2(SurfaceCode):
    """Surface code, with a set of small processors.

    Uses Litinski factories and the new layout with qubits on the sides.
    Here we use two distillation factories to allow parallel preparation.
    New Toffoli injection via lattice surgery.
    With 4 d time steps for the CNOT.
    """

    def __init__(self, params: Params):
        """Create the instance of the surface code."""
        # We don't call SurfaceCode.__init__() but from its parent.
        super(__class__.__bases__[0], self).__init__(params)
        # See Gidney2019Lowoverheadquantum for the time of a CNOT.
        _, _, _, d, _, tc, tr, pp, pbell = self.params.low_level
        if not d % 2 == 1:
            raise ValueError("The distance must be odd.")
        self._init_ccz_factory()
        log_qubits = logical_qubits(params, verb=False)

        # Determine the layout size and the number of logical qubits per processor
        ny, nb_log_qbit_per_proc = find_size_layout_v2(
            self._factory_qubits, lambda ny: proc_qubits_each_v2(ny, d)
        )

        # Calculate the number of processors and the required qubits
        self.nb_procs = ceil(log_qubits / nb_log_qbit_per_proc) + 2
        self.proc_qubits_each = proc_qubits_each_v2(ny, d)
        self.proc_qubits = (self.proc_qubits_each * (self.nb_procs - 2)
                            + 2 * self._factory_qubits)
        self.memory_qubits = None
        self.space_modes = None
        self.time_modes = None
        self.ny = ny

        # Errors also accumulate on unused qubits
        # Error rate of a giant logical routing patch for an auxiliary qubit in lattice surgery.
        # nb_seam = nb_procs - 1
        nb_link_between_procs = ceil(log_qubits / nb_log_qbit_per_proc) - 1

        # Compute error rates
        err_aux = self._surf_topological_error_aux(pp, pbell, d,
                                                   nb_link_between_procs, ny, 1)
        err_internal = self._surf_topological_error_intern(pp, d, log_qubits)
        err_internal_aux = self._surf_topological_error_inner_aux(pp, d, ny, 1)

        # Repetition `d` times
        self.correct_time = d * tc

        self.gate1 = PhysicalCost(0, 0)  # OK if Clifford or H at the beginning or end.

        # Same for CZ
        self.cnot = (d * PhysicalCost(err_aux, tc)) | (4 * d * PhysicalCost(err_internal, tc)) | (
            d * PhysicalCost(err_internal_aux, tc)
        )

        # We only initialize logical qubits of size d² internal to a processor outside the context
        # of lattice surgery which intervenes for the CNOT
        self.init = d * PhysicalCost(err_internal, tc)

        # We only measure logical qubits of size d² internal to a processor outside the context
        # of lattice surgery which intervenes for the CNOT
        self.mesure = tr / tc * PhysicalCost(err_internal, tc)

    @staticmethod
    def _surf_topological_error_aux(p, p_bell, distance, nb_seam, ny, nb=1):
        """Error probability due to decoherence during a CNOT in layout 2.
        this is the error model for the patch crossing all processors
        in the compact layout."""
        alpha3 = 0.053
        alphac = 0.21
        alpha1 = 0.099
        alpha2 = 0.045
        pbth = 7.2e-3
        psth = 0.298
        alpha1_tot = alpha1 * nb_seam
        alpha3_tot = alpha3 * nb_seam

        # Length of the routing qubit crossing all processors in units of d
        length_aux = (2 * distance + 2) / distance * (nb_seam + 1) + ny * (distance + 1) / distance
        alpha2_tot = alpha2 * length_aux

        # Errors for X_L
        errx = coupled_error_model_v2(p_bell, p, alpha1_tot, alpha2_tot,
                                      alpha3_tot, psth, pbth, alphac, distance)

        # Errors for Z_L
        distance_z = distance
        pbth_reg = 7.43e-3
        errz = 0.05 * (p / pbth_reg) ** ((distance_z + 1) / 2)
        return 1 - (1 - errx - errz) ** nb

    @staticmethod
    def _surf_topological_error_inner_aux(p, distance, ny, nb=1):
        """Error probability due to decoherence during a CNOT in layout 2.
        This is the error model for the merged patch inside a processor."""
        pbth = 7.43e-3
        # Length of the routing qubit spanning the entire processor
        length_aux = (ny + 1) * (distance + 1) / distance

        # Errors for X_L
        errx = 0.05 * (p / pbth) ** ((distance + 1) / 2)

        # Errors for Z_L
        errz = 0.05 * length_aux * (p / pbth) ** ((distance + 1) / 2)
        return 1 - (1 - errx - errz) ** nb

    @property
    def ccz_interact(self):
        """Interaction between the magic state and the target via lattice surgery.

        I assume here that no parallelization is possible.
        """
        _, _, _, d, _, tc, tr, pp, pbell = self.params.low_level
        log_qubits = logical_qubits(self.params, verb=False)
        nb_link_between_procs = self.nb_procs - 1
        ny = self.ny
         # Compute error rates
        # Error rate of a giant logical routing patch for an 
        # auxiliary qubit in lattice surgery.
        err_aux = self._surf_topological_error_aux(pp, pbell, d,
                                                   nb_link_between_procs, ny, 1)
        # Errors accumulate on unused qubits as well
        err_internal = self._surf_topological_error_intern(pp, d, log_qubits)
        interaction_error_model = (3*d * PhysicalCost(err_aux, tc)) | (
            7 * d * PhysicalCost(err_internal, tc))
        return interaction_error_model
    
    @property
    def ccz_fixup(self):
        """Characteristics of the 'fixup' step.

        1.5 because there is always 50% probability that the correction is required.
        """
        _, _, _, d, _, tc, tr, pp, _ = self.params.low_level
        return self.mesure + 1.5 * self.cnot

    @property
    def ccz_interact_fixup(self):
        """Characteristics of the interaction step followed by 'fixup'."""
        return self.ccz_interact + self.ccz_fixup

    @property
    def and_gate(self):
        """Cost of computing an auxiliary AND qubit.

        Warning: the preparation of the magic state can generally be parallelized.
        """
        return self._factory | self.ccz_interact_fixup

    @property
    def maj(self):
        """Cost of the MAJ operation."""
        nb_cnots = 2 if self.params.algo.parallel_cnots else 3
        return (self._factory | ((nb_cnots * self.cnot) + self.ccz_interact + self.ccz_fixup))

    @property
    def uma_dag(self):
        """Cost of the UMA operation with an auxiliary qubit."""
        return (self._factory | (3 * self.cnot + self.ccz_fixup + self.ccz_interact))

    @property
    def semi_classical_maj(self):
        """Cost of the semi-classical MAJ operation."""
        return (self._factory | (0.5 * self.cnot + self.gate1 + self.ccz_fixup + self.ccz_interact))

    @property
    def semi_classical_ctrl_maj(self):
        """Cost of the controlled semi-classical MAJ operation."""
        return (self._factory | (1.5 * self.cnot + self.ccz_fixup + self.ccz_interact))


class SurfaceCodeBigProcs(SurfaceCode):
    """Surface code, with a single large processor.

    Uses two factories and the compact layout.
    """

    def __init__(self, params: Params):
        """Creates the instance of the surface code."""
        # We don't call SurfaceCode.__init__() but from its parent.
        super(__class__.__bases__[0], self).__init__(params)
        # See Gidney2019Lowoverheadquantum for the time of a CNOT.
        _, _, _, d, _, tc, tr, pp, pbell = self.params.low_level
        if not d % 2 == 1:
            raise ValueError("The distance must be odd.")
        self._init_ccz_factory()
        log_qubits = logical_qubits(params, verb=False)

        # Errors also accumulate on unused qubits
        err = self._surf_topological_error_intern(pp, d, log_qubits)
        self.correct_time = d * tc  # Repeated d times
        self.gate1 = PhysicalCost(0, 0)  # OK if Clifford or H at the beginning or end.

        # Calculation of the CNOT cost
        self.cnot = (4 * d * PhysicalCost(err, tc))

        # We only initialize logical qubits of size d² internal to a processor outside the context
        # of lattice surgery, which intervenes for the CNOT
        self.init = d * PhysicalCost(err, tc)

        # We only measure logical qubits of size d² internal to a processor outside the context
        # of lattice surgery, which intervenes for the CNOT
        self.mesure = tr / tc * PhysicalCost(err, tc)

        # Processor size calculation
        nb_log_qbit_per_proc = log_qubits + 2 * ceil((self._factory_qubits + 1) / (
            (3 * d + 2) * (d + 1)) - 2)
        ny = nb_log_qbit_per_proc / 2
        self.nb_procs = 1
        self.proc_qubits_each = proc_qubits_monolythic(log_qubits/2, d)+ 2 * self._factory_qubits
        self.proc_qubits = self.proc_qubits_each 
        self.memory_qubits = None
        self.space_modes = None
        self.time_modes = None
        self.ny = ny

    # @property
    # def ccz_interact(self):
    #     """Interaction between the magic state and the target.

    #     I assume here that no parallelization is possible.
    #     """
    #     # Note: I do not take into account decoherence of the fixup qubits
    #     # while applying the CNOTs.
    #     # The fact that we use CCZ states to make Toffolis does not modify
    #     # this step because it adds a Hadamard that can be done in parallel with
    #     # the first two CNOTs.
    #     return 3 * self.cnot
        
    @property
    def ccz_interact(self):
        """Interaction between the magic state and the target via lattice surgery.

        I assume here that no parallelization is possible.
        """
        _, _, _, d, _, tc, _, pp, _ = self.params.low_level
        log_qubits = logical_qubits(self.params, verb=False)
         # Compute error rates
        # Error rate of a giant logical routing patch for an 
        # auxiliary qubit in lattice surgery that crosses the big processor.
        err_aux = (log_qubits/2*(d+1)/d)*self._surf_topological_error_intern(pp, d, 1)
        # Errors accumulate on unused qubits as well
        err_internal = self._surf_topological_error_intern(pp, d, log_qubits)
        interaction_error_model = (3*d * PhysicalCost(err_aux, tc)) | (
            7 * d * PhysicalCost(err_internal, tc))
        return interaction_error_model

    @property
    def ccz_fixup(self):
        """Characteristics of the 'fixup' step."""
        # CCX injection scheme with Clifford correction, see Elie's paper
        # approximate formula.
        _, _, _, d, _, tc, tr, pp, _ = self.params.low_level

        # Addition of the three Clifford corrections (note: same error model for CX and CZ)
        return self.mesure + 1.5 * self.cnot

    @property
    def ccz_interact_fixup(self):
        """Characteristics of the interaction step followed by 'fixup'."""
        # Warning: we are not using "time optimal quantum computing" from
        # Fowler; to do so it would be necessary to propagate the carry in space and
        # not in time, as explained in Fowler2019Flexiblelayoutsurface.
        # Anyway, with only one factory we are certainly limited by
        # the generation of magic states.
        # Note: in general, the fixup can be done in parallel with
        # the next operation.
        return self.ccz_interact + self.ccz_fixup

    # For the following operations, we consider the parallelization of the
    # preparation of the magic state with the CNOTs, possible because we have two factories;
    @property
    def and_gate(self):
        """Cost of computing an auxiliary AND qubit.

        Warning: the preparation of the magic state can generally be
        parallelized.
        """
        return self._factory | self.ccz_interact_fixup

    @property
    def maj(self):
        """Cost of the MAJ operation."""
        nb_cnots = 2 if self.params.algo.parallel_cnots else 3
        return (self._factory | ((nb_cnots * self.cnot) + self.ccz_interact + self.ccz_fixup))

    @property
    def uma_dag(self):
        """Cost of the UMA operation, with auxiliary qubit."""
        # HINT: no parallelization of the last CNOT (not enough qubits)
        return (self._factory | (3 * self.cnot + self.ccz_fixup + self.ccz_interact))

    @property
    def semi_classical_maj(self):
        """Cost of the semi-classical MAJ operation."""
        # The NOT gates (X) are not applied, they are followed classically
        # 0.5 before the CNOT because it is classically controlled by a bit that has a 50% chance of
        # being 0
        return (self._factory | (0.5 * self.cnot + self.gate1 + self.ccz_fixup + self.ccz_interact))

    @property
    def semi_classical_ctrl_maj(self):
        """Cost of the controlled semi-classical MAJ operation."""
        return (self._factory | (1.5 * self.cnot + self.ccz_fixup + self.ccz_interact))


# %% Fonctions annexes en lien avec l'algorithme
# Mapping between type name and class.
CLASS_MAP = {'surface': SurfaceCode,
             'surface_free_CCZ': SurfaceCodeFreeCCZ,
             'surface_small_procs': SurfaceCodeSmallProcs,
             'surface_small_procs_compact': SurfaceCodeSmallProcs_CompactLayout,
             'surface_small_procs_compact_v2': SurfaceCodeSmallProcs_CompactLayout_V2,
             'surface_big_procs': SurfaceCodeBigProcs
             }


def logical_qubits_exp_mod(params: Params, verb=True):
    """Total number of logical qubits for modular exponentiation.

    Optionally, it is given for the windowed algorithm.
    """
    if params.algo.windowed:
        # 11.4.0.3
        res = 4 * params.algo.n + 3 * params.algo.c + params.algo.we - 1
    elif not params.algo.windowed:
        res = 3 * (params.algo.n + params.algo.c) + 1

    if verb:
        print("Total number of logical qubits:", res)
    return res


def logical_qubits(params: Params, verb=True):
    """Total number of logical qubits.

    Selects the problem based on the parameters.
    """
    if params.algo.prob == 'rsa':
        # If the problem is RSA, we compute with modular exponentiation
        res = logical_qubits_exp_mod(params, verb)
    else:
        # If the problem is unknown, raise an error
        raise ValueError("Unknown problem type!")

    # Special case: if the algorithm type is 'alice&bob3' and the number of logical qubits is odd,
    # we increment it to make it even.
    if params.type == 'alice&bob3' and res % 2 == 1:
        res += 1  # enforce even logical qubits for alice&bob3

    return res


def logical_error_model(x, pth, alpha, d):
    """Error model for a regular surface code."""
    return (alpha*(x/pth)**((d+1)/2))


def decoupled_error_model(ps, pb, psth, pbth, alpha1, alpha2, d):
    """Logical error model with a seam for a naive ansatz.

    when we don't consider errors leaving the seam and coming back.
    """
    return (logical_error_model(ps, psth, alpha1, d)+logical_error_model(pb, pbth, alpha2, d))


def coupled_error_threshold_v2(pb, psth, pbth, alphac):
    """Modify threshold by excursion in the bulk."""
    return (psth/(1+alphac/(1-np.sqrt(pb/pbth)))**2)


def coupled_error_model_v2(ps, pb, alpha1, alpha2, alpha3, psth, pbth, alphac, d):
    """Complete formula with modification due to rotated lattice."""
    coupling_term = sum(((ps/coupled_error_threshold_v2(pb, psth, pbth, alphac))
                        ** (i/2)*(pb/pbth)**((d+1-i)/2) for i in range(1, d+1)))
    return (decoupled_error_model(ps, pb, psth, pbth, alpha1, alpha2, d) +
            alpha3*coupling_term)


def proc_qubits_each(nx, ny, d):
    """Give the total number of physical qubit per proc in small proc setting.

    For the layout with full routing (cnot in 2d time step but artificial overhead).
    """
    return (2*(((3*d+3)*nx//2 + (2*d+2)*(nx % 2) + d)
               * ((3*d+3)*ny//2 + (2*d+2)*(ny % 2) + d) + d) - 1 + 3*d)


def proc_qubits_each_v2(ny, d):
    """Give the total number of physical qubit per proc in small proc setting 
    for compact layout."""
    return (2*((3*d+2)*((d+1)*ny-1)+(2*d+1)*(d+1)+d) - 1 + 2*d)


def proc_qubits_monolythic(ny, d):
    """Give the total number of physical qubit per proc in big proc setting."""
    return (2*((3*d+2)*((d+1)*ny-1)+(2*d+1)*(d+1)) - 1)


def find_max_product(threshold, f, nx_max=10, ny_max=10):
    """Find the values of n1 and n2 (1 <= n1, n2 <= n_max) that maximize n1 * n2.

    Ensure f(n1, n2) <= threshold.

    param:
        threshold: Threshold value for f(n1, n2).
        f: Function that takes n1 and n2 as arguments and returns a value.
        n_max : maximal value for nx, ny
    return: Tuple (n1, n2, max_product) where n1 and n2 are the optimal values
             and max_product is their product.
    """
    max_product = 1
    best_n1, best_n2 = 1, 1

    for n1 in range(1, nx_max+1):
        for n2 in range(1, ny_max+1):
            if f(n1, n2) <= threshold:
                product = n1 * n2
                if product > max_product:
                    max_product = product
                    best_n1, best_n2 = n1, n2

    return best_n1, best_n2, max_product


def find_size_layout_v2(threshold, f, n_max=100):
    """Find the values of n1 and n2 (1 <= n1, n2 <= n_max) that maximize n1 * n2.

    Ensure f(n) <= threshold.

    param:
        threshold: Threshold value for f(n).
        f: Function that takes n1 and n2 as arguments and returns a value.
        n_max : maximal value for nx, ny
    return: Tuple (n1, n2, max_product) where n1 and n2 are the optimal values
             and max_product is their product.
    """
    n = 1
    nb_log_qbit_per_proc = 2*n
    while f(n) <= threshold and n < n_max:
        n += 1
        nb_log_qbit_per_proc = 2*n
    return n, nb_log_qbit_per_proc

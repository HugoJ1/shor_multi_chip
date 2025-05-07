#!/usr/bin/env python3
# coding: utf-8
"""
Outils pour le calcul de la complexité algorithmique.

Créé le Mon Feb  8 21:19:33 2021
@author: Élie Gouzien
"""
import numbers
from collections import namedtuple
from datetime import timedelta

AlgoOpts = namedtuple('AlgoOpts',
                      'prob, algo, s, n, we, wm, c, '
                      ' windowed, mesure_based_deand, mesure_based_unlookup,'
                      'parallel_cnots',
                      defaults=('rsa', 'Ekera', 1, None, None, None, None,
                                True, None, True,
                                False))
AlgoOpts.__doc__ = (
    """
AlgoOpts(prob, algo, s, n, we, wm, c, windowed,
         mesure_based_deand, mesure_based_unlookup, 'parallel_cnots')

Attributs :
    prob : problème : 'rsa' ou 'elliptic_log'
    algo : algorithme utilisé : 'Ekera' ou 'Shor'
    s    : compromis dans l'algo d'Ekera. Doit être 'None' pour Shor.
    n    : nombre de qubits du nombre exponentié (ou plutôt du module)
    we   : taille de la fenêtre de l'exposant
    wm   : taille de la fenêtre pour la multiplication
    c    : nombre de bits de décalage pour la représentation par classe
    windowed : utiliser l'algorithme fenêtré ? (True ou False)
    mesure_based_deand : décalculer le AND avec la mesure
                        (utilisé que quand type == '3dcolor', sinon prend par
                         défaut la meilleure option)
    mesure_based_unlookup : décalcul du lookup basé sur la mesure ?
                            (utilisable que quand type == 'alice&bob(n)',
                             car initialisation |0> compliquée)
    parallel_cnots : faire en un coup toutes les CNOT avec un control et un
                     nombre arbitraire de cibles ? Attention, je ne prends pas
                     pour autant en compte une vrai parallélisation des
                     différentes CNOTs (il peut y avoir des problèmes de
                                        routage).

Parameters:
    prob : addressed problem : 'rsa' or 'elliptic_log'
    algo : used algorithm : 'Ekera' or 'Shor'
    s    : tradeoff in Ekera's algorithm. Must be None if 'Shor' is chosen.
    n    : number of bits of N
    we   : size of the exponentiation window
    wm   : size of the multiplication window
    c    : number of bits added for coset representation
    windowed : use windowed algorithm ? (True or False)
    mesure_based_deand : measurement based AND uncomputation
                        (only for type == '3dcolor', otherwise takes by
                         default the obvious option)
    mesure_based_unlookup : unlookup based on measurement?
                            (only usable for type == 'alice&bob(n)' because |0>
                             preparation is long)
    parallel_cnots : CNOT with multiple targets cost as much as a single CNOT?
                     Be careful, it is not a full parallelisation of the CNOTs
                     (routing might cause problems).
""")

LowLevelOpts = namedtuple('LowLevelOpts', 'debitage, d1, d2, d, n, tc, tr, pp, pbell',
                          defaults=(2, None, None, None, None, 1e-6, 1e-6,
                                    1e-3, 0e-2))
LowLevelOpts.__doc__ = """LowLevelOpts(debitage, d1, d2, d, n, tc, tr, pp, pbell)

Attributs :
    debitage : débitage du tétraèdre pour la correction '3dcolor'
    d1   : distance du premier étage de distillation/application
    d2   : distance du second étage de distillation/application
    d    : distance du code principal
    n    : nombre moyen de photons pour le chat d'Alice et Bob
    tc   : temps de cycle
    tr   : temps de réaction
    pp   : probabilité d'erreur par porte physique (identité comprise)
    pbell: probabilité d'erreur sur la préparation des paires de Bell'
Parameters:
    debitage : cut of tetrahedron for '3dcolor' error correction code
    d1 : distance of first step of distillation/applying
    d2 : distance of second step of distillation/applying
    d  : main code distance
    n  : average number of photons, only for Alice&Bob's cat
    tc : cycle time
    tr : reaction time
    pp : error probability on physical gates (inc. identity)
    pbell: error probability on Bell state preparation
"""

Params = namedtuple('Params', 'type, algo, low_level')
Params.__doc__ = """Params(type, algo, low_level)

Attributs:
    type      : type de correction d'erreur : 'surface', '3dcolor',
                'alice&bob', 'alice&bob2', 'alice&bob3,' '3dsurface' ou None
    algo      : options de l'algoritme, de type AlgoOpts
    low_level : option bas niveau, de type LowLevelOpts

Parameters:
    type      : type of error correction : 'surface', '3dcolor',
                'alice&bob', 'alice&bob2', 'alice&bob3, '3dsurface' or None
    algo      : algorithm options, type AlgoOpts
    low_level : low level options, type LowLevelOpts
"""


def addable_namedtuple(*args, **kwargs):
    """Comme namedtyple, mais le type créé se comporte comme un vecteur."""
    class _TempName(namedtuple(*args, **kwargs)):
        def __add__(self, other):
            """Addition de deux vecteurs."""
            if not isinstance(other, __class__):
                return NotImplemented
            return __class__._make(i+j for i, j in zip(self, other))

        def __mul__(self, other):
            """Multiplication par un scalaire."""
            if not isinstance(other, numbers.Real):
                return NotImplemented
            return __class__._make(i * other for i in self)

        def __rmul__(self, other):
            return self * other

        def __truediv__(self, other):
            """Division par un scalaire."""
            return self * (1/other)

    _TempName.__name__ = args[0]
    _TempName.__qualname__ = _TempName.__qualname__.replace(
        'addable_namedtuple.<locals>._TempName', args[0])
    return _TempName


Profond = addable_namedtuple('Profond', ['gate1',  # Portes à 1 qubit
                                         'cnot',   # CNOT
                                         'ands',   # Calcul/décalcul ET
                                         'mes',    # Nombre de mesures
                                         'ccz'])   # Nombre de portes CCZ

Ressource = addable_namedtuple('Ressource', ['gate1',   # Portes à 1 qubit
                                             'cnot',    # CNOT
                                             'ands',    # Calcul/décalcul ET
                                             'mes',     # Nombre de mesures
                                             'aux',     # Qubits auxiliaires
                                             'ccz',     # Nombre de portes CCZ
                                             'prof'])   # Profondeur parallèle


class PhysicalCost(namedtuple('PhysicalCost', ('p', 't'))):
    """Coûts physique : probabilité d'erreur, et temps d'exécution.

    Attributs
    ---------
        p : probabilité d'erreur.
        t : temps d'exécution.

    Methods
    -------
        Has same interface as namedtuple, except for listed operators.

    Operators
    ---------
        a + b : cost of serial execution of a and b.
        k * a : cost of serial execution of a k times (k can be float).
        a | b : cost of parallel execution of a and b.
        k | b : cost of k parallel executions of b.
    """

    def __add__(self, other):
        """Addition de coûts physiques."""
        if not isinstance(other, __class__):
            return NotImplemented
        return __class__(1 - (1 - self.p)*(1 - other.p), self.t + other.t)

    def __mul__(self, other):
        """Multiplication par un scalaire, non entier tolléré pour moyennes."""
        if not isinstance(other, numbers.Real):
            return NotImplemented
        if self.p >= 1:
            return __class__(self.p, self.t * other)
        return __class__(1 - (1 - self.p)**other, self.t * other)

    def __rmul__(self, other):
        """Multiplication à droite."""
        return self * other

    def __sub__(self, other):
        """Soustraction : permet d'annuler une addition précédente."""
        return self + (-1 * other)

    def __or__(self, other):
        """Coût de l'éxécution parallèlle de self et other.

        Si other est un réel, on calcule l'exécution parallèle de other fois
        l'opération self.

        Remarque : pour les circuits logiques, il est impossible de calculer
        le coût du final des opérations parallèles.
        """
        if isinstance(other, __class__):
            return __class__(1 - (1 - self.p)*(1-other.p),
                             max(self.t, other.t))
        if isinstance(other, numbers.Real):
            return __class__(1 - (1 - self.p)**other, self.t)
        return NotImplemented

    def __ror__(self, other):
        """Coût de l'éxécution parallèle (à droite)."""
        return self | other

    @property
    def exp_t(self):
        """Temps moyen d'exécution."""
        if self.p is None:
            return self.t
        if self.p >= 1:
            return float('inf')
        return self.t / (1 - self.p)

    @property
    def exp_t_str(self):
        """Temps moyen d'exécution, formaté."""
        try:
            return timedelta(seconds=self.exp_t)
        except OverflowError:
            if self.exp_t == float('inf'):
                return "∞"
            return str(round(self.exp_t/(3600*24*365.25))) + " years"

    def __str__(self):
        """Convertie pour l'affichage."""
        # pylint: disable=C0103
        try:
            t = timedelta(seconds=self.t)
        except OverflowError:
            if self.t == float('inf'):
                t = "∞"
            else:
                t = str(round(self.t/(3600*24*365.25))) + " years"
        return f"PhysicalCost(p={self.p}, t={t}, exp_t={self.exp_t_str})"

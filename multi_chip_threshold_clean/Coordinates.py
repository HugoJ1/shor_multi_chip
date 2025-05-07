from collections import namedtuple

SPLIT_DIST = 0.2  # distance between the two qubits of a Bell pair.


# %% Tools and helpers
class Coord(namedtuple('Coord', ['i', 'j'])):
    """Point en 2D, en convention matrices, qui s'ajoutent."""

    def __add__(self, other):
        """Addition de deux points."""
        if not len(other) == 2:
            raise ValueError("Seul un point s'ajoute à un point.")
        # return Coord(*(self[k] + other[k] for k in range(len(self))))
        return Coord(*(a + b for a, b in zip(self, other)))

    def __radd__(self, other):
        """Right addition."""
        return self + other

    def __mul__(self, other):
        """Scalar multiplication."""
        return Coord(*(other*a for a in self))

    def __rmul__(self, other):
        """Right scalar multiplication."""
        return self*other

    def __neg__(self):
        """Unary minus."""
        return -1*self

    def __sub__(self, other):
        """Left subtraction."""
        return self + (-1)*other

    def __rsub__(self, other):
        """Right subtraction."""
        return -(other - self)

    def stim_coord(self, t=None):
        """Print the stim corrdinate, with eventually time."""
        if t is None:
            return (self.j, self.i)
        return (self.j, self.i, t)

    def bell_partner(self):
        """Return the other qubit of the same bell pair."""
        if self.i == int(self.i) and self.j == int(self.j):
            print("This is not a splitted Bell pair !")
        if self.j == int(self.j) + SPLIT_DIST:
            center = self - Coord(0, SPLIT_DIST)
            return (Coord(int(center.i), int(center.j)) - Coord(0, SPLIT_DIST))
        if self.j == int(self.j) + 1 - SPLIT_DIST:
            center = self + Coord(0, SPLIT_DIST)
            return (Coord(int(center.i), int(center.j)) + Coord(0, SPLIT_DIST))

import numpy as np
from rechunker.compat import prod


def test_prod():
    assert prod(()) == 1
    assert prod((2,)) == 2
    assert prod((2, 3)) == 6
    n = np.iinfo(np.int64).max
    assert prod((n, 2)) == n * 2

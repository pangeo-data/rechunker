from functools import reduce
import operator
from typing import Sequence


def prod(iterable: Sequence[int]) -> int:
    """Implementation of `math.prod()` all Python versions."""
    try:
        from math import prod as mathprod  # Python 3.8

        return mathprod(iterable)
    except ImportError:
        return reduce(operator.mul, iterable, 1)

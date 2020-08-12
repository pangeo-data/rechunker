from functools import reduce
import operator
from typing import Sequence


def prod(iterable: Sequence[int]) -> int:
    """Implementation of `math.prod()` for Python versions less than 3.8."""
    return reduce(operator.mul, iterable, 1)

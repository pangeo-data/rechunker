import math
import operator
from functools import reduce
from typing import Iterable, TypeVar

T = TypeVar("T", int, float)


try:
    from math import prod  # type: ignore  # Python 3.8
except ImportError:

    def prod(iterable: Iterable[T]) -> T:  # type: ignore
        """Implementation of `math.prod()` for all Python versions."""
        return reduce(operator.mul, iterable, 1)


try:
    from math import lcm  # type: ignore  # Python 3.9
except ImportError:

    def lcm(a: int, b: int) -> int:  # type: ignore
        """Implementation of `math.lcm()` for all Python versions."""
        return abs(a * b) // math.gcd(a, b)

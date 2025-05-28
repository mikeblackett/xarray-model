from collections.abc import Iterable
from collections import Counter
from typing import Any


def check_valid_size(values: Iterable[Any], min_size: int = 0) -> None:
    all_values = list(values)
    if not all_values:
        raise ValueError(
            f'Expected at least {min_size} value(s); got {len(all_values)!r}'
        )

def check_non_negative(value: int | float) -> None:
    if value < 0:
        raise ValueError(f'Expected non-negative value; got {value!r}')


def check_uniqueness(values: Iterable[Any]) -> None:
    all_values = list(values)
    unique_values = set(all_values)
    if len(unique_values) != len(all_values):
        counter = Counter(all_values)
        non_unique = [k for k, v in counter.items() if v > 1]
        raise ValueError(
            f'Expected values to be unique; found {non_unique!r} repeated values.'
        )

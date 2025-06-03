from typing import Any, Sequence
import hypothesis as hp
from hypothesis import strategies as st

import numpy as np

from xarray_model.components import Name


DTYPE_NAMES = [
    'int',
    'int8',
    'int16',
    'int32',
    'int64',
    'float',
    'float16',
    'float32',
    'float64',
    'bool',
    'str',
    'datetime64',
    'timedelta64',
]

DTYPES = [np.dtype(name) for name in DTYPE_NAMES]

REGEX_PATTERNS = [
    r'[A-Za-z]{3,10}',  # Letters only, 3-10 chars
    r'\d{2,4}',  # 2-4 digits
    r'[A-Z][a-z]+\d{2}',  # Capitalized word with 2 digits
    r'[a-z]+[-_][a-z]+',  # Two lowercase words with dash/underscore
    r'[A-Z0-9]{8}',  # 8 chars of uppercase letters and numbers
    r'^TEST_[A-Z]+$',  # Uppercase word with TEST_ prefix
    r'^[a-z]+_\d{3}$',  # Lowercase word with 3 digits suffix
    r'^\d{2}-[A-Z]{3}$',  # 2 digits followed by 3 uppercase letters
    r'^[A-Z][a-z]+$',  # Single capitalized word
    r'^user_\d+$',  # Username with number
    r'^\d{4}-[A-Z]{2}$',  # 4 digits followed by 2 uppercase letters
    r'^[a-z]{2}\d{2}$',  # 2 lowercase letters followed by 2 digits
    r'^v\d+\.\d+$',  # Version number format
    r'^[A-Z]{2}_\d{4}$',  # 2 uppercase letters followed by 4 digits
    r'^[a-z]+@test$',  # Lowercase word with @test suffix
]


@st.composite
def patterns(draw: st.DrawFn) -> str:
    return draw(st.sampled_from(REGEX_PATTERNS))


@st.composite
def positive_integers(draw, max_value: int | None = None) -> int:
    return draw(st.integers(min_value=1, max_value=max_value))


@st.composite
def non_negative_integers(draw, max_value: int | None = None) -> int:
    return draw(st.integers(min_value=0, max_value=max_value))


@st.composite
def multiples_of(
    draw, base: int, min_value: int = 0, max_value: int | None = None
) -> int:
    multiplier = draw(st.integers(min_value=1))
    result = base * multiplier
    hp.assume(result >= min_value)
    if max_value is not None:
        hp.assume(result <= max_value)
    return result


@st.composite
def names(
    draw: st.DrawFn,
    regex: str | None = None,
    min_length: int = 0,
    max_length: int | None = None,
) -> str:
    if regex is not None:
        return draw(st.from_regex(regex=regex, fullmatch=True))
    return draw(st.text(min_size=min_length, max_size=max_length))


@st.composite
def shapes(
    draw,
    max_value: int | None = None,
    min_value: int = 0,
    min_size: int = 1,
    max_size: int | None = None,
):
    return draw(
        st.lists(
            elements=st.integers(min_value=min_value, max_value=max_value),
            min_size=min_size,
            max_size=max_size,
        )
    )


@st.composite
def dims(
    draw, min_size: int = 0, max_size: int | None = None
) -> Sequence[str | Name]:
    return draw(
        st.lists(
            elements=st.text(),
            min_size=min_size,
            max_size=max_size,
        )
    )


@st.composite
def dtypes(draw) -> np.dtype:
    return draw(st.sampled_from(DTYPES))


@st.composite
def chunks(
    draw: st.DrawFn,
    min_value: int = 1,
    max_value: int | None = None,
    min_size: int = 1,
    max_size: int | None = None,
):
    if max_value is None:
        max_value = draw(st.integers(min_value=min_value + 1))
    if max_size is None:
        max_size = draw(st.integers(min_value=min_size + 1, max_value=4))
    block_size = draw(st.integers(min_value=min_value, max_value=max_value))
    multiplier = draw(st.integers(min_value=min_size, max_value=max_size))
    size = draw(
        st.integers(
            min_value=block_size * min_size, max_value=block_size * multiplier
        )
    )
    n = size // block_size
    last_block = size % block_size
    blocks = [block_size] * n
    if last_block:
        blocks.append(last_block)
    return blocks


@st.composite
def attrs(
    draw,
    min_size: int = 0,
    max_size: int | None = None,
) -> dict[str, Any]:
    return draw(
        st.dictionaries(
            keys=st.text(min_size=1),
            values=st.one_of(
                st.none(), st.booleans(), st.integers(), st.floats(), st.text()
            ),
            min_size=min_size,
            max_size=max_size,
        )
    )

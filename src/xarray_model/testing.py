import hypothesis as hp
import numpy as np
import xarray.testing.strategies as xrst
from hypothesis import strategies as st


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
def supported_dtype_likes(
    draw: st.DrawFn,
    dtype: np.dtype | None = None,
) -> np.dtype | str | type | None:
    """Generate only those numpy DTypeLike that xarray can handle.

    See @https://numpy.org/doc/stable/reference/arrays.dtypes.html to know what
    can be converted to a data-type object.

    If a dtype is provided, then only those values that are compatible with
    the dtype will be returned.
    """
    if dtype is not None:
        return draw(
            st.sampled_from(
                [
                    # string
                    dtype.name,
                    # array-protocol typestring
                    dtype.str,
                    # One-character strings
                    dtype.char,
                    # dtype
                    dtype,
                ],
            )
        )
    _dtype_strategy = xrst.supported_dtypes()
    return draw(
        st.sampled_from(
            [
                draw(st.none()),
                # string
                draw(_dtype_strategy).name,
                # array-protocol typestring
                draw(_dtype_strategy).str,
                # One-character strings
                draw(_dtype_strategy).char,
                # dtype
                draw(_dtype_strategy),
                # Built-in types
                draw(
                    st.sampled_from([int, float, bool, str, complex]),
                ),
            ],
        )
    )


@st.composite
def patterns(draw: st.DrawFn) -> str:
    return draw(st.sampled_from(REGEX_PATTERNS))


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
def dimension_shapes(
    draw,
    min_dims: int = 1,
    max_dims: int = 5,
    min_side: int = 0,
    max_side: int | None = None,
) -> tuple[int, ...]:
    return tuple(
        draw(
            xrst.dimension_sizes(
                min_dims=min_dims,
                max_dims=max_dims,
                min_side=min_side,
                max_side=max_side,
            )
        ).values()
    )


@st.composite
def uniform_chunks(
    draw: st.DrawFn,
    min_block_size: int = 1,
    max_block_size: int | None = None,
    min_dims: int = 1,
    max_dims: int = 10,
):
    if max_block_size is None:
        max_block_size = draw(st.integers(min_value=min_block_size + 1))
    block_size = draw(
        st.integers(min_value=min_block_size, max_value=max_block_size)
    )
    multiplier = draw(st.integers(min_value=min_dims, max_value=max_dims))
    size = draw(
        st.integers(
            min_value=block_size * min_dims, max_value=block_size * multiplier
        )
    )
    n = size // block_size
    last_block = size % block_size
    blocks = [block_size] * n
    if last_block:
        blocks.append(last_block)
    return blocks

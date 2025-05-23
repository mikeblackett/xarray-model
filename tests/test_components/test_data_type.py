import hypothesis as hp
import pytest as pt
import numpy as np
from hypothesis import strategies as st
from jsonschema import ValidationError
from numpy.typing import DTypeLike

from xarray_model import Datatype

DTYPE_LIKES = [
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
DTYPES = [np.dtype(name) for name in DTYPE_LIKES]

test: DTypeLike = str


@st.composite
def dtype_likes(draw):
    return draw(st.sampled_from(DTYPE_LIKES))


class TestDataType:
    @hp.given(dtype_like=dtype_likes())
    def test_global_match(self, dtype_like: DTypeLike):
        Datatype().validate(np.dtype(dtype_like))

    @hp.given(dtype_like=dtype_likes())
    def test_match(self, dtype_like: DTypeLike):
        Datatype(dtype_like).validate(np.dtype(dtype_like))

    def test_mismatch(self):
        with pt.raises(ValidationError):
            Datatype('float').validate(np.dtype('int'))

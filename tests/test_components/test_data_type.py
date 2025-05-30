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
    dtype = draw(st.sampled_from(DTYPE_LIKES))
    if draw(st.booleans()):
        return np.dtype(dtype)
    return dtype


class TestDataType:
    @hp.given(dtype=dtype_likes())
    def test_validates_with_defaults(self, dtype: DTypeLike):
        """Should pass with default values."""
        Datatype().validate(dtype)

    @hp.given(dtype=dtype_likes())
    def test_validates(self, dtype: DTypeLike):
        """Should pass if the instance matches a dtype."""
        Datatype(dtype).validate(dtype)

    @hp.given(data=st.data())
    def test_invalidates(self, data: st.DataObject):
        """Should fail if the instance does not match a dtype"""
        expected = data.draw(dtype_likes())
        actual = data.draw(dtype_likes())
        hp.assume(expected != actual)
        with pt.raises(ValidationError):
            Datatype(expected).validate(actual)

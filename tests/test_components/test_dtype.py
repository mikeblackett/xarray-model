import hypothesis as hp
import numpy as np
import pytest as pt
import xarray.testing.strategies as xrst
from hypothesis import strategies as st
from jsonschema import ValidationError
from numpy.typing import DTypeLike

from xarray_model import DType
from xarray_model.testing import supported_dtype_likes


class TestDataType:
    @hp.given(dtype_like=supported_dtype_likes())
    def test_arguments(self, dtype_like: DTypeLike) -> None:
        """Should always produce a valid JSON Schema"""
        assert DType(dtype_like).schema

    @hp.given(dtype=xrst.supported_dtypes())
    def test_validates_with_defaults(self, dtype: np.dtype) -> None:
        """Should pass with default values."""
        DType().validate(dtype)

    @hp.given(dtype=xrst.supported_dtypes(), data=st.data())
    def test_validates_with_matching_dtype_like(
        self, dtype: np.dtype, data: st.DataObject
    ) -> None:
        """Should pass if the instance matches a dtype-like."""
        dtype_like = data.draw(supported_dtype_likes(dtype))
        DType(dtype_like).validate(dtype)

    @hp.given(data=st.data())
    def test_invalidates_with_dtype(self, data: st.DataObject) -> None:
        """Should fail if the instance does not match a dtype."""
        expected_dtype = data.draw(xrst.supported_dtypes())
        instance = data.draw(xrst.supported_dtypes())
        hp.assume(expected_dtype != instance)
        dtype_like = data.draw(supported_dtype_likes(expected_dtype))
        with pt.raises(ValidationError):
            DType(dtype_like).validate(instance)

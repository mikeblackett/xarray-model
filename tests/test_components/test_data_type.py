import hypothesis as hp
import pytest as pt
import numpy as np
from hypothesis import strategies as st
from jsonschema import ValidationError

from xarray_model import DType

from xarray_model.testing import dtypes


class TestDataType:
    @hp.given(data=st.data())
    def test_arguments(self, data: st.DataObject) -> None:
        """Should always produce a valid JSON Schema"""
        dtype = data.draw(dtypes())
        expected_dtype = data.draw(st.sampled_from([None, dtype.name, dtype]))
        assert DType(expected_dtype).schema

    @hp.given(data=st.data())
    def test_validates_with_defaults(self, data: st.DataObject) -> None:
        """Should pass with default values."""
        dtype = data.draw(dtypes())
        instance = data.draw(st.sampled_from([dtype, dtype.name]))
        DType().validate(instance)

    @hp.given(data=st.data())
    def test_validates_with_dtype(self, data: st.DataObject) -> None:
        """Should pass if the instance matches a dtype."""
        dtype = data.draw(dtypes())
        instance = data.draw(st.sampled_from([dtype, dtype.name]))
        DType(dtype).validate(instance)

    @hp.given(data=st.data())
    def test_invalidates_with_dtype(self, data: st.DataObject) -> None:
        """Should fail if the instance does not match a dtype."""
        expected_dtype = data.draw(dtypes())
        instance_dtype = data.draw(dtypes())
        hp.assume(expected_dtype != instance_dtype)
        instance = data.draw(
            st.sampled_from([instance_dtype, instance_dtype.name])
        )
        with pt.raises(ValidationError):
            DType(expected_dtype).validate(instance)

    @hp.given(data=st.data())
    def test_validates_with_dtype_name(self, data: st.DataObject) -> None:
        """Should pass if the instance matches a dtype name"""
        dtype = data.draw(dtypes())
        instance = data.draw(st.sampled_from([dtype, dtype.name]))
        DType(dtype.name).validate(instance)

    @hp.given(data=st.data())
    def test_invalidates_with_dtype_name(self, data: st.DataObject) -> None:
        """Should fail if the instance does not match a dtype name"""
        expected_dtype = data.draw(dtypes())
        instance_dtype = data.draw(dtypes())
        hp.assume(expected_dtype != instance_dtype)
        instance = data.draw(
            st.sampled_from([instance_dtype, instance_dtype.name])
        )
        with pt.raises(ValidationError):
            DType(expected_dtype.name).validate(instance)

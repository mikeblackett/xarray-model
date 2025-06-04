from collections.abc import Sequence

import hypothesis as hp
import pytest as pt
from hypothesis import strategies as st
from jsonschema import ValidationError
from xarray_model import Dims

import xarray.testing.strategies as xrst


class TestDims:
    @hp.given(data=st.data())
    def test_arguments(self, data: st.DataObject) -> None:
        """Should always produce a valid JSON Schema"""
        dims = data.draw(st.one_of(st.none(), xrst.dimension_names()))
        contains = data.draw(st.one_of(st.none(), st.text()))
        min_size = data.draw(st.integers(min_value=0))
        max_size = data.draw(st.integers(min_value=min_size))
        assert Dims(
            dims, contains=contains, max_size=max_size, min_size=min_size
        ).schema

    @hp.given(instance=xrst.dimension_names())
    def test_validates_with_defaults(self, instance: Sequence[str]):
        """Should pass with default values."""
        Dims().validate(tuple(instance))

    @hp.given(instance=xrst.dimension_names(), flag=st.booleans())
    def test_validates_with_sequences(
        self, instance: Sequence[str], flag: bool
    ):
        """Should pass if the instance matches a sequence of names."""
        Dims(instance).validate(tuple(instance))

    @hp.given(
        expected=xrst.dimension_names(min_dims=1),
        instance=xrst.dimension_names(min_dims=1),
    )
    def test_invalidates_with_sequences(
        self, expected: Sequence[str], instance: Sequence[str]
    ):
        """Should fail if the instance does not match a sequence of names."""
        hp.assume(expected != instance)
        with pt.raises(ValidationError):
            Dims(expected).validate(tuple(instance))

    @hp.given(instance=xrst.dimension_names(min_dims=1), data=st.data())
    def test_validates_with_contains(
        self, instance: Sequence[str], data: st.DataObject
    ):
        """Should pass if the instance contains a name."""
        contains = data.draw(st.sampled_from(instance))
        Dims(contains=contains).validate(tuple(instance))

    @hp.given(instance=xrst.dimension_names(min_dims=1), data=st.data())
    def test_invalidates_with_contains(
        self, instance: Sequence[str], data: st.DataObject
    ):
        """Should fail if the instance does not contain a name."""
        expected = data.draw(st.text(min_size=1))
        hp.assume(expected not in instance)
        with pt.raises(ValidationError):
            Dims(contains=expected).validate(tuple(instance))

    @hp.given(data=st.data())
    def test_validates_with_size(self, data: st.DataObject):
        """Should pass if the instance is within the size constraints."""
        min_size = data.draw(st.integers(min_value=0, max_value=10))
        max_size = data.draw(st.integers(min_value=min_size, max_value=100))
        hp.assume(min_size <= max_size)
        actual = data.draw(
            xrst.dimension_names(min_dims=min_size, max_dims=max_size)
        )
        Dims(min_size=min_size, max_size=max_size).validate(tuple(actual))

    @hp.given(data=st.data())
    def test_invalidates_with_size(self, data: st.DataObject):
        """Should fail if the instance is outside the size constraints."""
        actual = data.draw(xrst.dimension_names(min_dims=0, max_dims=3))
        with pt.raises(ValidationError):
            Dims(min_size=4).validate(tuple(actual))

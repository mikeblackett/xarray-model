from collections.abc import Sequence

import hypothesis as hp
import pytest as pt
from hypothesis import strategies as st
from jsonschema import ValidationError

from xarray_model import Shape
from xarray_model.testing import shapes


class TestShape:
    @hp.given(
        shape=shapes(),
        min_items=st.integers(min_value=0),
        max_items=st.integers(min_value=0),
    )
    def test_arguments(
        self,
        shape: Sequence[int],
        min_items: int,
        max_items: int,
    ) -> None:
        """Should always produce a valid JSON Schema"""
        assert Shape(shape, min_items=min_items, max_items=max_items).schema

    @hp.given(shape=shapes())
    def test_validates_with_defaults(self, shape: Sequence[int]):
        """Should pass with default values."""
        Shape().validate(tuple(shape))

    @hp.given(shape=shapes())
    def test_validates_with_sequence_of_sizes(self, shape: Sequence[int]):
        """Should pass when the dimensions match a sequence of sizes."""
        instance = tuple(shape)
        Shape(shape).validate(instance)

    @hp.given(shape=shapes(), instance=shapes())
    def test_invalidates_with_sequence_of_sizes(
        self, shape: Sequence[int], instance: Sequence[int]
    ):
        """Should fail when all dimensions do not match sequence of sizes."""
        hp.assume(shape != instance)
        with pt.raises(ValidationError):
            Shape(shape).validate(tuple(instance))

    @hp.given(data=st.data())
    def test_validates_with_item_constraints(self, data: st.DataObject):
        """Should pass if the instance matches the item constraints."""
        min_items = data.draw(st.integers(min_value=1, max_value=100))
        max_items = data.draw(st.integers(min_value=min_items))
        instance = data.draw(shapes(max_size=max_items, min_size=min_items))
        Shape(max_items=max_items, min_items=min_items).validate(instance)

    @hp.given(data=st.data())
    def test_invalidates_with_item_constraints(self, data: st.DataObject):
        """Should fail if the instance does not match the item constraints."""
        min_items = data.draw(st.integers(min_value=2, max_value=100))
        max_items = data.draw(st.integers(min_value=min_items, max_value=100))
        instance = data.draw(shapes(min_size=max_items + 1))
        with pt.raises(ValidationError):
            Shape(max_items=max_items).validate(instance)
        instance = data.draw(shapes(max_size=min_items - 1))
        with pt.raises(ValidationError):
            Shape(min_items=min_items).validate(instance)

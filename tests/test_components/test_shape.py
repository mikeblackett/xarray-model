from collections.abc import Sequence

import hypothesis as hp
from jsonschema import ValidationError
import pytest as pt
from hypothesis import strategies as st

from xarray_model import Shape
from xarray_model.testing import dimension_shapes


class TestShape:
    @hp.given(
        shape=dimension_shapes(min_dims=1),
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

    @hp.given(shape=dimension_shapes())
    def test_validates_with_defaults(self, shape: Sequence[int]):
        """Should pass with default values."""
        Shape().validate(tuple(shape))

    @hp.given(shape=dimension_shapes())
    def test_validates_with_sequence_of_sizes(self, shape: Sequence[int]):
        """Should pass when the dimensions match a sequence of sizes."""
        instance = tuple(shape)
        Shape(shape).validate(instance)

    @hp.given(shape=dimension_shapes(), instance=dimension_shapes())
    def test_invalidates_with_sequence_of_sizes(
        self, shape: Sequence[int], instance: tuple[int]
    ):
        """Should fail when all dimensions do not match sequence of sizes."""
        hp.assume(shape != instance)
        with pt.raises(ValidationError):
            Shape(shape).validate(instance)

    @hp.given(data=st.data())
    def test_validates_with_item_constraints(self, data: st.DataObject):
        """Should pass if the instance matches the item constraints."""
        min_items = data.draw(st.integers(min_value=1, max_value=100))
        max_items = data.draw(st.integers(min_value=min_items))
        instance = data.draw(
            dimension_shapes(max_dims=max_items, min_dims=min_items)
        )
        Shape(max_items=max_items, min_items=min_items).validate(instance)

    @hp.given(data=st.data())
    def test_invalidates_with_item_constraints(self, data: st.DataObject):
        """Should fail if the instance does not match the item constraints."""
        expected = data.draw(st.integers(min_value=2, max_value=4))
        instance = data.draw(dimension_shapes(min_dims=expected + 1))
        with pt.raises(ValidationError):
            Shape(max_items=expected).validate(instance)
            Shape(min_items=expected).validate(instance)

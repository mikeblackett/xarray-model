import hypothesis as hp
import pytest as pt
from hypothesis import strategies as st
from jsonschema import ValidationError

from xarray_model import Size
from xarray_model.testing import (
    multiples_of,
)


class TestSize:
    @hp.given(
        size=st.integers(min_value=0),
        multiple_of=st.integers(min_value=0),
        maximum=st.integers(min_value=0),
        minimum=st.integers(min_value=0),
    )
    def test_arguments(
        self,
        size: int,
        multiple_of: int,
        maximum: int,
        minimum: int,
    ) -> None:
        """Should always produce a valid JSON Schema"""
        assert Size(
            size, multiple_of=multiple_of, maximum=maximum, minimum=minimum
        ).schema

    @hp.given(size=st.integers(min_value=0))
    def test_validates_with_defaults(self, size: int):
        """Should pass with default values."""
        Size().validate(size)

    @hp.given(value=st.integers(min_value=0))
    def test_validates_with_integers(self, value: int):
        Size(value).validate(value)

    @hp.given(data=st.data())
    def test_invalidates_with_integers(self, data: st.DataObject):
        expected = data.draw(st.integers(min_value=0))
        instance = data.draw(st.integers(min_value=expected + 1))
        with pt.raises(ValidationError):
            Size(expected).validate(instance)

    @hp.given(base=st.integers(min_value=1, max_value=10), data=st.data())
    def test_validates_with_multiple_of(self, base: int, data: st.DataObject):
        instance = data.draw(multiples_of(base=base))
        Size(multiple_of=base).validate(instance)

    @hp.given(base=st.integers(min_value=2), data=st.data())
    def test_invalidates_with_multiple_of(
        self, base: int, data: st.DataObject
    ):
        multiple = base * data.draw(st.integers(min_value=1))
        instance = multiple - (multiple % base) + 1
        with pt.raises(ValidationError):
            Size(multiple_of=base).validate(instance)

    @hp.given(maximum=st.integers(min_value=1), data=st.data())
    def test_validates_with_maximum(self, maximum: int, data: st.DataObject):
        instance = data.draw(st.integers(max_value=maximum))
        Size(maximum=maximum).validate(instance)

    @hp.given(maximum=st.integers(min_value=1), data=st.data())
    def test_invalidates_with_maximum(self, maximum: int, data: st.DataObject):
        instance = data.draw(st.integers(min_value=maximum + 1))
        with pt.raises(ValidationError):
            Size(maximum=maximum).validate(instance)

    @hp.given(minimum=st.integers(min_value=1), data=st.data())
    def test_validates_with_minimum(self, minimum: int, data: st.DataObject):
        instance = data.draw(st.integers(min_value=minimum))
        Size(minimum=minimum).validate(instance)

    @hp.given(minimum=st.integers(min_value=1), data=st.data())
    def test_invalidates_with_minimum(self, minimum: int, data: st.DataObject):
        instance = data.draw(st.integers(max_value=minimum - 1))
        with pt.raises(ValidationError):
            Size(minimum=minimum).validate(instance)

from collections.abc import Sequence

import hypothesis as hp
import pytest as pt
from hypothesis import strategies as st
from jsonschema import ValidationError

from xarray_model import Shape


@st.composite
def dims(draw, min_size=1, max_size=10):
    return draw(
        st.iterables(
            elements=st.text(),
            min_size=min_size,
            max_size=max_size,
        )
    )


class TestDims:
    @hp.given(dims=dims())
    def test_global_match(self, shape: Sequence[int]):
        Shape().validate(tuple(shape))

    @hp.given(dims=dims())
    def test_sequence_match(self, shape: Sequence[int]):
        Shape(shape).validate(tuple(shape))

    @hp.given(dims=dims(), data=st.data())
    def test_sequence_match_with_wildcard(
        self, shape: Sequence[int], data: st.DataObject
    ):
        expected = list(shape)
        idx = data.draw(st.integers(min_value=0, max_value=len(expected) - 1))
        expected[idx] = -1
        Shape(shape).validate(tuple(shape))

    @hp.given(data=st.data())
    def test_size_match(self, data: st.DataObject):
        min_size = data.draw(st.integers(min_value=1, max_value=2))
        max_size = data.draw(st.integers(min_value=min_size, max_value=4))
        hp.assume(min_size <= max_size)
        shape = data.draw(dims(min_size=min_size, max_size=max_size))
        Shape(min_size=min_size, max_size=max_size).validate(tuple(shape))

    @pt.mark.parametrize(
        'expected, actual',
        [
            ([1, 2, 3], (1, 2)),
            ([1, 2, 3], (1, 2, 4)),
        ],
    )
    def test_sequence_mismatch(self, expected: Sequence[int], actual: tuple):
        with pt.raises(ValidationError):
            Shape(expected).validate(actual)

    def test_size_mismatch(self):
        with pt.raises(ValidationError):
            Shape(min_size=1, max_size=2).validate((1, 2, 3))

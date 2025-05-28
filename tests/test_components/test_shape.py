from collections.abc import Sequence

import hypothesis as hp
import pytest as pt
from hypothesis import strategies as st
from jsonschema import ValidationError

from xarray_model import Shape


@st.composite
def shapes(
    draw,
    min_dims: int = 1,
    max_dims: int | None = None,
):
    if max_dims is None:
        max_dims = 3
    return draw(
        st.lists(
            elements=st.integers(min_value=1),
            min_size=min_dims,
            max_size=max_dims,
        )
    )


class TestShape:
    @hp.given(shape=shapes())
    def test_global_match(self, shape: Sequence[int]):
        Shape().validate(tuple(shape))

    @hp.given(shape=shapes())
    def test_sequence_match(self, shape: Sequence[int]):
        Shape(shape).validate(tuple(shape))

    @hp.given(shape=shapes(), data=st.data())
    def test_sequence_match_with_wildcard(
        self, shape: list[int], data: st.DataObject
    ):
        expected = shape
        if data.draw(st.booleans()):
            idx = data.draw(
                st.integers(min_value=0, max_value=len(expected) - 1)
            )
            expected[idx] = -1
        Shape(expected).validate(tuple(shape))

    @hp.given(data=st.data())
    def test_size_match(self, data: st.DataObject):
        min_size = data.draw(st.integers(min_value=1, max_value=2))
        max_size = data.draw(st.integers(min_value=min_size, max_value=4))
        hp.assume(min_size <= max_size)
        shape = data.draw(shapes(min_dims=min_size, max_dims=max_size))
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

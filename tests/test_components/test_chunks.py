from collections.abc import Sequence

import pytest as pt
import hypothesis as hp
from hypothesis import strategies as st
from jsonschema import ValidationError

from xarray_model import Chunks
from xarray_model.components import _Chunk
from xarray_model.testing import uniform_chunks


class TestChunk:
    @hp.given(shape=st.one_of(st.integers(), st.lists(st.integers())))
    def test_arguments(self, shape: int | Sequence[int]):
        """Should always produce a valid JSON Schema"""
        schema = _Chunk(shape).schema

    def test_validation_is_not_implemented(self):
        """_Chunk should be composed with the Chunks model."""
        with pt.raises(NotImplementedError):
            _Chunk(False).validate(_=None)


class TestChunks:
    @hp.given(
        expected=st.one_of(
            st.booleans(),
            st.integers(min_value=0),
            st.lists(st.integers(min_value=0)),
            st.lists(st.lists(st.integers(min_value=0))),
        )
    )
    def test_arguments(
        self, expected: bool | int | Sequence[int | Sequence[int]]
    ):
        """Should always produce a valid JSON Schema"""
        assert Chunks(expected).schema

    @hp.given(data=st.data())
    def test_validates_with_boolean_match(self, data: st.DataObject):
        """Should pass when the chunked state matches a boolean."""
        expected = data.draw(st.booleans())
        actual = (
            data.draw(st.lists(uniform_chunks(), min_size=1))
            if expected
            else None
        )
        Chunks(expected).validate(actual)

    @hp.given(data=st.data())
    def test_invalidates_with_boolean_mismatch(self, data: st.DataObject):
        """Should fail when the chunked state does not match a boolean."""
        expected = data.draw(st.booleans())
        actual = (
            None
            if expected
            else data.draw(st.lists(uniform_chunks(), min_size=1))
        )
        with pt.raises(ValidationError):
            Chunks(expected).validate(actual)

    @hp.given(data=st.data())
    def test_validates_with_integer_match(self, data: st.DataObject):
        """Should pass if the block sizes of all dimensions match an integer.

        In xarray, passing an integer to `chunk` creates uniform chunks (except
        the last chunk) along all dimensions.

        This is equivalent to testing the output of:
        DataArray(dims=('x', 'y'), ...).chunk(10).chunks
        """
        expected = data.draw(st.integers(min_value=1))
        actual = data.draw(
            st.lists(
                uniform_chunks(
                    min_block_size=expected, max_block_size=expected
                ),
                min_size=1,
            )
        )
        Chunks(expected).validate(actual)

    @hp.given(data=st.data())
    def test_integer_mismatch(self, data: st.DataObject):
        """Should fail if the block sizes of all dimensions do not match an integer."""
        expected = data.draw(st.integers(min_value=1, max_value=10))
        actual = data.draw(
            st.lists(uniform_chunks(min_block_size=expected + 1), min_size=1)
        )
        with pt.raises(ValidationError):
            Chunks(expected).validate(actual)

    @hp.given(data=st.data())
    def test_sequence_of_integer_match(self, data: st.DataObject):
        """Should pass if the block sizes of each dimension match a sequence of integers.

        This is equivalent to testing the output of:
        DataArray(dims=('x', 'y'), ...).chunk(x=10, y=11).chunks
        """
        expected = data.draw(st.lists(st.integers(min_value=1), min_size=1))
        actual = data.draw(
            st.tuples(
                *[
                    uniform_chunks(min_block_size=value, max_block_size=value)
                    for value in expected
                ]
            )
        )
        Chunks(expected).validate(actual)

    @pt.mark.parametrize(
        'expected, actual',
        [
            ([1], ((2,),)),
            ([1, 2], ((1,), (1, 1))),
            ([1, 2, 3], ((3,), (2, 2), (1, 1, 1))),
            ([1, 2, 3], ((1,), (2, 3), (3, 3, 4))),
        ],
    )
    def test_sequence_of_integer_mismatch(
        self, expected: Sequence[int], actual: tuple[tuple[int]]
    ):
        """Should fail if the block sizes of each dimension do not match a sequence of integers."""
        with pt.raises(ValidationError):
            Chunks(expected).validate(actual)

    @pt.mark.parametrize(
        'expected, actual',
        [
            ([[1]], ((1,),)),
            ([[1], [2, 2]], ((1,), (2, 2))),
            ([[1], [2, 2], [3, 3, 3]], ((1,), (2, 2), (3, 3, 3))),
        ],
    )
    def test_sequence_of_integer_sequence_match(
        self, expected: Sequence[Sequence[int]], actual: tuple[tuple[int]]
    ):
        """
        Test exact block sizes per dimension.

        It is equivalent to testing the output of:
        `DataArray(dims=('x', 'y'), ...).chunk(x=(1, 2, 3), y=(1, 2, 3)).chunks`
        """
        Chunks(expected).validate(actual)

    @pt.mark.parametrize(
        'expected, actual',
        [
            ([[1]], ((2,),)),
            ([[1], [2, 2]], ((1,), (2, 3))),
            ([[1], [2, 2], [3, 3, 3]], ((3,), (2, 2), (1, 1, 1))),
            ([[1], [2, 2], [3, 3, 3]], ((1,), (2, 2), (3, 3, 4))),
        ],
    )
    def test_sequence_of_integer_sequence_mismatch(
        self, expected: Sequence[Sequence[int]], actual: tuple[tuple[int]]
    ):
        with pt.raises(ValidationError):
            Chunks(expected).validate(actual)

    @hp.given(data=st.data())
    def test_wildcard_integer_match(self, data: st.DataObject):
        actual = data.draw(
            st.lists(
                uniform_chunks(max_dims=1),
                min_size=1,
            )
        )
        Chunks(-1).validate(actual)

    @hp.given(data=st.data())
    def test_wildcard_integer_mismatch(self, data: st.DataObject):
        actual = data.draw(
            st.lists(
                uniform_chunks(min_dims=2),
                min_size=1,
            )
        )
        with pt.raises(ValidationError):
            Chunks(-1).validate(actual)

from collections.abc import Sequence

import hypothesis as hp
import pytest as pt
from hypothesis import strategies as st
from jsonschema.exceptions import ValidationError

from xarray_model import Dims, Name


@st.composite
def dims(draw, min_size: int = 1, max_size: int | None = None):
    return draw(
        st.lists(
            elements=st.text(min_size=1),
            min_size=min_size,
            max_size=max_size,
        )
    )


class TestDims:
    @hp.given(instance=dims())
    def test_validates_default(self, instance: Sequence[str]):
        """Should always validate with default values."""
        Dims().validate(tuple(instance))

    @hp.given(instance=dims(), flag=st.booleans())
    def test_validates_sequence(self, instance: Sequence[str], flag: bool):
        """Should validate that the instance matches a sequence of names."""
        # NOTE: Complex validation provided by `Name` is tested separately in
        #  `test_name.py`
        expected = [Name(name) for name in instance] if flag else instance
        Dims(expected).validate(tuple(instance))

    @hp.given(instance=dims(), expected=dims(), flag=st.booleans())
    def test_invalidates_sequence(
        self, instance: Sequence[str], expected: Sequence[str], flag: bool
    ):
        """Should fail when the instance does not match a sequence of names."""
        hp.assume(expected != instance)
        expected = [Name(name) for name in expected] if flag else expected
        with pt.raises(ValidationError):
            Dims(expected).validate(tuple(instance))

    @hp.given(instance=dims(), data=st.data())
    def test_validates_contains(
        self, instance: Sequence[str], data: st.DataObject
    ):
        """Should validate that the instance contains a name."""
        expected = Name(data.draw(st.sampled_from(instance)))
        Dims(contains=expected).validate(tuple(instance))

    @hp.given(data=st.data())
    def test_validates_size(self, data: st.DataObject):
        min_size = data.draw(st.integers(min_value=1, max_value=10))
        max_size = data.draw(st.integers(min_value=min_size, max_value=100))
        hp.assume(min_size <= max_size)
        actual = data.draw(dims(min_size=min_size, max_size=max_size))
        Dims(min_size=min_size, max_size=max_size).validate(tuple(actual))

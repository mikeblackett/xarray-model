from collections.abc import Sequence

import hypothesis as hp
import pytest as pt
from hypothesis import strategies as st
from jsonschema import ValidationError

from xarray_model import Name


class TestName:
    @hp.given(name=st.text())
    def test_global_match(self, name: str):
        Name().validate(name)

    @hp.given(name=st.text())
    def test_string_match(self, name: str):
        Name(name).validate(name)

    @hp.given(data=st.data())
    def test_size_match(self, data: st.DataObject):
        min_size = data.draw(st.integers(min_value=1, max_value=10))
        max_size = data.draw(st.integers(min_value=min_size, max_value=20))
        hp.assume(min_size <= max_size)
        name = data.draw(st.text(min_size=min_size, max_size=max_size))
        Name(min_size=min_size, max_size=max_size).validate(name)

    @pt.mark.parametrize('name', [r'[a-z]+', r'^[a-z]+$'])
    @hp.given(data=st.data())
    def test_regex_match(self, data: st.DataObject, name: str):
        actual = data.draw(st.from_regex(regex=name))
        Name(name, regex=True).validate(actual)

    @hp.given(
        names=st.lists(st.text(min_size=1), min_size=1, unique=True),
        data=st.data(),
    )
    def test_iterable_match(self, names: Sequence[str], data: st.DataObject):
        actual = data.draw(st.sampled_from(names))
        Name(names).validate(actual)

    def test_raises_with_incompatible_args(self):
        with pt.raises(ValueError):
            Name(['a', 'b', 'c'], regex=True)

    def test_raises_with_string_mismatch(self):
        with pt.raises(ValidationError):
            Name('expected').validate('expected')

    def test_raises_with_size_mismatch(self):
        with pt.raises(ValidationError):
            Name(min_size=2, max_size=10).validate('a')

    def test_raises_with_iterable_mismatch(self):
        with pt.raises(ValidationError):
            Name(['a', 'b', 'c']).validate('z')

    def test_raises_with_regex_mismatch(self):
        name = r'[a-z]'
        with pt.raises(ValidationError):
            Name(name, regex=True).validate('A')

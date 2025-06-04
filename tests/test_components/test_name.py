import hypothesis as hp
import pytest as pt
from hypothesis import strategies as st
from jsonschema import ValidationError

from xarray_model import Name
import xarray.testing.strategies as xrst

from xarray_model.testing import patterns


class TestName:
    @hp.given(data=st.data())
    def test_arguments(self, data: st.DataObject) -> None:
        """Should always produce a valid JSON Schema"""
        regex = data.draw(st.one_of(st.none(), st.booleans()))
        if regex:
            name = data.draw(patterns())
        else:
            name = data.draw(
                st.one_of(st.none(), st.text(), st.lists(st.text()))
            )
        min_length = data.draw(st.integers(min_value=0))
        max_length = data.draw(st.integers(min_value=min_length))
        hp.assume(min_length < max_length)
        assert Name(
            name=name,
            regex=regex,
            min_length=min_length,
            max_length=max_length,
        ).schema

    @hp.given(instance=xrst.names())
    def test_validates_with_defaults(self, instance: str):
        """Should pass with default values."""
        Name().validate(instance)

    @hp.given(name=xrst.names())
    def test_validates_with_string_match(self, name: str):
        """Should pass if the name matches a string."""
        Name(name).validate(name)

    @hp.given(instance=xrst.names())
    def test_invalidates_with_string_mismatch(self, instance: str):
        """Should fail if the name does not match a string."""
        name = 'expected'
        hp.assume(name != instance)
        with pt.raises(ValidationError):
            Name(name).validate(instance)

    @hp.given(data=st.data())
    def test_validates_with_sequence_member(self, data: st.DataObject):
        """Should pass if the name is in a sequence."""
        name = data.draw(st.lists(xrst.names(), min_size=1, unique=True))
        instance = data.draw(st.sampled_from(name))
        Name(name).validate(instance)

    @hp.given(data=st.data())
    def test_invalidates_with_sequence_nonmember(self, data: st.DataObject):
        """Should fail if the name is not in a sequence."""
        name = data.draw(st.lists(xrst.names(), min_size=1, unique=True))
        instance = 'instance'
        hp.assume(instance not in name)
        with pt.raises(ValidationError):
            Name(name).validate(instance)

    @hp.given(name=patterns(), data=st.data())
    def test_validates_with_regex(self, name: str, data: st.DataObject):
        """Should pass if the name matches a regex pattern"""
        instance = data.draw(st.from_regex(name))
        Name(name, regex=True).validate(instance)

    @hp.given(pattern=patterns(), data=st.data())
    def test_invalidates_with_regex(self, pattern: str, data: st.DataObject):
        """Should fail if the name does not match a regex pattern"""
        name = r'^expected$'
        instance = data.draw(st.from_regex(pattern))
        with pt.raises(ValidationError):
            Name(name, regex=True).validate(instance)

    @hp.given(data=st.data())
    def test_validates_with_length_constraints(self, data: st.DataObject):
        """Should pass if the name satisfies the length constraints"""
        min_length = data.draw(st.integers(min_value=0, max_value=10))
        max_length = data.draw(st.integers(min_value=min_length, max_value=20))
        instance = data.draw(st.text(min_size=min_length, max_size=max_length))
        Name(min_length=min_length, max_length=max_length).validate(instance)

    @hp.given(data=st.data())
    def test_invalidates_with_size_constraints(self, data: st.DataObject):
        """Should fail if the name does not satisfy the length constraints"""
        min_length = data.draw(st.integers(min_value=1, max_value=10))
        max_length = data.draw(st.integers(min_value=min_length, max_value=20))
        instance = data.draw(
            st.one_of(
                st.text(min_size=max_length + 1),
                st.text(max_size=min_length - 1),
            )
        )
        with pt.raises(ValidationError):
            Name(min_length=min_length, max_length=max_length).validate(
                instance
            )

import hypothesis as hp
import pytest as pt
from hypothesis import strategies as st
from jsonschema import ValidationError

from xarray_model import Name


REGEX_PATTERNS = [
    r'[A-Za-z]{3,10}',  # Letters only, 3-10 chars
    r'\d{2,4}',  # 2-4 digits
    r'[A-Z][a-z]+\d{2}',  # Capitalized word with 2 digits
    r'[a-z]+[-_][a-z]+',  # Two lowercase words with dash/underscore
    r'[A-Z0-9]{8}',  # 8 chars of uppercase letters and numbers
    r'^TEST_[A-Z]+$',  # Uppercase word with TEST_ prefix
    r'^[a-z]+_\d{3}$',  # Lowercase word with 3 digits suffix
    r'^\d{2}-[A-Z]{3}$',  # 2 digits followed by 3 uppercase letters
    r'^[A-Z][a-z]+$',  # Single capitalized word
    r'^user_\d+$',  # Username with number
    r'^\d{4}-[A-Z]{2}$',  # 4 digits followed by 2 uppercase letters
    r'^[a-z]{2}\d{2}$',  # 2 lowercase letters followed by 2 digits
    r'^v\d+\.\d+$',  # Version number format
    r'^[A-Z]{2}_\d{4}$',  # 2 uppercase letters followed by 4 digits
    r'^[a-z]+@test$',  # Lowercase word with @test suffix
]


@st.composite
def patterns(draw: st.DrawFn) -> str:
    return draw(st.sampled_from(REGEX_PATTERNS))


@st.composite
def names(
    draw: st.DrawFn,
    regex: str | None = None,
    min_length: int = 0,
    max_length: int | None = None,
) -> str:
    if regex is not None:
        return draw(st.from_regex(regex=regex, fullmatch=True))
    return draw(st.text(min_size=min_length, max_size=max_length))


class TestName:
    @hp.given(data=st.data())
    def test_arguments(self, data: st.DataObject) -> None:
        """Should produce a valid JSON Schema with any combination of valid arguments"""
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

    @hp.given(instance=names())
    def test_validates_with_defaults(self, instance: str):
        """Should pass with default values."""
        Name().validate(instance)

    @hp.given(name=names())
    def test_validates_with_strings(self, name: str):
        """Should pass if the name matches a string."""
        Name(name).validate(name)

    @hp.given(instance=names())
    def test_invalidates_with_strings(self, instance: str):
        """Should fail if the name does not match a string."""
        name = 'expected'
        hp.assume(name != instance)
        with pt.raises(ValidationError):
            Name(name).validate(instance)

    @hp.given(data=st.data())
    def test_validates_with_sequences(self, data: st.DataObject):
        """Should pass if the name is in a sequence."""
        name = data.draw(st.lists(names(), min_size=1, unique=True))
        instance = data.draw(st.sampled_from(name))
        Name(name).validate(instance)

    @hp.given(data=st.data())
    def test_invalidates_with_sequences(self, data: st.DataObject):
        """Should fail if the name is not in a sequence."""
        name = data.draw(st.lists(names(), min_size=1, unique=True))
        instance = 'instance'
        hp.assume(instance not in name)
        with pt.raises(ValidationError):
            Name(name).validate(instance)

    @hp.given(name=patterns(), data=st.data())
    def test_validates_with_regex(self, name: str, data: st.DataObject):
        """Should pass if the name matches a regex pattern"""
        instance = data.draw(names(regex=name))
        Name(name, regex=True).validate(instance)

    @hp.given(pattern=patterns(), data=st.data())
    def test_invalidates_with_regex(self, pattern: str, data: st.DataObject):
        """Should fail if the name does not match a regex pattern"""
        name = r'^expected$'
        instance = data.draw(names(regex=pattern))
        with pt.raises(ValidationError):
            Name(name, regex=True).validate(instance)

    @hp.given(data=st.data())
    def test_validates_with_length_constraints(self, data: st.DataObject):
        """Should pass if the name satisfies the length constraints"""
        min_length = data.draw(st.integers(min_value=0, max_value=100))
        max_length = data.draw(st.integers(min_value=min_length))
        instance = data.draw(names(min_length=min_length, max_length=max_length))
        Name(min_length=min_length, max_length=max_length).validate(instance)

    @hp.given(data=st.data())
    def test_invalidates_with_size_constraints(self, data: st.DataObject):
        """Should fail if the name does not satisfy the length constraints"""
        min_length = data.draw(st.integers(min_value=1, max_value=100))
        max_length = data.draw(
            st.integers(min_value=min_length, max_value=1000)
        )
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

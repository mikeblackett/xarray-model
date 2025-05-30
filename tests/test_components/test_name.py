import hypothesis as hp
import pytest as pt
from hypothesis import strategies as st
from jsonschema import ValidationError

from xarray_model import Name


@st.composite
def names(
    draw,
    regex: str | None = None,
    min_size: int = 1,
    max_size: int | None = None,
) -> str:
    if regex is not None:
        return draw(st.from_regex(regex=regex, fullmatch=True))
    return draw(st.text(min_size=min_size, max_size=max_size))


class TestName:
    @hp.given(instance=names(min_size=0))
    def test_default_validation(self, instance: str):
        """Default values should validate with any string"""
        Name().validate(instance)

    @hp.given(instance=names())
    def test_validates_with_strings(self, instance: str):
        """Exact string matches should pass validation"""
        Name(instance).validate(instance)

    @hp.given(instance=names())
    def test_invalidates_with_strings(self, instance: str):
        """Exact string mismatches should fail validation"""
        expected = 'expected'
        hp.assume(expected != instance)  # probability is non-zero...!
        with pt.raises(ValidationError):
            Name('expected').validate(instance)

    @hp.given(data=st.data())
    def test_validates_with_sequences(self, data: st.DataObject):
        """A sequence of strings should validate against acceptable values"""
        expected = data.draw(st.lists(names(), min_size=1, unique=True))
        instance = data.draw(st.sampled_from(expected))
        Name(expected).validate(instance)

    @hp.given(data=st.data())
    def test_invalidates_with_sequences(self, data: st.DataObject):
        expected = data.draw(st.lists(names(), min_size=1, unique=True))
        instance = 'instance'
        hp.assume(instance not in expected)
        with pt.raises(ValidationError):
            Name(expected).validate(instance)

    @hp.given(data=st.data())
    def test_validates_with_regex(self, data: st.DataObject):
        """A regex pattern should validate against acceptable values"""
        pattern = r'[a-z]+[0-9]{2}$'
        actual = data.draw(names(regex=pattern))
        Name(pattern, regex=True).validate(actual)

    @hp.given(data=st.data())
    def test_invalidates_with_regex(self, data: st.DataObject):
        expected = r'[a-z]+[0-9]{2}$'
        instance = data.draw(names(regex=r'[0-9]{2}[a-z]+$'))
        with pt.raises(ValidationError):
            Name(expected, regex=True).validate(instance)

    @hp.given(data=st.data())
    def test_validates_with_length_constraints(self, data: st.DataObject):
        min_length = data.draw(st.integers(min_value=1, max_value=10))
        max_length = data.draw(st.integers(min_value=min_length))
        name = data.draw(st.text(min_size=min_length, max_size=max_length))
        Name(min_length=min_length, max_length=max_length).validate(name)

    @hp.given(data=st.data())
    def test_invalidates_with_size_constraints(self, data: st.DataObject):
        min_length = 1
        max_length = 5
        name = data.draw(st.text(min_size=max_length + 1))
        with pt.raises(ValidationError):
            Name(min_length=min_length, max_length=max_length).validate(name)

    # @pt.mark.parametrize('expected', [1, {'foo': 'bar'}])
    # def test_raises_with_invalid_args(self, expected: Any):
    #     with pt.raises(AssertionError):
    #         Name(expected).schema

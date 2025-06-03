from typing import Any
import hypothesis as hp
import pytest as pt
from hypothesis import strategies as st
from jsonschema import ValidationError

from xarray_model import Attr, Attrs
from xarray_model.testing import patterns


@st.composite
def attrs(
    draw,
    min_size: int = 0,
    max_size: int | None = None,
) -> dict[str, Any]:
    return draw(
        st.dictionaries(
            keys=st.text(min_size=1),
            values=st.one_of(
                st.none(), st.booleans(), st.integers(), st.floats(), st.text()
            ),
            min_size=min_size,
            max_size=max_size,
        )
    )


class TestAttr:
    @hp.given(
        key=st.text(min_size=1),
        regex=st.booleans(),
        value=st.one_of(
            st.none(), st.booleans(), st.integers(), st.floats(), st.text()
        ),
        required=st.booleans(),
        data=st.data(),
    )
    def test_arguments(
        self,
        key: str,
        regex: bool,
        value: Any,
        required: bool,
        data: st.DataObject,
    ):
        """Should always produce a valid JSON Schema"""
        if data.draw(st.booleans()) and value is not None:
            value = type(value)
        schema = Attr(key, regex=regex, value=value, required=required).schema

    @hp.given(
        key=st.text(min_size=1),
        regex=st.booleans(),
        value=st.one_of(
            st.none(), st.booleans(), st.integers(), st.floats(), st.text()
        ),
        required=st.booleans(),
        data=st.data(),
    )
    def test_validation_is_not_implemented(
        self,
        key: str,
        regex: bool,
        value: Any,
        required: bool,
        data: st.DataObject,
    ):
        """Attr should be composed with the Attrs model."""
        if data.draw(st.booleans()) and value is not None:
            value = type(value)
        with pt.raises(NotImplementedError):
            Attr(key, regex=regex, value=value, required=required).validate(
                _=None
            )


class TestAttrs:
    @hp.given(attrs_=attrs(), allow_extra_items=st.booleans())
    def test_arguments(self, attrs_: dict, allow_extra_items: bool):
        """Should always produce a valid JSON Schema"""
        expected = [Attr(key, value=value) for key, value in attrs_.items()]
        assert Attrs(expected, allow_extra_items=allow_extra_items).schema

    @hp.given(instance=attrs())
    def test_validates_with_defaults(self, instance: dict):
        """Should pass with default values."""
        Attrs().validate(instance)

    @hp.given(instance=attrs(min_size=1))
    def test_validates_with_extra_keys(self, instance: dict):
        """Should pass if the instance is not empty and extra items are allowed."""
        Attrs(allow_extra_items=True).validate(instance)

    @hp.given(instance=attrs(min_size=1))
    def test_invalidates_with_extra_keys(self, instance: dict):
        """Should fail if the instance is not empty and extra items are not allowed."""
        with pt.raises(ValidationError):
            Attrs(allow_extra_items=False).validate(instance)

    @hp.given(instance=attrs(min_size=1), required=st.booleans())
    def test_validates_with_key_match(self, instance: dict, required: bool):
        """Should pass if the instance contains an optional or required key."""
        key = next(iter(instance.keys()))
        expected = [Attr(key, required=required)]
        Attrs(expected).validate(instance)

    @hp.given(instance=attrs(min_size=1))
    def test_validates_with_optional_key(self, instance: dict):
        """Should pass if the instance does not contain an optional key."""
        key = 'random'
        hp.assume(key not in instance.keys())
        expected = [Attr(key, required=False)]
        Attrs(expected).validate(instance)

    @hp.given(instance=attrs(min_size=1))
    def test_invalidates_with_missing_required_key(self, instance: dict):
        """Should fail if the instance does not contain a required key."""
        key = 'random'
        hp.assume(key not in instance.keys())
        expected = [Attr(key, required=True)]
        with pt.raises(ValidationError):
            Attrs(expected).validate(instance)

    @hp.given(instance=attrs(min_size=1))
    def test_validates_with_required_keys(self, instance: dict):
        """Should pass if the instance contains the required key."""
        key = next(iter(instance.keys()))
        expected = [Attr(key)]
        Attrs(expected).validate(instance)

    @hp.given(instance=attrs(min_size=1))
    def test_invalidates_with_key_mismatch(self, instance: dict):
        """Should fail if the instance does not contain the specified key."""
        key = 'random'
        hp.assume(key not in instance.keys())
        expected = [Attr(key)]
        with pt.raises(ValidationError):
            Attrs(expected).validate(instance)

    @hp.given(instance=attrs(min_size=1), pattern=patterns(), data=st.data())
    def test_validates_with_key_pattern_match(
        self, instance: dict, pattern: str, data: st.DataObject
    ):
        """Should pass if the instance contains a key that matches the specified pattern."""
        expected = [Attr(key=pattern, regex=True)]
        key = data.draw(st.from_regex(pattern))
        instance |= {key: 'value'}
        Attrs(expected).validate(instance)

    def test_invalidates_with_key_pattern_mismatch(self):
        """Should fail if the instance contains a key that does not match the specified pattern."""
        expected = [Attr(key=r'^expected$', regex=True)]
        instance = {
            'random': 'value',
        }
        with pt.raises(ValidationError):
            # The ``required`` attribute doesn't apply to pattern properties,
            # so we need to apply `allow_extra_items=False`
            Attrs(expected, allow_extra_items=False).validate(instance)

    @hp.given(instance=attrs(min_size=1), data=st.data())
    def test_validates_with_key_value_match(
        self, instance: dict, data: st.DataObject
    ):
        """Should pass if the instance contains the specified key-value pair."""
        key, value = next(iter(instance.items()))
        expected = [Attr(key, value=value)]
        Attrs(expected).validate(instance)

    @hp.given(instance=attrs(min_size=1), data=st.data())
    def test_invalidates_with_key_value_mismatch(
        self, instance: dict, data: st.DataObject
    ):
        """Should fail if the instance does not contain the specified key-value pair."""
        # TODO: (mike) fix None values...
        key, value = next(iter(instance.items()))
        expected_value = data.draw(
            st.one_of(
                # st.none(),
                st.booleans(),
                st.integers(min_value=1),
                st.floats(),
                st.text(),
            )
        )
        hp.assume(value != expected_value)
        expected = [Attr(key, value=expected_value)]
        print(f'{expected_value=} vs {value=}')
        with pt.raises(ValidationError):
            Attrs(expected).validate(instance)

    @hp.given(instance=attrs(min_size=1))
    def test_validates_with_key_type_match(self, instance: dict):
        """Should pass if the instance contains the specified key-type pair."""
        key, value = next(iter(instance.items()))
        expected = [Attr(key, value=type(value))]
        Attrs(expected).validate(instance)

    @hp.given(instance=attrs(min_size=1), data=st.data())
    def test_invalidates_with_key_type_mismatch(
        self, instance: dict, data: st.DataObject
    ):
        """Should fail if the instance does not contain the specified key-type pair."""
        # TODO: (mike) think about float vs int validation
        key, value = next(iter(instance.items()))
        expected_value = data.draw(
            st.one_of(
                # st.none(),
                st.booleans(),
                # st.integers(min_value=1),
                # st.floats(),
                st.text(),
            )
        )
        hp.assume(type(value) is not type(expected_value))
        expected = [Attr(key, value=type(expected_value))]
        print(f'{expected_value=} vs {value=}')
        with pt.raises(ValidationError):
            Attrs(expected).validate(instance)

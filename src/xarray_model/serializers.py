from abc import ABC
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from re import Pattern
from typing import Any, Self, Type

from xarray_model.checks import (
    check_uniqueness,
    check_non_negative,
    check_valid_size,
)
from xarray_model.encoders import (
    encode_value,
    encode_keyword,
    decode_keyword,
    encode_type,
    decode_type,
)

__all__ = [
    'ArraySerializer',
    'BooleanSerializer',
    'ConstSerializer',
    'DeserializationError',
    'EnumSerializer',
    'IntegerSerializer',
    'NotSerializer',
    'NullSerializer',
    'NumberSerializer',
    'ObjectSerializer',
    'SerializationError',
    'Serializer',
    'StringSerializer',
    'TypeSerializer',
    'as_schema',
]


class SerializationError(Exception):
    """Raised when serialization fails."""

    def __init__(self, message: str):
        super().__init__(message)


class DeserializationError(Exception):
    """Raised when deserialization fails."""

    def __init__(self, message: str):
        super().__init__(message)


def as_schema(obj: 'Serializer') -> dict:
    return asdict(obj, dict_factory=_schema_factory)


@dataclass(frozen=True, kw_only=True)
class Serializer(ABC):
    title: str | None = None
    description: str | None = None
    comment: str | None = None

    def serialize(self) -> dict[str, Any]:
        return as_schema(self)

    @classmethod
    def from_schema(cls, obj: Mapping[str, Any]) -> Self:
        kwargs = {
            decode_keyword(f.name): _decode_json_value(
                obj.get(encode_keyword(f.name))
            )
            for f in fields(cls)
            if f.init
        }
        return cls(**kwargs)

    def __repr__(self):
        # repr signature with only non-default arguments
        args = [
            (f.name, getattr(self, f.name))
            for f in fields(self)
            if getattr(self, f.name) != f.default and f.init
        ]
        args_string = ', '.join(f'{name}={value}' for name, value in args)
        return f'{self.__class__.__name__}({args_string})'


@dataclass(frozen=True, kw_only=True, repr=False)
class EnumSerializer(Serializer):
    def __post_init__(self):
        check_valid_size(self.enum, min_size=1)
        check_uniqueness(self.enum)

    enum: Iterable[Any] = field(kw_only=False)


@dataclass(frozen=True, kw_only=True, repr=False)
class ObjectSerializer(Serializer):
    type: str = field(default='object', init=False)

    properties: Mapping[str, Serializer] | None = None
    pattern_properties: Mapping[str, Serializer] | None = None
    additional_properties: Serializer | bool | None = None
    required: Iterable[str] | None = None
    max_properties: int | None = None
    min_properties: int | None = None

    def __post_init__(self):
        if self.min_properties is not None:
            check_non_negative(self.min_properties)
        if self.max_properties is not None:
            check_non_negative(self.max_properties)
        if self.required is not None:
            check_uniqueness(self.required)


@dataclass(frozen=True, kw_only=True, repr=False)
class ArraySerializer(Serializer):
    type: str = field(default='array', init=False)

    items: Serializer | bool | None = None
    prefix_items: Sequence[Serializer] | None = None
    unevaluated_items: Serializer | None = None
    contains: Serializer | None = None
    max_contains: int | None = None
    min_contains: int | None = None
    max_items: int | None = None
    min_items: int | None = None

    def __post_init__(self):
        if self.min_items is not None:
            check_non_negative(self.min_items)
        if self.max_items is not None:
            check_non_negative(self.max_items)
        if self.min_contains is not None:
            check_non_negative(self.min_contains)
        if self.max_contains is not None:
            check_non_negative(self.max_contains)


@dataclass(frozen=True, kw_only=True, repr=False)
class StringSerializer(Serializer):
    type: str = field(default='string', init=False)

    max_length: int | None = None
    min_length: int | None = None
    pattern: str | Pattern[str] | None = None

    def __post_init__(self):
        if self.min_length is not None:
            check_non_negative(self.min_length)
        if self.max_length is not None:
            check_non_negative(self.max_length)


@dataclass(frozen=True, kw_only=True, repr=False)
class IntegerSerializer(Serializer):
    type: str = field(default='integer', init=False)

    multiple_of: int | None = None
    maximum: int | None = None
    minimum: int | None = None
    exclusive_maximum: int | None = None
    exclusive_minimum: int | None = None

    def __post_init__(self):
        if self.multiple_of is not None:
            check_non_negative(self.multiple_of)


@dataclass(frozen=True, kw_only=True, repr=False)
class NumberSerializer(Serializer):
    type: str = field(default='number', init=False)

    multiple_of: int | float | None = None
    maximum: int | float | None = None
    minimum: int | float | None = None
    exclusive_maximum: int | float | None = None
    exclusive_minimum: int | float | None = None

    def __post_init__(self):
        if self.multiple_of is not None:
            check_non_negative(self.multiple_of)


@dataclass(frozen=True, repr=False, kw_only=False)
class NullSerializer(Serializer):
    type: str = field(default='null', init=False)


@dataclass(frozen=True, repr=False, kw_only=False)
class BooleanSerializer(Serializer):
    type: str = field(default='boolean', init=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class ConstSerializer(Serializer):
    const: Any = field(kw_only=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class TypeSerializer(Serializer):
    type_: Type = field(kw_only=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class NotSerializer(Serializer):
    not_: Serializer = field(kw_only=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class OneOfSerializer(Serializer):
    one_of: Iterable[Serializer] = field(kw_only=False)


def _encode_field_value(value: Any):
    """Encode a Serializer field value into a JSON schema."""
    if isinstance(value, Serializer):
        return value.serialize()
    if isinstance(value, type):
        return encode_type(value)
    return encode_value(value)


def _decode_json_value(value: Any):
    try:
        # TODO: (mike) this is hacky...
        return decode_type(value)
    except Exception:
        pass
    return value


def _encode_dict(data) -> dict:
    """Encode a dictionary into a JSON schema."""
    schema = {}
    for k, v in data.items():
        if v is not None:
            schema[encode_keyword(k)] = (
                _encode_dict(v)
                if isinstance(v, Mapping)
                else _encode_field_value(v)
            )
    return schema


def _schema_factory(obj: 'Serializer') -> dict:
    """Encode a serializer into a JSON schema."""
    data = dict(obj)
    return _encode_dict(data)

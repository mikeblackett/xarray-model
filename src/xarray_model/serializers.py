from abc import ABC
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from re import Pattern
from typing import Any, Type

from xarray_model.encoders import (
    encode_value,
    encode_keyword,
)

__all__ = [
    'AnySerializer',
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
    """Encode a Serializer as a JSON Schema schema"""
    return asdict(obj, dict_factory=_schema_factory)


@dataclass(frozen=True, kw_only=True)
class Serializer(ABC):
    """Abstract base class for serializing Python objects to JSON Schema.

    Each serializer should correspond to a JSON Schema Data Type. The attributes
    of a Serializer class should correspond to JSON Schema keywords. The values
    corresponding to the keywords can be Python primitives, or other `Serializers`.

    Keyword attributes should be declared in snake_case without prefixes, they
    will be converted to JSON Schema when serializing.

    Attributes
    ----------
    title : str, optional
        A short description of the instance described by this schema.
    description : str, optional
        A description about the purpose of the instance described by this schema.
    comment : str, optional
        Notes that may be useful to future editors of a JSON schema, but should
        not be used to communicate to users of the schema.
    """

    title: str | None = None
    description: str | None = None
    comment: str | None = None

    def serialize(self) -> dict[str, Any]:
        """Convert this serializer to a JSON schema.

        This is a convenience method that wraps `as_schema`.
        """
        return as_schema(self)

    # @classmethod
    # def from_schema(cls, obj: Mapping[str, Any]) -> Self:
    #     # TODO: (mike) implement `Serializer.from_schema`
    #     kwargs = {
    #         decode_keyword(f.name): _decode_json_value(
    #             obj.get(encode_keyword(f.name))
    #         )
    #         for f in fields(cls)
    #         if f.init
    #     }
    #     return cls(**kwargs)

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
    """Serializer for enum type

    Attributes
    ----------
    enum : Iterable[Any]
        An iterable describing a fixed set of acceptable values. The iterable
        must contain at least one value and each value must be unique.
    """

    enum: Iterable[Any] = field(kw_only=False)


@dataclass(frozen=True, kw_only=True, repr=False)
class ObjectSerializer(Serializer):
    """Serializer for mapping type

    Attributes
    ----------
    properties : Mapping[str, Serializer] | None, default None
        A mapping where each key is the name of a property and each value is a
        `Serializer` used to validate that property.
    pattern_properties : Mapping[str, Serializer] | None, default None
        A mapping where each key is a regular expression used to match the name
        of a property and each value is a `Serializer` used to validate that
        property.
    additional_properties : Serializer | bool | None, default None
        A schema that will be used to validate any properties in the instance
        that are not matched by `properties` or `patternProperties`. Boolean
        values can be used to allow/disallow any additional properties.
    required : Iterable[str] | None, default None
        An iterable of zero or more unique strings describing a list of
        required properties. Any properties not included in this list are treated
        as optional.
    max_properties : int | None, default None
        A non-negative integer used to restrict the number of properties on an
        object.
    min_properties : int | None, default None
        A non-negative integer used to restrict the number of properties on an
        object.
    """

    # TODO: (mike) replace `additionalProperties` keyword with `unevaluatedProperties`
    # TODO: (mike) add `propertyNames` keyword

    type: str = field(default='object', init=False)

    properties: Mapping[str, Serializer] | None = None
    pattern_properties: Mapping[str, Serializer] | None = None
    additional_properties: Serializer | bool | None = None
    required: Iterable[str] | None = None
    max_properties: int | None = None
    min_properties: int | None = None


@dataclass(frozen=True, kw_only=True, repr=False)
class ArraySerializer(Serializer):
    """Serializer for sequence type

    Attributes
    ----------
    items : Serializer | bool | None, default None
        A single schema that will be used to validate all the items in the array.
        The empty array is always valid.
    prefix_items : Sequence[Serializer] | None, default None
        An array, where each item is a `Serializer` that corresponds to each
        index of the instance's array.
    unevaluated_items : Serializer | None, default None
        A schema that applies to any values not evaluated by the `items`,
        `prefix_items`, or `contains` keyword.
    contains : Serializer | None, default None
        A schema that only needs to validate against one or more items in the
        array.
    max_contains : int | None, default None
        A non-negative integer used to restrict the number of times a schema
        matches a `contains` constraint.
    min_contains : int | None, default None
        A non-negative integer used to restrict the number of times a schema
        matches a `contains` constraint.
    max_items : int | None, default None
        A non-negative integer used to restrict the number of items in an array.
    min_items : int | None, default None
        A non-negative integer used to restrict the number of items in an array.
    """

    type: str = field(default='array', init=False)

    items: Serializer | bool | None = None
    prefix_items: Sequence[Serializer] | None = None
    unevaluated_items: Serializer | None = None
    contains: Serializer | None = None
    max_contains: int | None = None
    min_contains: int | None = None
    max_items: int | None = None
    min_items: int | None = None


@dataclass(frozen=True, kw_only=True, repr=False)
class StringSerializer(Serializer):
    """Serializer for string type

    Attributes
    ----------
    max_length : int | None, default None
        A non-negative integer used to restrict the number of characters in a
        string.
    min_length : int | None, default None
        A non-negative integer used to restrict the number of characters in a
        string.
    pattern : str | Pattern[str] | None, default None
        A pattern used to restrict a string to a particular regular expression.
    """

    type: str = field(default='string', init=False)

    max_length: int | None = None
    min_length: int | None = None
    pattern: str | Pattern[str] | None = None


@dataclass(frozen=True, kw_only=True, repr=False)
class IntegerSerializer(Serializer):
    """Serializer for integer type

    Attributes
    ----------
    multiple_of : int | float | None, default None
        A positive number used to restrict the instance to a multiple of a
        given number
    maximum : int | None, default None
        A number used to restrict the instance to a maximum value.
    minimum : int | None, default None
        A number used to restrict the instance to a minimum value.
    exclusive_maximum : int | None, default None
        A number used to restrict the instance to a maximum value.
    exclusive_minimum : int | None, default None
        A number used to restrict the instance to a minimum value.
    """

    type: str = field(default='integer', init=False)

    multiple_of: int | None = None
    maximum: int | None = None
    minimum: int | None = None
    exclusive_maximum: int | None = None
    exclusive_minimum: int | None = None


@dataclass(frozen=True, kw_only=True, repr=False)
class NumberSerializer(Serializer):
    """Serializer for numeric type, either integers or floating point numbers.

    Attributes
    ----------
    multiple_of : int | float | None, default None
        A positive number used to restrict the instance to a multiple of a
        given number
    maximum : int | float | None, default None
        A number used to restrict the instance to a maximum value.
    minimum : int | float | None, default None
        A number used to restrict the instance to a minimum value.
    exclusive_maximum : int | float | None, default None
        A number used to restrict the instance to a maximum value.
    exclusive_minimum : int | float | None, default None
        A number used to restrict the instance to a minimum value.
    """

    type: str = field(default='number', init=False)

    multiple_of: int | float | None = None
    maximum: int | float | None = None
    minimum: int | float | None = None
    exclusive_maximum: int | float | None = None
    exclusive_minimum: int | float | None = None


@dataclass(frozen=True, repr=False, kw_only=False)
class NullSerializer(Serializer):
    """Serializer for null type"""

    type: str = field(default='null', init=False)


@dataclass(frozen=True, repr=False, kw_only=False)
class BooleanSerializer(Serializer):
    """Serializer for boolean type"""

    type: str = field(default='boolean', init=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class ConstSerializer(Serializer):
    """Serializer for constant type

    Attributes
    ----------

    const : Any
        The expected value of the instance
    """

    const: Any = field(kw_only=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class AnySerializer(Serializer):
    """Serializer for data type keyword"""

    ...


@dataclass(frozen=True, repr=False, kw_only=True)
class TypeSerializer(Serializer):
    """Serializer for data type keyword"""

    type_: Type = field(kw_only=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class AllOfSerializer(Serializer):
    all_of: Iterable[Serializer] = field(kw_only=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class AnyOfSerializer(Serializer):
    any_of: Iterable[Serializer] = field(kw_only=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class OneOfSerializer(Serializer):
    one_of: Iterable[Serializer] = field(kw_only=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class NotSerializer(Serializer):
    not_: Serializer = field(kw_only=False)


def _schema_factory(data):
    """Custom dict_factory for dataclasses.asdict."""
    return {
        encode_keyword(k): encode_value(v) for k, v in data if v is not None
    }

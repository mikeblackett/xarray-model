from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Any, Iterable, Self

from numpy.typing import DTypeLike

from xarray_model.serializers import (
    ConstSerializer,
    DeserializationError,
    IntegerSerializer,
    StringSerializer,
    TypeSerializer, ArraySerializer,
)

__all__ = [
    'Attrs',
    'Attr',
    'Datatype',
    'Dims',
    'Name',
    'Shape',
]


@dataclass(frozen=True, kw_only=True, repr=False)
class Component(ABC):
    """Base class for components of a xarray schema."""

    title: str | None
    description: str | None

    @cached_property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        return {
            'title': self.title,
            'description': self.description,
        }

    @classmethod
    @abstractmethod
    def from_schema(cls, data: dict[str, Any]) -> Self: ...

    def __repr__(self):
        # Override repr to show only non-default arguments.
        args = [
            (f.name, getattr(self, f.name))
            for f in fields(self)
            if getattr(self, f.name) != f.default
        ]
        args_string = ''.join(f'{name}={value}' for name, value in args)
        return f'{self.__class__.__name__}({args_string!r})'


@dataclass(frozen=True, kw_only=True, repr=False)
class Shape(Component):
    shape: Sequence[int | None] = field(kw_only=False)
    title: str | None = 'Array shape'
    description: str | None = 'Tuple of array dimensions.'

    @cached_property
    def schema(self) -> dict[str, Any]:
        return super().schema | ArraySerializer(prefix_items=[
            ConstSerializer(size)
            if size
            else IntegerSerializer()
            for size in self.shape
        ]).serialize()
        # return {
        #     **super().schema,
        #     'type': 'array',
        #     'prefixItems': [
        #         ConstSerializer(size).serialize()
        #         if size
        #         else IntegerSerializer().serialize()
        #         for size in self.shape
        #     ],
        # }

    @classmethod
    def from_schema(cls, data: dict[str, Any]) -> Self:
        match data:
            case {'prefixItems': list()}:
                return cls(shape=[])
            case _:
                raise ValueError('Invalid dimensions')


@dataclass(frozen=True, kw_only=True, repr=False)
class Datatype(Component):
    dtype: DTypeLike = field(kw_only=False)
    title: str | None = 'Array dtype'
    description: str | None = 'Data-type of the array’s elements.'

    @cached_property
    def schema(self) -> dict[str, Any]:
        return super().schema | ConstSerializer(self.dtype).serialize()

    @classmethod
    def from_schema(cls, data: dict[str, Any]) -> Self:
        return ConstSerializer.deserialize(**data)


@dataclass(frozen=True, kw_only=True, repr=False)
class Name(Component):
    name: str = field(kw_only=False)
    regex: bool = False
    title: str | None = 'Array name'
    description: str | None = 'The name of this array.'

    @cached_property
    def schema(self) -> dict[str, Any]:
        result = super().schema
        if self.regex:
            return result | StringSerializer(self.name).serialize()
        return result | ConstSerializer(self.name).serialize()

    @classmethod
    def from_schema(cls, data: dict[str, Any]) -> Self:
        try:
            kwargs = StringSerializer.deserialize(**data)
        except DeserializationError:
            kwargs = ConstSerializer.deserialize(**data)
        return cls(**kwargs)


@dataclass(frozen=True, kw_only=True, repr=False)
class Dims(Component):
    names: Sequence['Name']
    title: str | None = 'Dimension names'
    description: str | None = (
        'Tuple of dimension names associated with this array.'
    )

    @cached_property
    def schema(self) -> dict[str, Any]:
        return {
            **super().schema,
            'type': 'array',
            'prefixItems': [name.schema for name in self.names],
        }

    @classmethod
    def from_schema(cls, data: dict[str, Any]) -> Self:
        match data:
            case {'prefixItems': list()}:
                return cls(
                    names=[
                        Name.from_schema(**item)
                        for item in data['prefixItems']
                    ]
                )
            case _:
                raise ValueError('Invalid dimensions')


@dataclass(frozen=True, kw_only=True, repr=False)
class Attr(Component):
    name: str
    regex: bool = False
    value: Any | None = None
    required: bool = True
    description: str | None = 'Arbitrary metadata value'
    title: str | None = 'Attr'

    @cached_property
    def schema(self) -> dict[str, Any]:
        if isinstance(self.value, type):
            print(self.value)
            return TypeSerializer(self.value).serialize()
        return ConstSerializer(self.value).serialize()

    @classmethod
    def from_schema(cls, data: dict[str, Any]) -> Self:
        try:
            kwargs = StringSerializer.deserialize(**data)
        except DeserializationError:
            kwargs = ConstSerializer.deserialize(**data)
        return cls(**kwargs)


@dataclass(frozen=True, kw_only=True, repr=False)
class Attrs(Component):
    attrs: Iterable[Attr] = field(kw_only=False)
    allow_extra_keys: bool = True

    title: str | None = 'Metadata'
    description: str | None = (
        'Dictionary storing arbitrary metadata with this array.'
    )

    @cached_property
    def _required(self):
        return [
            attr.name
            for attr in self.attrs
            if attr.required and not attr.regex
        ]

    @cached_property
    def _properties(self) -> dict[str, Any]:
        return {
            attr.name: attr.schema for attr in self.attrs if not attr.regex
        }

    @cached_property
    def _pattern_properties(self) -> dict[str, Any]:
        return {attr.name: attr.schema for attr in self.attrs if attr.regex}

    @cached_property
    def schema(self) -> dict[str, Any]:
        schema = {**super().schema, 'type': 'object'}
        if self._properties:
            schema |= {'properties': self._properties}
        if self._pattern_properties:
            schema |= {'patternProperties': self._pattern_properties}
        if self._required:
            schema |= {'required': self._required}
        schema |= {'additionalProperties': self.allow_extra_keys}
        return schema

    @classmethod
    def from_schema(cls, **kwargs) -> Self: ...

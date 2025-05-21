from collections.abc import Sequence, Hashable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Iterable

from numpy.typing import DTypeLike
from xarray_model.base import Base
from xarray_model.serializers import (
    ConstSerializer,
    Serializer,
    StringSerializer,
    TypeSerializer,
    ArraySerializer,
    IntegerSerializer,
    ObjectSerializer,
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
class Shape(Base):
    shape: Sequence[int | None] = field(kw_only=False)
    title: str | None = 'Array shape'
    description: str | None = 'Tuple of array dimensions.'

    @cached_property
    def serializer(self) -> Serializer:
        return ArraySerializer(
            title=self.title,
            description=self.description,
            prefix_items=[
                ConstSerializer(size) if size else IntegerSerializer()
                for size in self.shape
            ],
        )

    def validate(self, shape: tuple[int]) -> None:
        return super()._validate(instance=(list(shape)))


@dataclass(frozen=True, kw_only=True, repr=False)
class Datatype(Base):
    # TODO: (mike) support numpy subdytpes
    dtype: DTypeLike = field(kw_only=False)
    title: str | None = 'Array dtype'
    description: str | None = 'Data-type of the array’s elements.'

    @cached_property
    def serializer(self) -> Serializer:
        return ConstSerializer(
            title=self.title,
            description=self.description,
            const=self.dtype,
        )

    def validate(self, dtype: DTypeLike) -> None:
        return super()._validate(instance=dtype)


@dataclass(frozen=True, kw_only=True, repr=False)
class Name(Base):
    name: str = field(kw_only=False)
    regex: bool = False
    title: str | None = 'Array name'
    description: str | None = 'The name of this array.'

    @cached_property
    def serializer(self) -> Serializer:
        if self.regex:
            return StringSerializer(
                title=self.title,
                description=self.description,
                pattern=self.name,
            )
        return ConstSerializer(
            title=self.title,
            description=self.description,
            const=self.name,
        )

    def validate(self, name: str) -> None:
        return super()._validate(instance=name)


@dataclass(frozen=True, kw_only=True, repr=False)
class Dims(Base):
    names: Sequence['Name']
    title: str | None = 'Dimension names'
    description: str | None = (
        'Tuple of dimension names associated with this array.'
    )

    @cached_property
    def serializer(self) -> Serializer:
        return ArraySerializer(
            title=self.title,
            description=self.description,
            prefix_items=[name.serializer for name in self.names],
        )

    def validate(self, dims: tuple[Hashable]) -> None:
        return super()._validate(instance=list(dims))


@dataclass(frozen=True, kw_only=True, repr=False)
class Attr(Base):
    name: str
    regex: bool = False
    value: Any | None = None
    required: bool = True
    description: str | None = 'Arbitrary metadata value'
    title: str | None = 'Attr'

    @cached_property
    def serializer(self) -> Serializer:
        if isinstance(self.value, type):
            return TypeSerializer(self.value)
        return ConstSerializer(self.value)

    def validate(self, attr) -> None:
        raise NotImplementedError(
            'Attr is not meant to be validated in isolation. '
            'You should compose it inside the Attrs schema.'
        )


@dataclass(frozen=True, kw_only=True, repr=False)
class Attrs(Base):
    attrs: Iterable[Attr] = field(kw_only=False)
    allow_extra_keys: bool = True

    title: str | None = 'Metadata'
    description: str | None = (
        'Dictionary storing arbitrary metadata with this array.'
    )

    @cached_property
    def serializer(self) -> Serializer:
        return ObjectSerializer(
            title=self.title,
            description=self.description,
            properties={
                attr.name: attr.serializer
                for attr in self.attrs
                if not attr.regex
            },
            pattern_properties={
                attr.name: attr.serializer for attr in self.attrs if attr.regex
            },
            required=[
                attr.name
                for attr in self.attrs
                if attr.required and not attr.regex
            ],
            additional_properties=self.allow_extra_keys,
        )

    def validate(self, attrs: Mapping[str, Any]) -> None:
        return super()._validate(instance=attrs)

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
    NullSerializer,
)

__all__ = [
    'Attrs',
    'Attr',
    'Chunks',
    'Datatype',
    'Dims',
    'Name',
    'Shape',
]

from xarray_model.types import ChunksType


@dataclass(frozen=True, kw_only=True, repr=False)
class _Chunk(Base):
    """
    DataArray chunk model

    This model represents the tuple of block sizes for a single dimension, it
    is supposed to be used inside the Chunks schema.

    Parameters
    ----------
    shape : int | Sequence[int]
    The expected shape of the chunk. If a single ``int`` is provided, it is
    matched against the first block size in the tuple. If a ``Sequence`` is
    provided, a full match is required.
    """

    # TODO: (mike) support `-1` wildcard to match chunksize to dimension size?

    shape: int | Sequence[int] = field(kw_only=False)

    @cached_property
    def serializer(self) -> Serializer:
        if isinstance(self.shape, int):
            if self.shape == -1:
                prefix_items = [IntegerSerializer()]
                items = False
            else:
                prefix_items = [ConstSerializer(const=self.shape)]
                items = IntegerSerializer()
        elif isinstance(self.shape, Sequence):
            prefix_items = [
                ConstSerializer(const=size) if size else IntegerSerializer()
                for size in self.shape
            ]
            items = False
        else:
            raise ValueError(
                f'Invalid shape: {self.shape}. '
                'Expected int or sequence of ints.'
            )

        return ArraySerializer(
            prefix_items=prefix_items,
            items=items,
        )

    def validate(self, _) -> None:
        raise NotImplementedError(
            'Chunk is not meant to be validated in isolation. '
            'You should compose it inside the Chunks schema.'
        )


@dataclass(frozen=True, kw_only=True, repr=False)
class Chunks(Base):
    """
    DataArray chunks model

    Use this model to validate the result of ``DataArray.chunks``

    Parameters
    ----------
    chunks : bool | Sequence[Chunk]
        If a boolean is provided, it is used to validate whether the
        ``DataArray`` is chunked or not. To validate the actual chunk sizes, a
        sequence of ``Chunk`` models should be provided.
    """

    chunks: bool | Sequence[Sequence[int] | int] = field(kw_only=False)

    title: str | None = 'Chunks'
    description: str | None = (
        'Tuple of block lengths for this dataarray’s data'
    )

    def __post_init__(self):
        _chunks = [
            chunk if isinstance(chunk, _Chunk) else _Chunk(chunk)
            for chunk in self.chunks
        ]
        object.__setattr__(self, 'chunks', _chunks)

    @cached_property
    def serializer(self) -> Serializer:
        if not self.chunks:
            return NullSerializer(
                title=self.title,
                description=self.description,
            )
        if isinstance(self.chunks, Sequence):
            prefix_items = [chunk.serializer for chunk in self.chunks]
            items = None
        else:
            items = ArraySerializer(items=IntegerSerializer())
            prefix_items = None
        return ArraySerializer(
            title=self.title,
            description=self.description,
            prefix_items=prefix_items,
            items=items,
        )

    def validate(self, chunks: ChunksType) -> None:
        chunks = list(list(chunk) for chunk in chunks) if chunks else None
        return super()._validate(instance=chunks)


@dataclass(frozen=True, kw_only=True, repr=False)
class Shape(Base):
    """
    DataArray shape model

    Use this model to validate the result of ``DataArray.shape``
    """

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
    names: Sequence['Name'] = field(kw_only=False)
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

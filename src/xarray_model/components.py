from collections.abc import Sequence, Hashable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Iterable

from numpy.typing import DTypeLike
from xarray_model.base import Base
from xarray_model.serializers import (
    ConstSerializer,
    EnumSerializer,
    IntegerSerializer,
    NullSerializer,
    ObjectSerializer,
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
                # `-1` is a dask wildcard meaning "use the full dimension size."
                # Expect a length 1 array of integers.
                prefix_items = [IntegerSerializer()]
                items = False
            else:
                # A single integer is used to represent a uniform chunk size.
                # In this case, we expect only the first block size in the tuple to
                # match the provided shape.
                # TODO: (mike) this is not a very robust test... Ideally we
                #  would want to specify an `items` keyword together with a
                #  `min_items` and `max_items` equal to the number of chunks -1,
                #  which would correspond to the dask criteria for "uniform chunks sizes".
                #  But we don't know the number of chunks at schema creation time,
                #  only at validation time...
                prefix_items = [ConstSerializer(const=self.shape)]
                items = IntegerSerializer()
        elif isinstance(self.shape, Sequence):
            # Expect a full match of the provided shape to the chunk size.
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
    DataArray chunks validation model

    Use this model to validate the result of ``DataArray.chunks``

    Parameters
    ----------
    chunks : ChunksType
        If a boolean is provided, it is used to validate whether the
        ``DataArray`` is chunked or not. To validate the actual block sizes, a
        sequence of chunk sizes can be provided. A sequence of integers is used
        to represent an expected uniform chunk size. To perform an exact match,
        a sequence of integer sequences can be used.
    """

    chunks: bool | Sequence[Sequence[int] | int] = field(kw_only=False)

    def __post_init__(self):
        _chunks = [
            chunk if isinstance(chunk, _Chunk) else _Chunk(chunk)
            for chunk in self.chunks
        ]
        object.__setattr__(self, 'chunks', _chunks)

    @cached_property
    def serializer(self) -> Serializer:
        if not self.chunks:
            return NullSerializer()
        if isinstance(self.chunks, Sequence):
            prefix_items = [chunk.serializer for chunk in self.chunks]  # type: ignore[union-attr]
            items = False
            min_items = max_items = len(prefix_items)
        else:
            # Must be `True`
            items = ArraySerializer(items=IntegerSerializer())
            prefix_items = None
            min_items = max_items = None
        return ArraySerializer(
            prefix_items=prefix_items,
            items=items,
            min_items=min_items,
            max_items=max_items,
        )

    def validate(self, chunks: tuple[tuple[int, ...], ...] | None) -> None:
        chunks = list(list(chunk) for chunk in chunks) if chunks else None
        return super()._validate(instance=chunks)


@dataclass(frozen=True, kw_only=True, repr=False)
class Shape(Base):
    """
    DataArray shape validation model

    Use this model to validate the result of ``DataArray.shape``

    Parameters
    ----------
    shape : Sequence[int] | None, default None
        Expected shape of the array. The default value of ``None`` will match
        any shape. An integer value of ``-1`` can be used as a wildcard.
    min_size : int | None, default None
        Minimum length of the shape sequence i.e., the number of dimensions.
    max_size : int | None, default None
        Maximum length of the shape sequence i.e., the number of dimensions.
    """

    shape: Sequence[int] | None = field(default=None, kw_only=False)
    min_size: int | None = None
    max_size: int | None = None

    @cached_property
    def serializer(self) -> Serializer:
        if isinstance(self.shape, Sequence):
            prefix_items = [
                IntegerSerializer() if size == -1 else ConstSerializer(size)
                for size in self.shape
            ]
            items = None
            min_items = (
                len(self.shape) if self.min_size is None else self.min_size
            )
            max_items = (
                len(self.shape) if self.max_size is None else self.max_size
            )
        else:
            prefix_items = None
            items = IntegerSerializer()
            min_items = self.min_size
            max_items = self.max_size
        return ArraySerializer(
            prefix_items=prefix_items,
            items=items,
            min_items=min_items,
            max_items=max_items,
        )

    def validate(self, shape: tuple[int, ...]) -> None:
        return super()._validate(instance=(list(shape)))


@dataclass(frozen=True, kw_only=True, repr=False)
class Datatype(Base):
    """
    DataArray dtype validation model

    Use this model to validate the result of ``DataArray.dtype``

    Parameters
    ----------
    dtype : DTypeLike | None, default None
        The expected data type of the array. This can be a NumPy dtype or the
        name of a dtype. The default value of ``None`` will match any dtype.
    """

    # TODO: (mike) support numpy subdytpes
    dtype: DTypeLike | None = field(default=None, kw_only=False)

    @cached_property
    def serializer(self) -> Serializer:
        return ConstSerializer(self.dtype)

    def validate(self, dtype: np.dtype) -> None:
        return super()._validate(instance=dtype)


@dataclass(frozen=True, kw_only=True, repr=False)
class Name(Base):
    """
    DataArray name validation model

    Parameters
    ----------
    name : str | Sequence[str] | None, default None
        Expected value, sequence of acceptable values, or regex pattern to
        match the name against. The default value of ``None`` will match any
        name.
    regex : bool, default False
        Flag to indicate that the ``name`` parameter should be treated as a
        regex pattern.
    min_size : int, default None
        Minimum length of the name. Only used if ``name`` is a string.
    max_size : int, default None
        Maximum length of the name. Only used if ``name`` is a string.
    """

    name: str | Sequence[str] | None = field(default=None, kw_only=False)
    regex: bool = False
    min_length: int | None = None
    max_length: int | None = None
    title: str | None = 'Array name'
    description: str | None = 'The name of this array.'

    @cached_property
    def serializer(self) -> Serializer:
        if self.regex:
            return StringSerializer(
                pattern=self.name,
                min_length=self.min_length,
                max_length=self.max_length,
            )
        return ConstSerializer(self.name)

    def validate(self, name: str) -> None:
        return super()._validate(instance=name)


@dataclass(frozen=True, kw_only=True, repr=False)
class Dims(Base):
    names: Sequence[Name | str] | None = field(kw_only=False, default=None)
    contains: Name | None = None
    min_contains: int | None = None
    max_contains: int | None = None
    min_items: int | None = None
    max_items: int | None = None
    title: str | None = 'Dimension names'
    description: str | None = (
        'Tuple of dimension names associated with this array.'
    )

    def __post_init__(self):
        _names = [
            name if isinstance(name, Name) else Name(name)
            for name in self.names
        ]
        object.__setattr__(self, 'names', _names)

    @cached_property
    def serializer(self) -> Serializer:
        prefix_items = (
            [name.serializer for name in self.names] if self.names else None
        )
        items = False if prefix_items else None
        contains = self.contains.serializer if self.contains else None
        return ArraySerializer(
            prefix_items=prefix_items,
            items=items,
            contains=contains,
            min_contains=self.min_contains,
            max_contains=self.max_contains,
            min_items=self.min_items,
            max_items=self.max_items,
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
        pattern_properties = {
            attr.name: attr.serializer for attr in self.attrs if attr.regex
        }
        required = [
            attr.name
            for attr in self.attrs
            if (attr.required and not attr.regex)
        ]
        return ObjectSerializer(
            properties={
                attr.name: attr.serializer
                for attr in self.attrs
                if not attr.regex
            },
            pattern_properties=pattern_properties or None,
            required=required or None,
            additional_properties=self.allow_extra_keys,
        )

    def validate(self, attrs: Mapping[str, Any]) -> None:
        return super()._validate(instance=attrs)

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Iterable, assert_never

import numpy as np
from numpy.typing import DTypeLike

from xarray_model.base import Base
from xarray_model.encoders import encode_value
from xarray_model.serializers import (
    AnySerializer,
    ArraySerializer,
    ConstSerializer,
    EnumSerializer,
    IntegerSerializer,
    NullSerializer,
    ObjectSerializer,
    Serializer,
    StringSerializer,
    TypeSerializer,
)

__all__ = [
    'Attrs',
    'Attr',
    'Chunks',
    'DType',
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

    Attributes
    ----------
    shape : int | Sequence[int]
    The expected shape of the chunk. If a single ``int`` is provided, it is
    matched against the first block size in the tuple. If a ``Sequence`` is
    provided, a full match is required.
    """

    shape: int | Sequence[int] = field(kw_only=False)

    @cached_property
    def serializer(self) -> Serializer:
        match self.shape:
            case -1:
                # `-1` is a dask wildcard meaning "use the full dimension size."
                # Expect a length 1 array of integers.
                return ArraySerializer(
                    items=IntegerSerializer(), min_items=1, max_items=1
                )
            case int():
                # A single integer is used to represent a uniform chunk size
                # where all chunks should be equal to `shape` except the last one,
                # which is free to vary.
                # TODO: (mike) this schema doesn't validate that the
                #  variable chunk is last in the array. That is currently not
                #  possible with JSON Schema (https://github.com/json-schema-org/json-schema-spec/issues/1060).
                #  We could opt for custom validation...
                return ArraySerializer(
                    # First chunk is equal to `shape`.
                    prefix_items=[ConstSerializer(self.shape)],
                    # Other chunks are `shape` or smaller...
                    items=IntegerSerializer(maximum=self.shape),
                    # but there is only one "other" chunk.
                    contains=IntegerSerializer(exclusive_maximum=self.shape),
                    max_contains=1,
                    min_contains=0,
                )
            case Sequence() if not isinstance(self.shape, str):
                # Expect a full match of the provided shape to the chunk size.
                # Can use -1 as a wildcard.
                prefix_items = [
                    IntegerSerializer()
                    if size == -1
                    else ConstSerializer(size)
                    for size in self.shape
                ]
                return ArraySerializer(
                    prefix_items=prefix_items or None,
                    items=False,
                )
            case _:
                raise ValueError(
                    f'Invalid shape: {self.shape}. '
                    'Expected int or sequence of ints.'
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

    Attributes
    ----------
    chunks : ChunksType
        If a boolean is provided, it is used to validate whether the
        ``DataArray`` is chunked or not. To validate the actual block sizes, a
        sequence of chunk sizes can be provided. A sequence of integers is used
        to represent an expected uniform chunk size. To perform an exact match,
        a sequence of integer sequences can be used.
    """

    chunks: bool | int | Sequence[int | Sequence[int]] = field(
        default=True, kw_only=False
    )

    def __post_init__(self):
        if isinstance(self.chunks, Sequence):
            _chunks = [
                chunk if isinstance(chunk, _Chunk) else _Chunk(chunk)
                for chunk in self.chunks
            ]
            object.__setattr__(self, 'chunks', _chunks)

    @cached_property
    def serializer(self) -> Serializer:
        match self.chunks:
            case True:
                return ArraySerializer(
                    items=ArraySerializer(items=IntegerSerializer())
                )
            case False:
                return NullSerializer()
            case int():
                return ArraySerializer(items=_Chunk(self.chunks).serializer)
            case Sequence() if not isinstance(self.chunks, str):
                prefix_items = [
                    chunk.serializer
                    if isinstance(chunk, _Chunk)
                    else _Chunk(chunk).serializer
                    for chunk in self.chunks
                ]
                return ArraySerializer(
                    prefix_items=prefix_items or None,
                    items=False,
                )
            case _:
                # Should be handled by `_Chunks
                assert_never(self.chunks)

    def validate(self, chunks: Iterable[Iterable[int]] | None) -> None:
        chunks = list(list(chunk) for chunk in chunks) if chunks else None
        return super()._validate(instance=chunks)


@dataclass(frozen=True, kw_only=True, repr=False)
class Shape(Base):
    """
    DataArray shape validation model

    Use this model to validate the result of ``DataArray.shape``

    Attributes
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
class DType(Base):
    """
    DataArray dtype validation model

    Use this model to validate the result of ``DataArray.dtype``

    Attributes
    ----------
    dtype : DTypeLike | None, default None
        The expected data type of the array. This can be a NumPy dtype or the
        name of a dtype. The default value of ``None`` will match any dtype.
    """

    # TODO: (mike) support numpy subdytpes
    dtype: DTypeLike | None = field(default=None, kw_only=False)

    @cached_property
    def serializer(self) -> Serializer:
        if self.dtype is None:
            return StringSerializer()
        return ConstSerializer(self.dtype)

    def validate(self, dtype: np.dtype | str) -> None:
        return super()._validate(instance=encode_value(dtype))


@dataclass(frozen=True, kw_only=True, repr=False)
class Name(Base):
    """Validation model for xarray DataArray names

    Attributes
    ----------
    name : str | Sequence[str] | None, default None
        Expected value, sequence of acceptable values, or regex pattern to
        match the name against. The default value of ``None`` will match any
        name.
    regex : bool, default False
        Flag to indicate that the ``name`` parameter should be treated as a
        regex pattern.
    min_length : int, default None
        Non-negative integer specifying the minimum length of the name.
        If a ``name`` argument is provided, this argument is silently ignored.
    max_length : int, default None
        Non-negative integer specifying the maximum length of the name.
        If a ``name`` argument is provided, this argument is silently ignored.

    Notes
    -----
    If ``name`` is a sequence or a string (not a regex pattern), the
    ``min_length`` and ``max_length`` arguments are silently ignored.

    Examples
    --------
    >>> da = xarray.DataArray(np.arange(5), dims=['x'], name='foo')
    # Validate an exact match
    >>> Name('foo').validate(da.name) # passes
    >>> Name('bar').validate(da.name) # fails
    # Validate a regex pattern
    >>> Name(r'^fo{2}$', regex=True).validate(da.name) # passes
    >>> Name(r'^fo{3}$', regex=True).validate(da.name) # fails
    # Validate a sequence of acceptable values
    >>> Name(['foo', 'bar']).validate(da.name) # passes
    >>> Name(['baz', 'qux']).validate(da.name) # fails
    # Length constraints
    >>> Name(max_length=3).validate(da.name) # passes
    >>> Name(min_length=5).validate(da.name) # fails
    """

    name: str | Sequence[str] | None = field(default=None, kw_only=False)
    regex: bool | None = None
    min_length: int | None = None
    max_length: int | None = None

    @cached_property
    def serializer(self) -> Serializer:
        match self.name:
            case None:
                return StringSerializer(
                    min_length=self.min_length,
                    max_length=self.max_length,
                )
            case str() if self.regex:
                return StringSerializer(
                    pattern=self.name,
                    min_length=self.min_length,
                    max_length=self.max_length,
                )
            case str():
                return ConstSerializer(self.name)
            case Sequence():
                return EnumSerializer(self.name)
            case _:  # pragma: no cover
                raise ValueError(
                    'Expected "name" argument to be one of str | Sequence[str] | None;'
                    ' got {self.name} which is type {type(self.name)} '
                )

    def validate(self, name: Hashable | None) -> None:
        return super()._validate(instance=name)


@dataclass(frozen=True, kw_only=True, repr=False)
class Dims(Base):
    """
    DataArray dims validation model

    Attributes
    ----------
    dims : Sequence[str | Name] | None, default None
        A sequence of expected names for the dimensions. The
        names can either be strings or instances of `Name` models for more
        complex matching. The default value of ``None`` will match any names.
    contains : str | Name | None, default None
        A string or `Name` model describing a name that must be included in the
        dimensions.
    max_size : int | None, default None
        The maximum number of dimensions.
    min_size : int | None, default None
        The minimum number of dimensions.

    See Also
    --------
    Name : DataArray name validation model
    """

    dims: Sequence[str | Name] | None = field(kw_only=False, default=None)
    contains: str | Name | None = None
    max_size: int | None = None
    min_size: int | None = None

    def _coerce_to_name(self, name: str | Name) -> Name:
        return name if isinstance(name, Name) else Name(name)

    @cached_property
    def serializer(self) -> Serializer:
        prefix_items = (
            [self._coerce_to_name(name).serializer for name in self.dims]
            if self.dims
            else None
        )
        # All prefix items are required
        min_items = (
            len(prefix_items)
            if self.min_size is None and prefix_items
            else self.min_size
        )
        # Additional items are not allowed
        items = False if prefix_items else StringSerializer()
        contains = (
            self._coerce_to_name(self.contains).serializer
            if self.contains
            else None
        )
        return ArraySerializer(
            prefix_items=prefix_items,
            items=items,
            contains=contains,
            max_items=self.max_size,
            min_items=min_items,
        )

    def validate(self, dims: tuple[Hashable, ...]) -> None:
        return super()._validate(instance=encode_value(dims))


@dataclass(frozen=True, kw_only=True, repr=False)
class Attr(Base):
    """Metadata attribute key-value pair validation model

    Attributes
    ----------
    name : str
        The expected name of the attribute. The name can be a regex pattern if
        the ``regex`` flag is set to ``True``.
    value : Any | None, default None
        The expected value of the attribute. The value can be a specific value
         for exact matching, or a Python type for more generic matching.
    regex : bool, default False
        A flag to indicate that the ``name`` parameter should be treated as a
        regex pattern.
    required: bool, default True
        A flag to indicate that the attribute is required.
    """

    name: str
    regex: bool = False
    value: Any | None = None
    required: bool = True

    @cached_property
    def serializer(self) -> Serializer:
        if isinstance(self.value, type):
            return TypeSerializer(self.value)
        if self.value is None:
            return AnySerializer()
        return ConstSerializer(self.value)

    def validate(self, attr) -> None:
        raise NotImplementedError(
            'Attr is not meant to be validated in isolation. '
            'You should compose it inside the Attrs schema.'
        )


@dataclass(frozen=True, kw_only=True, repr=False)
class Attrs(Base):
    """Metadata validation model

    Attributes
    ----------
    attrs : Iterable[Attr]
        An iterable of ``Attr`` models describing the expected metadata key-value
        pairs.
    allow_extra_keys : bool, default True
        A flag indicating whether keys not described by the ``attrs`` parameter
        are allowed/disallowed.
    """

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

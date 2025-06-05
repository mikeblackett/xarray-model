import warnings
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
    'Size',
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
                return ArraySerializer(
                    items=self.chunks.serializer
                    if isinstance(self.chunks, _Chunk)
                    else _Chunk(self.chunks).serializer
                )
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
            case _:  # pragma: no cover
                raise ValueError(
                    'Invalid argument for "chunks";'
                    ' expected one of bool | int | Sequence[int | Sequence[int]],'
                    ' got {type(self.chunks).__name__}'
                )

    def validate(self, chunks: tuple[tuple[int, ...], ...] | None) -> None:
        return super()._validate(instance=chunks)


@dataclass(frozen=True, kw_only=True, repr=False)
class Size(Base):
    """Dimension size validation model

    This model should be composed with the Shape model.

    Attributes
    ----------
    size : int | None, default None
        A non-negative integer specifying the expected size of the dimension.
        The default value of ``None`` will validate any size.
    multiple_of : int | None, default None
        Restrict the size of the dimension to multiples of a given number.
    maximum : int | None, default None
        A non-negative integer specifying the maximum size of the dimension.
    minimum : int | None, default None
        A non-negative integer specifying the minimum size of the dimension.
    """

    size: int | None = field(default=None, kw_only=False)
    multiple_of: int | None = None
    maximum: int | None = None
    minimum: int | None = None

    @cached_property
    def serializer(self) -> Serializer:
        match self.size:
            case None:
                return IntegerSerializer(
                    multiple_of=self.multiple_of,
                    maximum=self.maximum,
                    minimum=self.minimum,
                )
            case int():
                return ConstSerializer(self.size)
            case _:  # pragma: no cover
                assert_never(self.size)

    def validate(self, size: int) -> None:
        return super()._validate(instance=size)


@dataclass(frozen=True, kw_only=True, repr=False)
class Shape(Base):
    """
    DataArray shape validation model

    Use this model to validate the result of ``DataArray.shape``

    Attributes
    ----------
    shape : Sequence[int | Size] | None, default None
        Sequence of expected sizes for the dimensions of the array. The
        default value of ``None`` will match any shape.
    min_items : int | None, default None
        Minimum length of the shape sequence i.e., the number of dimensions.
    max_items : int | None, default None
        Maximum length of the shape sequence i.e., the number of dimensions.
    """

    shape: Sequence[int | Size] | None = field(default=None, kw_only=False)
    min_items: int | None = None
    max_items: int | None = None

    @cached_property
    def serializer(self) -> Serializer:
        prefix_items = None
        min_items = self.min_items

        match self.shape:
            case None:
                items = IntegerSerializer()
            case Sequence():
                items = False
                prefix_items = [
                    size.serializer
                    if isinstance(size, Size)
                    else Size(size).serializer
                    for size in self.shape
                ]
                min_items = len(prefix_items)
            case _:  # pragma: no cover
                assert_never(self.shape)

        return ArraySerializer(
            prefix_items=prefix_items,
            items=items,
            min_items=min_items,
            max_items=self.max_items,
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
        return ConstSerializer(np.dtype(self.dtype))

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
    """Metadata key-value pair validation model

    Attributes
    ----------
    key : str
        The expected name of the attribute. The name can be a regex pattern if
        the ``regex`` flag is set to ``True``.
    value : Any | None, default None
        The expected value of the attribute. The value can be a specific value
         for exact matching, or a Python type for more generic matching. The
         default value of ``None`` will match any value.
    regex : bool, default False
        A flag to indicate that the ``name`` parameter should be treated as a
        regex pattern.
    required: bool, default True
        A flag to indicate that the attribute is required. This attribute is
        silently ignored if the regex flag is set to ``True``.
    """

    key: str = field(kw_only=False)
    regex: bool = False
    value: Any | None = None
    required: bool = True

    @cached_property
    def serializer(self) -> Serializer:
        match self.value:
            case type():
                return TypeSerializer(self.value)
            case None:
                return AnySerializer()
            case str():
                return ConstSerializer(self.value)
            case Mapping():
                # TODO: (mike) Implement nested dict validation
                warnings.warn(
                    'Nested dict attribute validation is not yet implemented.'
                )
                return ObjectSerializer()
            case Sequence():
                # TODO: (mike) Implement array validation
                warnings.warn(
                    'Array attribute validation is not yet implemented.'
                )
                return ArraySerializer()
            case _:
                return ConstSerializer(encode_value(self.value))

    def validate(self, _) -> None:
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
    allow_extra_items : bool | None, default None
        A flag indicating whether items not described by the ``attrs`` parameter
        are allowed/disallowed. The default value of ``None`` is equivalent to
        ``True``.
    """

    attrs: Iterable[Attr] | None = field(default=None, kw_only=False)
    allow_extra_items: bool | None = None

    @cached_property
    def serializer(self) -> Serializer:
        if self.attrs is None:
            return ObjectSerializer(
                additional_properties=self.allow_extra_items
            )
        properties = {
            attr.key: attr.serializer for attr in self.attrs if not attr.regex
        }
        pattern_properties = {
            attr.key: attr.serializer for attr in self.attrs if attr.regex
        }
        required = [
            attr.key
            for attr in self.attrs
            if (attr.required and not attr.regex)
        ]
        return ObjectSerializer(
            properties=properties or None,
            pattern_properties=pattern_properties or None,
            required=required or None,
            additional_properties=self.allow_extra_items,
        )

    def validate(self, attrs: Mapping[str, Any]) -> None:
        return super()._validate(
            {k: encode_value(v) for k, v in attrs.items()}
        )

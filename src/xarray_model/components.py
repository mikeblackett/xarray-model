import re
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from xarray_model.base import (
    BaseModel,
    ConversionError,
    DeserializationError,
    SerializationError,
)
from xarray_model.types import (
    AttrType,
    AttrsType,
    ChunkType,
    ChunksType,
    DimsType,
    DTypeLike,
    NameType,
    SizesType,
)


@dataclass(frozen=True)
class NameModel(BaseModel):
    """
    A schema class to validate the name of an array.

    Attributes
    ----------
    name : NameType
        The expected type of the array's name. Can be a string, a regex pattern,
        or a sequence of acceptable strings.
    """

    _title = 'Name'
    _description = 'The name of this array.'

    name: NameType

    def serialize(self) -> dict[str, Any]:
        match self.name:
            case str():
                schema = {'const': self.name}
            case re.Pattern():
                schema = {'pattern': self.name.pattern}
            case Sequence():
                schema = {'enum': list(self.name)}
            case _:
                raise SerializationError
        return {'type': 'string'} | schema

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> Self:
        match data:
            case {'const': str()}:
                name = data['const']
            case {'pattern': str()}:
                name = re.compile(data['pattern'])
            case {'enum': list()}:
                name = list(data['enum'])
            case _:
                raise DeserializationError
        return cls(name=name)

    @classmethod
    def _convert(cls, data: NameType) -> Self:
        match data:
            case str():
                name = data
            case re.Pattern():
                name = data
            case Sequence():
                name = tuple(data)
            case _:
                raise ConversionError
        return cls(name=name)

    def validate(self, name: Hashable) -> None:
        return super()._validate(instance=name)


@dataclass(frozen=True)
class DimsModel(BaseModel):
    """
    A schema class to validate the dimensions of an array.

    Attributes
    ----------
    dims : DimsType
        An iterable representing the expected names of the array's dimensions.
        The value ``None`` can be used as a wildcard.
    """

    # TODO: (mike) Support dimension name pattern/enum matching?

    _title = 'Dims'
    _description = 'Tuple of dimension names associated with this array.'

    dims: DimsType

    def serialize(self) -> dict[str, Any]:
        return {
            'type': 'array',
            'prefixItems': [
                {'const': dim} if dim else {'type': 'string'}
                for dim in self.dims
            ],
        }

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> Self:
        try:
            dims = [item.get('const') for item in data['prefixItems']]
        except KeyError as error:
            raise DeserializationError from error
        return cls(dims=dims)

    @classmethod
    def _convert(cls, data: DimsType) -> Self:
        if not isinstance(data, Sequence):
            raise ConversionError
        return cls(dims=data)

    def validate(self, dims: tuple[Hashable, ...]) -> None:
        return super()._validate(instance=list(dims))


@dataclass(frozen=True)
class DTypeModel(BaseModel):
    """
    A schema class to validate the dtype of an array.

    Attributes
    ----------
    dtype : DTypeLike
        The expected dtype of the array's data. Can be a numpy dtype or a value
        that is coercible by ``numpy.dtype``.

    Notes
    -----
    Currently only supports exact matches. Support for sub-dtype matching is planned.
    """

    # TODO: (mike) Support sub-dtypes

    _title = 'DType'
    _description = 'Data-type of the array’s elements.'

    dtype: DTypeLike

    def serialize(self) -> dict[str, Any]:
        return {
            'const': np.dtype(self.dtype).name,
        }

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> Self:
        return cls(dtype=data['dtype'])

    def validate(self, dtype: np.dtype) -> None:
        return super()._validate(instance=dtype.name)

    @classmethod
    def _convert(cls, data: DTypeLike) -> Self:
        match data:
            case np.dtype():
                dtype = data
            case str():
                dtype = np.dtype(data)
            case _:
                raise ConversionError
        return cls(dtype=dtype)


@dataclass(frozen=True)
class SizesModel(BaseModel):
    _title = 'Sizes'
    _description = 'Ordered mapping from dimension names to lengths.'

    sizes: SizesType
    allow_extra_keys: bool = False
    require_all_keys: bool = True

    def serialize(self) -> dict[str, Any]:
        schema = {
            'type': 'object',
            'properties': {
                key: {'const': value} if value else {'type': 'integer'}
                for key, value in self.sizes.items()
            },
            'additionalProperties': self.allow_extra_keys,
        }
        if self.require_all_keys:
            schema['required'] = list(self.sizes.keys())
        return schema

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> Self:
        try:
            sizes = {
                key: value.get('const')
                for key, value in data['properties'].items()
            }
            allow_extra_keys = data.get('additionalProperties', False)
            require_all_keys = data.get('required', True)
        except KeyError as error:
            raise DeserializationError from error
        return cls(
            sizes=sizes,
            allow_extra_keys=allow_extra_keys,
            require_all_keys=require_all_keys,
        )

    def validate(self, sizes: Mapping[Hashable, int]) -> None:
        return super()._validate(instance=dict(sizes))

    @classmethod
    def _convert(cls, data: SizesType) -> Self:
        if not isinstance(data, Mapping):
            raise ConversionError
        return cls(sizes=data)


@dataclass(frozen=True)
class _AttrModel(BaseModel):
    _description = 'Arbitrary metadata value'
    _title = 'Attr'

    attr: AttrType

    def serialize(self) -> dict[str, Any]:
        if self.attr is None:
            return {}
        if 'type' in self.attr:
            return {'type': self.attr['type']}
        if 'value' in self.attr:
            return {'const': self.attr['value']}
        raise SerializationError

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> Self:
        if 'type' in data:
            return cls(attr={'type': data['type']})
        if 'const' in data:
            return cls(attr={'const': data['const']})
        return cls(attr=None)

    def validate(self, attr: Mapping[str, Any]) -> None:
        return super()._validate(instance=attr)

    @classmethod
    def _convert(cls, data: Mapping[str, Any] | None) -> Self:
        if data is None:
            return cls(attr=None)
        if 'type' in data:
            return cls(attr={'type': data['type']})
        if 'value' in data:
            return cls(attr={'value': data['value']})
        raise ConversionError


@dataclass(frozen=True)
class AttrsModel(BaseModel):
    _title = 'Attrs'
    _description = 'Dictionary storing arbitrary metadata with this array.'

    attrs: Mapping[str, AttrType]
    allow_extra_keys: bool = True
    require_all_keys: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'attrs',
            {
                key: _AttrModel.convert(attr)
                for key, attr in self.attrs.items()
            },
        )

    def serialize(self) -> dict[str, Any]:
        schema = {
            'type': 'object',
            'properties': {
                key: attr.serialize()  # type: ignore
                for key, attr in self.attrs.items()
            },
            'additionalProperties': self.allow_extra_keys,
        }
        if self.require_all_keys:
            schema['required'] = list(self.attrs.keys())
        return schema

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> Self:
        try:
            attrs = {
                key: _AttrModel.convert(attr).serialize()
                for key, attr in data['properties'].items()
            }
        except KeyError as error:
            raise DeserializationError from error
        allow_extra_keys = data.get('allow_extra_keys', True)
        require_all_keys = data.get('required', True)
        return cls(
            attrs=attrs,
            allow_extra_keys=allow_extra_keys,
            require_all_keys=require_all_keys,
        )

    def validate(self, attrs: Mapping[str, Any]) -> None:
        return super()._validate(instance=attrs)

    @classmethod
    def _convert(cls, data: AttrsType) -> Self:
        if not isinstance(data, Mapping):
            raise ConversionError
        return cls(attrs=data)


@dataclass(frozen=True)
class _ChunkModel(BaseModel):
    _title = 'Chunk'
    _description = 'Chunking strategy for a single dimension.'
    _type = 'array'

    chunk: ChunkType

    def serialize(self) -> dict[str, Any]:
        match self.chunk:
            case bool():
                schema = {'minItems': 1} if self.chunk else {'maxItems': 0}
            case int():
                schema = {'contains': {'const': self.chunk}}
            case Sequence():
                schema = {
                    'prefixItems': [
                        {'type': 'integer', 'const': item}
                        for item in self.chunk
                    ],
                }
            case _:
                raise SerializationError
        return {'type': self._type} | schema

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> Self:
        if 'prefixItems' in data:
            return cls(
                chunk=[int(item['const']) for item in data['prefixItems']]
            )
        if 'contains' in data:
            return cls(chunk=int(data['contains']['const']))
        raise DeserializationError

    def validate(self, value: Sequence[int]) -> None:
        return super()._validate(instance=value)

    @classmethod
    def _convert(cls, data: ChunkType) -> Self:
        return cls(chunk=data)


@dataclass(frozen=True)
class ChunksModel(BaseModel):
    _title = 'Chunk sizes'
    _description = 'Mapping from dimension names to block lengths for this dataarray’s data.'

    chunks: ChunksType
    allow_extra_keys: bool = True
    require_all_keys: bool = True

    def serialize(self) -> dict[str, Any]:
        match self.chunks:
            case bool():
                schema = (
                    {'minProperties': 1}
                    if self.chunks
                    else {'maxProperties': 0}
                )
            case Mapping():
                schema = {
                    'additionalProperties': self.allow_extra_keys,
                    'properties': {
                        key: _ChunkModel(chunk).serialize()
                        for key, chunk in self.chunks.items()
                    },
                }
            case _:
                raise SerializationError
        return {'type': 'object'} | schema

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> Self:
        match data:
            case {'minProperties': 1}:
                chunks = True
            case {'maxProperties': 0}:
                chunks = False
            case {'properties': Mapping()}:
                chunks = {
                    key: _ChunkModel.convert(chunk).serialize()
                    for key, chunk in data['properties'].items()
                }
            case _:
                raise DeserializationError
        allow_extra_keys = data.get('additionalProperties', True)
        return cls(
            chunks=chunks,
            allow_extra_keys=allow_extra_keys,
        )

    def validate(self, chunksizes: Mapping[Any, tuple[int, ...]]) -> None:
        return super()._validate(instance=dict(chunksizes))

    @classmethod
    def _convert(cls, data: ChunksType) -> Self:
        if not isinstance(data, Mapping):
            raise ConversionError
        return cls(chunks=data)

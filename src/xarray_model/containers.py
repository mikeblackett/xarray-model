from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from xarray_model import Chunks
from xarray_model.base import Base
from xarray_model.components import (
    Attrs,
    Datatype,
    Dims,
    Name,
    Shape,
)
from xarray_model.serializers import Serializer, ObjectSerializer


@dataclass(frozen=True, kw_only=True)
class CoordsModel(Base):
    title = 'Coords'
    description = (
        'Mapping of DataArray objects corresponding to coordinate variables.'
    )
    coords: Mapping[str, 'DataArrayModel'] = field(kw_only=False)
    require_all_keys: bool = True
    allow_extra_keys: bool = False

    @cached_property
    def serializer(self) -> Serializer:
        return ObjectSerializer(
            title=self.title,
            description=self.description,
            properties={
                name: data_array.serializer
                for name, data_array in self.coords.items()
            },
            required=list(self.coords.keys())
            if self.require_all_keys
            else None,
            additional_properties=self.allow_extra_keys,
        )

    def validate(self, coords: Mapping[str, Any]) -> None:
        instance = preprocess_data_dict(coords)
        return super()._validate(instance=instance)


@dataclass(frozen=True, kw_only=True)
class DataArrayModel(Base):
    title: str | None = 'xarray DataArray'
    description: str | None = (
        'N-dimensional array with labeled coordinates and dimensions.'
    )
    attrs: Attrs | None = None
    chunks: Chunks | None = None
    coords: CoordsModel | None = None
    dims: Dims | None = None
    dtype: Datatype | None = None
    name: Name | None = None
    shape: Shape | None = None

    @cached_property
    def serializer(self) -> Serializer:
        return ObjectSerializer(
            title=self.title,
            description=self.description,
            properties={
                'attrs': self.attrs.serializer if self.attrs else None,
                'chunks': self.chunks.serializer if self.chunks else None,
                'coords': self.coords.serializer if self.coords else None,
                'dims': self.dims.serializer if self.dims else None,
                'dtype': self.dtype.serializer if self.dtype else None,
                'name': self.name.serializer if self.name else None,
                'shape': self.shape.serializer if self.shape else None,
            },
            additional_properties=True,  # for extra items from DataArray.to_dict()
        )

    def validate(self, data_array: Any) -> None:
        instance = data_array.to_dict(data=False)
        instance = preprocess_data_dict(instance)
        instance['chunks'] = (
            list(list(chunk) for chunk in data_array.chunks)
            if data_array.chunks
            else None
        )
        return super()._validate(instance=instance)


@dataclass(frozen=True, kw_only=True)
class DataVarsModel(Base):
    data_vars: Mapping[str, 'DataArrayModel'] = field(kw_only=False)
    title = 'DataVars'
    description = (
        'Dictionary of DataArray objects corresponding to data variables'
    )
    require_all_keys: bool = True
    allow_extra_keys: bool = False

    @cached_property
    def serializer(self) -> Serializer:
        return ObjectSerializer(
            title=self.title,
            description=self.description,
            properties={
                name: data_array.serializer
                for name, data_array in self.data_vars.items()
            },
            required=list(self.data_vars.keys())
            if self.require_all_keys
            else None,
            additional_properties=self.allow_extra_keys,
        )

    def validate(self, data_vars: Mapping[str, Any]) -> None:
        instance = preprocess_data_dict(data_vars)
        return super()._validate(instance=instance)


@dataclass(frozen=True, kw_only=True)
class DatasetModel(Base):
    title = 'Dataset'
    description = 'A multi-dimensional, in memory, array database.'

    coords: CoordsModel | None = None
    data_vars: DataVarsModel | None = None
    attrs: Attrs | None = None

    @cached_property
    def serializer(self) -> Serializer:
        return ObjectSerializer(
            title=self.title,
            description=self.description,
            properties={
                'coords': self.coords.serializer if self.coords else None,
                'data_vars': self.data_vars.serializer
                if self.data_vars
                else None,
                'attrs': self.attrs.serializer if self.attrs else None,
            },
            additional_properties=True,  # for extra items from Dataset.to_dict()
        )

    def validate(self, data_array: Any) -> None:
        instance = data_array.to_dict(data=False)
        instance = preprocess_data_dict(instance)
        return super()._validate(instance=instance)


def preprocess_data_dict(data: dict) -> dict:
    # Convert tuples to lists for JSON Schema validation
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = preprocess_data_dict(v)
        else:
            result[k] = list(v) if isinstance(v, tuple) else v
    return result

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import xarray as xr

from xarray_model import Chunks
from xarray_model.base import Base
from xarray_model.components import (
    Attrs,
    Dims,
    DType,
    Name,
    Shape,
)
from xarray_model.serializers import (
    ObjectSerializer,
    Serializer,
)


@dataclass(frozen=True, repr=False, kw_only=True)
class DataArrayModel(Base):
    attrs: Attrs | None = None
    chunks: Chunks | None = None
    coords: 'CoordsModel | None' = None
    description: str | None = None
    dims: Dims | None = None
    dtype: DType | None = None
    name: Name | None = None
    shape: Shape | None = None
    title: str | None = None

    @cached_property
    def serializer(self) -> Serializer:
        return ObjectSerializer(
            title=self.title,
            description=self.description,
            properties={
                k: getattr(self, k).serializer
                for k in (
                    'attrs',
                    'chunks',
                    'coords',
                    'dims',
                    'dtype',
                    'name',
                    'shape',
                )
                if getattr(self, k) is not None
            },
        )

    def validate(self, data_array: xr.DataArray) -> None:
        instance = data_array.to_dict(data=False)
        return super()._validate(instance=instance)


@dataclass(frozen=True, repr=False, kw_only=True)
class CoordsModel(Base):
    coords: Mapping[str, DataArrayModel] = field(kw_only=False)
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
            }
            or None,
            required=list(self.coords.keys())
            if self.require_all_keys
            else None,
            additional_properties=self.allow_extra_keys,
        )

    def validate(self, coords: Mapping[str, Any]) -> None:
        return super()._validate(instance=coords)


@dataclass(frozen=True, repr=False, kw_only=True)
class DataVarsModel(Base):
    data_vars: Mapping[str, 'DataArrayModel'] = field(kw_only=False)
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
            }
            or None,
            required=list(self.data_vars.keys())
            if self.require_all_keys
            else None,
            additional_properties=self.allow_extra_keys,
        )

    def validate(self, data_vars: Mapping[str, Any]) -> None:
        return super()._validate(instance=data_vars)


@dataclass(frozen=True, repr=False, kw_only=True)
class DatasetModel(Base):
    coords: CoordsModel | None = None
    data_vars: DataVarsModel | None = None
    attrs: Attrs | None = None

    @cached_property
    def serializer(self) -> Serializer:
        return ObjectSerializer(
            title=self.title,
            description=self.description,
            properties={
                k: getattr(self, k).serializer
                for k in (
                    'attrs',
                    'coords',
                    'data_vars',
                )
                if getattr(self, k) is not None
            },
            additional_properties=True,  # for extra items from Dataset.to_dict()
        )

    def validate(self, data_array: Any) -> None:
        instance = data_array.to_dict(data=False)
        return super()._validate(instance=instance)

from collections.abc import Mapping
import json
from abc import abstractmethod
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Any, ClassVar, Self

import jsonschema

from xarray_model.components import (
    Attrs,
    Component,
    Dims,
    Datatype,
    Name,
    Shape,
)
from xarray_model.types import JSONDataType

DIALECT = 'https://json-schema.org/draft/2020-12/schema'


@dataclass(frozen=True, kw_only=True)
class Container(Component):
    """Base class for JSON schema xarray models."""

    # Class variables
    _dialect: ClassVar[str] = DIALECT
    _type: ClassVar[JSONDataType]
    _validator: ClassVar = jsonschema.Draft7Validator

    @cached_property
    def validator(self):
        """The validator for this schema."""
        return jsonschema.Draft7Validator(schema=self.schema)

    def _validate(self, instance: Any) -> None:
        return self.validator.validate(instance=instance)

    @abstractmethod
    def validate(self, *args, **kwargs) -> None:
        """Validate an object against this schema."""
        ...

    def to_json(self) -> str:
        """Return the schema as a JSON string."""
        return json.dumps(self.schema)

    @classmethod
    def from_json(cls, schema: str) -> Self:
        """Instantiate this model from a JSON string."""
        return cls.from_schema(**json.loads(schema))


@dataclass(frozen=True, kw_only=True)
class DataArraySchema(Container):
    title: str | None = 'xarray DataArray'
    description: str | None = (
        'N-dimensional array with labeled coordinates and dimensions.'
    )
    dims: Dims | None = None
    attrs: Attrs | None = None
    dtype: Datatype | None = None
    name: Name | None = None
    shape: Shape | None = None

    @cached_property
    def schema(self) -> dict[str, Any]:
        schema = {
            'title': self.title,
            'description': self.description,
            'type': 'object',
            'additionalProperties': True,
        }
        properties = {}
        if self.dims is not None:
            properties.update(dims=self.dims.schema)
        if self.attrs is not None:
            properties.update(attrs=self.attrs.schema)
        if self.dtype is not None:
            properties.update(dtype=self.dtype.schema)
        if self.shape is not None:
            properties.update(shape=self.shape.schema)
        if self.name is not None:
            properties.update(name=self.name.schema)
        return schema | {'properties': properties}

    @classmethod
    def from_schema(cls, data: Mapping[str, Any]) -> Self:
        kwargs = {f.name: getattr(data, f.name) for f in fields(cls)}
        return cls(**kwargs)

    def validate(self, data_array: Any) -> None:
        return super()._validate(instance=data_array.to_dict(data=False))

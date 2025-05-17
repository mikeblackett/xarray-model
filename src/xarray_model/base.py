import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Self

import jsonschema

from xarray_model.types import JSONDataType

DIALECT = 'http://json-schema.org/draft-07/schema'


class ModelError(Exception):
    """Raised when a model fails validation."""

    ...


class ConversionError(ModelError):
    """Raised when conversion fails."""


class SerializationError(ModelError):
    """Raised when serialization fails."""

    ...


class DeserializationError(ModelError):
    """Raised when deserialization fails."""

    ...


@dataclass(frozen=True)
class BaseModel(ABC):
    """Base class for JSON schema xarray models."""

    # Defaults
    _dialect: ClassVar[str] = DIALECT
    _type: ClassVar[JSONDataType]
    _validator = jsonschema.Draft7Validator
    # Annotations
    _title: ClassVar[str] = ''
    _description: ClassVar[str] = ''
    _default: ClassVar[str] = ''

    @property
    def schema(self) -> dict[str, Any]:
        """The JSON schema for this schema."""
        _schema = {'$schema': self._dialect}
        if self._title:
            _schema |= {'title': self._title}
        if self._description:
            _schema |= {'description': self._description}
        if self._default:
            _schema |= {'default': self._default}
        return _schema | self.serialize()

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

    @abstractmethod
    def serialize(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the schema constraints."""
        ...

    @classmethod
    @abstractmethod
    def deserialize(cls, data: Mapping[str, Any]) -> Self:
        """Instantiate this schema from a JSON-serializable representation of schema constraints."""
        ...

    @classmethod
    @abstractmethod
    def _convert(cls, data: Any) -> Self: ...

    @classmethod
    def convert(cls, data: Any) -> Self:
        """Attempt to convert ``data`` to this schema."""
        if isinstance(data, cls):
            return data
        return cls._convert(data)

    def to_json_schema(self) -> str:
        """Return the schema as a JSON string."""
        return json.dumps(self.serialize())

    @classmethod
    def from_json_schema(cls, schema: str) -> Self:
        """Instantiate this model from a JSON string."""
        return cls.deserialize(json.loads(schema))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Convert a dictionary to this schema."""
        # Subclasses should normally override this method...
        return cls(**data)

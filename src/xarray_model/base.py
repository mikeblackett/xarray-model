import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Any, ClassVar, Self

import jsonschema

from xarray_model.serializers import Serializer
from xarray_model.types import JSONDataType

DIALECT = 'https://json-schema.org/draft/2020-12/schema'


class ModelError(Exception):
    """Base exception for xarray model errors."""


class InvalidArgumentError(ModelError):
    """Error raised when an invalid argument is encountered."""

    pass


class NotYetImplementedError(ModelError):
    """
    Error raised when a planned feature is not yet implemented.

    This is **different** from ``NotImplementedError``, which is used when a
    method is not implemented in a base class.
    """

    pass


@dataclass(frozen=True, kw_only=True, repr=False)
class Base(ABC):
    """Base class for xarray models."""

    _dialect: ClassVar[str] = DIALECT
    _type: ClassVar[JSONDataType]
    _validator: ClassVar = jsonschema.Draft202012Validator

    title: str | None = None
    description: str | None = None

    @cached_property
    @abstractmethod
    def serializer(self) -> Serializer: ...

    @cached_property
    def schema(self) -> dict[str, Any]:
        schema = self.serializer.serialize()
        self._validator.check_schema(schema=schema)
        return schema

    @cached_property
    def validator(self):
        """The validator for this schema."""
        return self._validator(schema=self.schema)

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

    def __repr__(self):
        # Override repr to show only non-default arguments.
        args = [
            (f.name, getattr(self, f.name))
            for f in fields(self)
            if getattr(self, f.name) != f.default
        ]
        args_string = ''.join(f'{name}={value}' for name, value in args)
        return f'{self.__class__.__name__}({args_string})'

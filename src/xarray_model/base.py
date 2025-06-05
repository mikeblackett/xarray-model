import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Any, ClassVar


from xarray_model.serializers import Serializer
from xarray_model.validators import XarrayModelValidator


DIALECT = 'https://json-schema.org/draft/2020-12/schema'


class ModelError(Exception):
    """Base exception for xarray model errors."""


class NotYetImplementedError(ModelError):
    """Error raised when a planned feature is not yet implemented."""


@dataclass(frozen=True, kw_only=True, repr=False)
class Base(ABC):
    """Base class for xarray validations models."""

    _dialect: ClassVar[str] = DIALECT
    """The version of JSON Schema used by this model."""
    _validator: ClassVar = XarrayModelValidator
    """The JSON Schema validator class used by this model."""

    title: str | None = None
    description: str | None = None

    @cached_property
    @abstractmethod
    def serializer(self) -> Serializer:
        """The serializer for this schema."""
        raise NotImplementedError

    @cached_property
    def schema(self) -> dict[str, Any]:
        """The JSON Schema schema for this model."""
        schema = self.serializer.serialize()
        self._validator.check_schema(schema=schema)
        return schema

    @cached_property
    def validator(self):
        """The validator for this schema."""
        return self._validator(schema=self.schema)

    def _validate(self, instance: Any) -> None:
        """Validate an instance against this schema.

        Subclasses should normally call this method in their `validate` method.
        """
        return self.validator.validate(instance=instance)

    @abstractmethod
    def validate(self, *args, **kwargs) -> None:
        """Validate an instance against this schema.

        Subclasses should implement this method and perform any necessary
        preprocessing of arguments before passing to `_validate`.
        """
        ...

    def to_json(self) -> str:
        """Return the schema as a JSON string."""
        return json.dumps(self.schema)

    # @classmethod
    # def from_json(cls, schema: str) -> Self:
    #     """Instantiate this model from a JSON string."""
    #     # TODO: implement `Base.from_schema`
    #     return cls.from_schema(**json.loads(schema))

    # @classmethod
    # def from_xarray(cls, *args, **kwargs) -> Self:
    #     # TODO: implement `Base.from_xarray`
    #     ...

    def __repr__(self):
        # Show only non-default arguments...
        args = [
            (f.name, getattr(self, f.name))
            for f in fields(self)
            if getattr(self, f.name) != f.default
        ]
        args_string = ''.join(f'{name}={value}' for name, value in args)
        return f'{self.__class__.__name__}({args_string})'

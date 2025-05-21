from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from re import Pattern
from typing import Any, Type

from xarray_model.encoding import decode_type, encode_type, encode_value


class SerializationError(Exception):
    """Raised when serialization fails."""

    ...


class DeserializationError(Exception):
    """Raised when deserialization fails."""

    ...


@dataclass(frozen=True)
class Serializer[T](ABC):
    @abstractmethod
    def serialize(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def deserialize(cls, obj: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True)
class StringSerializer(Serializer):
    pattern: str | Pattern[str] | None = None

    def serialize(self) -> dict[str, Any]:
        print(self._serialize())
        result = {'type': 'string'}
        if self.pattern is None:
            return result
        try:
            result |= {'pattern': encode_value(self.pattern)}
        except TypeError as error:
            raise SerializationError from error

    @classmethod
    def deserialize(cls, obj: Mapping[str, Any]) -> dict[str, Any]:
        match obj:
            case {'pattern': str()}:
                return {'pattern': kwargs['pattern']}
        raise DeserializationError


@dataclass(frozen=True)
class AnnotationSerializer(Serializer):
    title: str | None
    description: str | None

    def serialize(self) -> dict[str, Any]:
        result = {}
        if self.title is not None:
            result |= {'title': self.title}
        if self.description is not None:
            result |= {'description': self.description}
        return result


@dataclass(frozen=True)
class ArraySerializer(Serializer):
    items: Serializer | None = None
    prefix_items: Iterable[Serializer] | None = None
    min_items: int | None = None
    max_items: int | None = None

    def serialize(self) -> dict[str, Any]:
        result = {'type': 'array'}
        if self.min_items is not None:
            result |= {'minItems': self.min_items}
        if self.max_items is not None:
            result |= {'maxItems': self.max_items}
        if self.prefix_items:
            result |= {
                'prefixItems': [item.serialize() for item in self.prefix_items]
            }
        if self.items:
            result |= {'items': self.items.serialize()}
        return result

    @classmethod
    def deserialize(cls, obj: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True)
class IntegerSerializer(Serializer):
    minimum: int | None = None
    maximum: int | None = None

    def serialize(self) -> dict[str, Any]:
        result = {'type': 'integer'}
        if self.minimum is not None:
            result |= {'minimum': self.minimum}
        if self.maximum is not None:
            result |= {'maximum': self.maximum}
        return result

    @classmethod
    def deserialize(cls, obj: Mapping[str, Any]) -> dict[str, Any]:
        try:
            assert obj['type'] == 'integer'
        except AssertionError as error:
            raise DeserializationError(error) from error
        match obj:
            case {'minimum': int() as minimum}:
                return {'minimum': minimum}
            case {'maximum': int() as maximum}:
                return {'maximum': maximum}
        raise DeserializationError


@dataclass(frozen=True)
class TypeSerializer(Serializer):
    type_: Type

    def serialize(self) -> dict[str, Any]:
        try:
            result = {'type': encode_type(self.type_)}
        except TypeError as error:
            raise SerializationError(error) from error
        return result

    @classmethod
    def deserialize(cls, obj: Mapping[str, Any]) -> dict[str, Any]:
        match obj:
            case {'type': str() as type_}:
                return {'type': decode_type(type_)}
        raise DeserializationError


@dataclass(frozen=True)
class ConstSerializer[T](Serializer):
    const: T

    def serialize(self) -> dict[str, Any]:
        try:
            result = {'const': encode_value(self.const)}
        except TypeError as error:
            raise SerializationError from error
        return result

    @classmethod
    def deserialize(cls, obj: Mapping[str, Any]) -> T:
        try:
            result = obj['const']
        except KeyError as error:
            raise DeserializationError from error
        return result


def _camel_case_to_snake_case(string: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', '_', string).lower()


def _snake_case_to_camel_case(string: str) -> str:
    string = ''.join(word.title() for word in string.split('_'))
    return string[0].lower() + string[1:]

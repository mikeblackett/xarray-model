from functools import singledispatch
import re
from collections.abc import Mapping, Sequence
from types import NoneType
from typing import Any, Type

import numpy as np

DECODE_TYPES = {
    'array': list,
    'bool': bool,
    'float': float,
    'integer': int,
    'null': None,
    'object': dict,
    'string': str,
}

ENCODE_KEYWORDS = {
    'anchor': '$anchor',
    'comment': '$comment',
    'id': '$id',
    'ref': '$ref',
    'schema': '$schema',
    'vocabulary': '$vocabulary',
}

DECODE_KEYWORDS = {
    '$anchor': 'anchor',
    '$comment': 'comment',
    '$id': 'id',
    '$ref': 'ref',
    '$schema': 'schema',
    '$vocabulary': 'vocabulary',
}


def encode_keyword(keyword: str):
    """Encode a Python-formatted keyword to JSON Schema."""
    result = _snake_case_to_camel_case(keyword)
    return ENCODE_KEYWORDS.get(result, result)


@singledispatch
def encode_value(value: Any) -> Any:
    return value  # Default: identity


@encode_value.register
def _(value: str) -> str:
    return value


@encode_value.register
def _(value: Sequence) -> list:
    return list(value)


@encode_value.register
def _(value: set) -> list:
    return list(value)


@encode_value.register
def _(value: np.dtype) -> str:
    return value.name


@encode_value.register
def _(value: re.Pattern) -> str:
    return value.pattern


@encode_value.register
def _(value: type) -> str:
    return _encode_type(value)


@encode_value.register
def _(value: np.ndarray) -> list:
    return value.tolist()


def _encode_type(type_: Type) -> str:
    if issubclass(type_, str):
        return 'string'
    elif issubclass(type_, bool):
        return 'boolean'
    elif issubclass(type_, int):
        return 'integer'
    elif issubclass(type_, float):
        return 'number'
    elif issubclass(type_, Mapping):
        return 'object'
    elif issubclass(type_, Sequence) and not issubclass(type_, str):
        return 'array'
    elif issubclass(type_, np.ndarray):
        return 'array'
    elif issubclass(type_, NoneType):
        return 'null'
    raise TypeError(
        f'Error encoding python type {type_!r} as JSON Schema type.'
    )


def _snake_case_to_camel_case(string: str) -> str:
    if '_' not in string:
        return string
    string = ''.join(word.title() for word in string.split('_'))
    return string[0].lower() + string[1:]


def _camel_case_to_snake_case(string: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', '_', string).lower()


# def decode_type(type_: str) -> Type:
#     return DECODE_TYPES.get(type_, type_)
#
#
# def decode_keyword(keyword: str):
#     """Decode a JSON-Schema-formatted keyword to Python."""
#     result = _camel_case_to_snake_case(keyword)
#     return DECODE_KEYWORDS.get(result, result)

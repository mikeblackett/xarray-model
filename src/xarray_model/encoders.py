from functools import singledispatch
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Type

import numpy as np

DECODE_TYPES = {
    'string': str,
    'integer': int,
    'float': float,
    'bool': bool,
    'object': dict,
    'array': list,
}

PYTHON_TO_JSON_KEYWORDS = {
    'anchor': '$anchor',
    'comment': '$comment',
    'id': '$id',
    'ref': '$ref',
    'schema': '$schema',
    'vocabulary': '$vocabulary',
}

JSON_TO_PYTHON_KEYWORDS = {
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
    return PYTHON_TO_JSON_KEYWORDS.get(result, result)


def decode_keyword(keyword: str):
    """Decode a JSON-Schema-formatted keyword to Python."""
    result = _camel_case_to_snake_case(keyword)
    return JSON_TO_PYTHON_KEYWORDS.get(result, result)


@singledispatch
def encode_value(value: Any) -> Any:
    if isinstance(value, Iterable) and not isinstance(value, str):
        return list(value)
    return value  # Default: identity


@encode_value.register
def _(value: np.dtype) -> str:
    return value.name


@encode_value.register
def _(value: re.Pattern) -> str:
    return value.pattern


def encode_type(type_: Type) -> str:
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
    raise TypeError(
        f'Error encoding python type {type_!r} as JSON Schema type.'
    )


def decode_type(type_: str) -> Type:
    return DECODE_TYPES.get(type_, type_)


def _camel_case_to_snake_case(string: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', '_', string).lower()


def _snake_case_to_camel_case(string: str) -> str:
    if '_' not in string:
        return string
    string = ''.join(word.title() for word in string.split('_'))
    return string[0].lower() + string[1:]

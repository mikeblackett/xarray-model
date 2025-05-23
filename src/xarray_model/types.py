from collections.abc import Sequence
from typing import Literal

from numpy.typing import DTypeLike

__all__ = ['ChunksType', 'JSONDataType', 'ShapeType', 'DTypeLike']

type JSONDataType = Literal[
    'array',
    'boolean',
    'const',
    'integer',
    'null',
    'number',
    'object',
    'pattern',
    'string',
]

type ChunksType = bool | Sequence[Sequence[int] | int]
type ShapeType = Sequence[int]
type NameType = str
type DimsType = Sequence[NameType | str]

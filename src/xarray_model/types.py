from typing import Literal

__all__ = ['JSONDataType']

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

type ChunksType = tuple[tuple[int, ...], ...] | None

"""
Custom JSON Schema validators for xarray-model.
"""

from jsonschema import validators, Draft202012Validator


def is_array_like(checker, instance):
    return Draft202012Validator.TYPE_CHECKER.is_type(
        instance, 'array'
    ) or isinstance(instance, tuple)


XarrayModelValidator = validators.extend(
    validator=Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
        'array', is_array_like
    ),
)
"""
A JSON Schema validator that allows tuples and arrays to be used as
instances of the 'array' type.

This facilitates the validation of xarray `dims` which are have type `tuple[Hashable]`
"""

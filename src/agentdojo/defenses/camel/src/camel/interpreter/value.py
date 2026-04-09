# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import copy
import dataclasses
import types
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, MutableSequence, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeGuard, TypeVar, runtime_checkable

import pydantic
from typing_extensions import Self

from camel.capabilities import Capabilities, readers, sources
from camel.conditional_cache import conditional_lru_cache
from camel.interpreter import op_protocols

if TYPE_CHECKING:
    from camel.interpreter import namespace as ns


class FunctionCallWithSideEffectError(Exception): ...


class PythonComparable(Protocol):
    def __lt__(self, other: Self, /) -> bool: ...
    def __gt__(self, other: Self, /) -> bool: ...


def is_value(value: object) -> TypeGuard["CaMeLValue"]:
    return isinstance(value, CaMeLValue)


_T = TypeVar("_T", covariant=False)


@runtime_checkable
class CaMeLValue(Generic[_T], Protocol):
    _python_value: _T
    _metadata: Capabilities
    _dependencies: tuple["CaMeLValue", ...]
    is_builtin: bool = False

    def __repr__(self) -> str:
        return self._repr_helper(indent_level=0)

    def _repr_helper(self, indent_level: int) -> str:
        indent = "  " * indent_level
        next_indent = "  " * (indent_level + 1)

        metadata_str = repr(self._metadata)

        return f"""{type(self).__name__}(
  {next_indent}value={self._python_value!r},
  {next_indent}metadata={metadata_str},
  {next_indent}dependencies=...
  {indent})"""

    def get_dependencies(
        self, visited_objects: frozenset[int] = frozenset()
    ) -> tuple[tuple["CaMeLValue", ...], frozenset[int]]:
        return self._dependencies, frozenset({id(self)})

    @property
    def metadata(self) -> Capabilities:
        return self._metadata

    def __eq__(self, other) -> bool:
        if not is_value(other):
            return False
        return (
            type(self) == type(other)
            and self._python_value == other._python_value
            and self._metadata == other._metadata
            and self._dependencies == other._dependencies
        )

    def new_with_python_value(self, value: _T) -> Self:
        new_self = copy.copy(self)
        new_self._python_value = value
        return new_self

    def new_with_dependencies(self, dependencies: tuple["CaMeLValue", ...]) -> Self:
        new_self = copy.copy(self)
        new_self._dependencies = self._dependencies + dependencies
        return new_self

    def new_with_metadata(self, metadata: Capabilities) -> Self:
        new_self = copy.copy(self)
        new_self._metadata = metadata
        return new_self

    def string(self) -> "CaMeLStr":
        return CaMeLStr.from_raw(str(self._python_value), Capabilities.camel(), (self,))

    @property
    def raw(self) -> Any:
        return self._python_value

    def type(self) -> "CaMeLStr":
        return CaMeLStr.from_raw(str(type(self._python_value)), Capabilities.camel(), (self,))

    @property
    def raw_type(self) -> str:
        return type(self._python_value).__name__

    def is_(self, other: "CaMeLValue") -> "CaMeLBool":
        camel_metadata = Capabilities.camel()
        return (
            CaMeLTrue(camel_metadata, (self, other))
            if self._python_value is other._python_value
            else CaMeLFalse(camel_metadata, (self, other))
        )

    def is_not(self, other: "CaMeLValue") -> "CaMeLBool":
        camel_metadata = Capabilities.camel()
        return (
            CaMeLTrue(camel_metadata, (self, other))
            if self._python_value is not other._python_value
            else CaMeLFalse(camel_metadata, (self, other))
        )

    def truth(self) -> "CaMeLBool":
        if bool(self._python_value):
            return CaMeLTrue(Capabilities.camel(), (self,))
        return CaMeLFalse(Capabilities.camel(), (self,))

    def not_(self) -> "CaMeLBool":
        if bool(self._python_value):
            return CaMeLFalse(Capabilities.camel(), (self,))
        return CaMeLTrue(Capabilities.camel(), (self,))

    def eq(self, value: "CaMeLValue") -> "CaMeLBool":
        return (
            CaMeLTrue(Capabilities.camel(), (self, value))
            if self._python_value == value._python_value
            else CaMeLFalse(Capabilities.camel(), (self, value))
        )

    def neq(self, value: "CaMeLValue") -> "CaMeLBool":
        return (
            CaMeLTrue(Capabilities.camel(), (self, value))
            if self._python_value != value._python_value
            else CaMeLFalse(Capabilities.camel(), (self, value))
        )

    def hash(self) -> "CaMeLInt":
        return CaMeLInt(hash(self.raw), Capabilities.camel(), (self,))

    def __hash__(self) -> int:
        return hash(self.raw) ^ hash(self._metadata)

    def freeze(self) -> "CaMeLNone": ...


_V = TypeVar("_V", bound=CaMeLValue)
_CT = TypeVar("_CT", bound=PythonComparable)


@runtime_checkable
class TotallyOrdered(CaMeLValue[_CT], Protocol):
    def cmp(self, y: Self) -> "CaMeLInt":
        if self.raw > y.raw:
            return CaMeLInt(1, Capabilities.camel(), (self, y))
        if self.raw < y.raw:
            return CaMeLInt(-1, Capabilities.camel(), (self, y))
        return CaMeLInt(0, Capabilities.camel(), (self, y))


_T = TypeVar("_T")


@runtime_checkable
class CaMeLCallable(CaMeLValue[Callable[..., _T]], Protocol):
    _python_value: Callable[..., _T]
    _metadata: Capabilities
    _name: str
    _recv: CaMeLValue | None
    _bound_python_value: Callable[..., _T] | None = None
    is_class_method: bool = False

    def name(self) -> "CaMeLStr":
        return CaMeLStr.from_raw(self._name, Capabilities.camel(), (self,))

    @property
    def raw(self) -> Any:
        return self._bound_python_value or self._python_value

    def freeze(self) -> "CaMeLNone":
        return CaMeLNone(Capabilities.camel(), (self,))  # already immutable

    def string(self) -> "CaMeLStr":
        return self.name().add(CaMeLStr.from_raw("()", Capabilities.camel(), (self,)))

    def wrap_output(
        self,
        value: _T,
        args: "CaMeLTuple",
        kwargs: "CaMeLDict[CaMeLStr, CaMeLValue]",
        namespace: "ns.Namespace",
    ) -> CaMeLValue[_T]:
        return value_from_raw(
            value,
            Capabilities(
                frozenset({sources.Tool(self.name().raw)}),
                readers.Public(),
            ),
            namespace,
            (self, args, kwargs),
        )

    def call(
        self, args: "CaMeLTuple", kwargs: "CaMeLDict[CaMeLStr, CaMeLValue]", namespace: "ns.Namespace"
    ) -> tuple[CaMeLValue[_T], dict[str, Any]]:
        raw_args = args.raw
        raw_kwargs = kwargs.raw
        output = self._python_value(*raw_args, **raw_kwargs)
        if args.raw != raw_args or kwargs.raw != raw_kwargs:
            raise FunctionCallWithSideEffectError(
                "Call to a function or method with side-effects detected. "
                "Use functions and methods that have no side-effects. "
                "For example, instead of `list.append`, use list comprehensions "
                "or the [*l, new_element] syntax."
            )
        wrapped_output = self.wrap_output(output, args, kwargs, namespace)
        args_by_keyword = self._make_args_by_keyword(args, kwargs)
        return wrapped_output, args_by_keyword

    def bind_recv(self, recv: CaMeLValue):
        self._recv = recv
        # Bind also to the original python method
        self._bound_python_value = types.MethodType(self._python_value, self._recv.raw)

    def receiver(self) -> CaMeLValue | None:
        return self._recv

    def make_args_by_keyword_preserve_values(
        self, args: "CaMeLTuple", kwargs: "CaMeLDict[CaMeLStr, CaMeLValue]"
    ) -> dict[str, CaMeLValue]: ...

    def _make_args_by_keyword(self, args: "CaMeLTuple", kwargs: "CaMeLDict[CaMeLStr, CaMeLValue]") -> dict[str, Any]:
        args_by_keyword = self.make_args_by_keyword_preserve_values(args, kwargs)
        return {k: v.raw for k, v in args_by_keyword.items()}


_IT = TypeVar("_IT", bound=Iterable)


class CaMeLIterable(Generic[_IT, _V], CaMeLValue[_IT]):
    def get_dependencies(
        self, visited_objects: frozenset[int] = frozenset()
    ) -> tuple[tuple["CaMeLValue", ...], frozenset[int]]:
        dependencies = self._dependencies
        if id(self) in visited_objects:
            return dependencies, visited_objects
        for el in self._python_value:
            (new_dependencies, visited_objects) = el.get_dependencies(visited_objects | {id(self)})
            dependencies += new_dependencies
        return dependencies, visited_objects | {id(self)}

    def iterate(self) -> "CaMeLIterator[_V]":
        return CaMeLIterator(iter(self._python_value), Capabilities.camel(), (self,))

    def eq(self, value: "CaMeLValue") -> "CaMeLBool":
        if not isinstance(value, type(self)):
            return CaMeLFalse(Capabilities.camel(), (self, value))
        for self_c, value_c in zip(self._python_value, value._python_value):
            if not self_c.eq(value_c).raw:
                return CaMeLFalse(Capabilities.camel(), (self, value))
        return CaMeLTrue(Capabilities.camel(), (self, value))

    def iterate_python(self) -> Iterator[_V]:
        return iter(self._python_value)

    def contains(self, other: CaMeLValue) -> "CaMeLBool":
        inner_element = next((el for el in self.iterate_python() if el.eq(other)), None)
        if inner_element is not None:
            return CaMeLTrue(Capabilities.camel(), (self, other, inner_element))
        # Add metadata from elements as well as False reveal something about all of them
        # (i.e., that none of them is `other`).
        return CaMeLFalse(Capabilities.camel(), (*self.get_dependencies()[0], other))


_ST = TypeVar("_ST", bound=Sequence)


class CaMeLSequence(Generic[_ST, _V], CaMeLIterable[_ST, _V]):
    def index(self, index: "CaMeLInt") -> _V:
        return self._python_value[index.raw].new_with_dependencies((self, index))

    def slice(self, start: "CaMeLInt | CaMeLNone", end: "CaMeLInt | CaMeLNone", step: "CaMeLInt | CaMeLNone") -> Self:
        s = slice(start.raw, end.raw, step.raw)
        new_python_value: _ST = self._python_value[s]  # type: ignore -- can't specify CT[T] bc Python does not support higher-kinded types
        return self.new_with_python_value(new_python_value).new_with_dependencies((self, start, end, step))

    def len(self) -> "CaMeLInt":
        return CaMeLInt(len(self._python_value), Capabilities.camel(), (self, *self._python_value))


_MCT = TypeVar("_MCT", bound=MutableSequence)


class CaMeLMutableSequence(Generic[_MCT, _V], CaMeLSequence[_MCT, _V]):
    def set_index(self, index: "CaMeLInt", value: _V) -> "CaMeLNone":
        self._python_value[index.raw] = value
        return CaMeLNone(Capabilities.camel(), (self, index))


class CaMeLIterator(Generic[_V], CaMeLValue[Iterator[_V]]):
    def freeze(self) -> "CaMeLNone":
        return CaMeLNone(Capabilities.camel(), (self,))  # already immutable

    def __init__(
        self,
        iterator: Iterator[_V],
        metadata: Capabilities,
        dependencies: tuple[CaMeLValue, ...],
    ) -> None:
        self._python_value = iterator
        self._metadata = metadata
        self._dependencies = dependencies

    def next(self) -> _V:
        return next(self._python_value)


_MT = TypeVar("_MT", bound=Mapping)
_KV = TypeVar("_KV", bound=CaMeLValue)
_VV = TypeVar("_VV", bound=CaMeLValue)


class CaMeLMapping(Generic[_MT, _KV, _VV], CaMeLValue[_MT]):
    def get_dependencies(
        self, visited_objects: frozenset[int] = frozenset()
    ) -> tuple[tuple["CaMeLValue", ...], frozenset[int]]:
        dependencies = self._dependencies
        if id(self) in visited_objects:
            return dependencies, visited_objects
        visited_objects |= {id(self)}
        for k, v in self._python_value.items():
            k_dependencies, k_visited_objects = k.get_dependencies(visited_objects)
            v_dependencies, v_visited_objects = v.get_dependencies(k_visited_objects)
            dependencies += k_dependencies + v_dependencies
            visited_objects = v_visited_objects
        return dependencies, visited_objects

    def get(self, key: _KV) -> _VV:
        dict_key = next((el for el in self.iterate_python() if el.eq(key)), None)
        if dict_key is None:
            raise KeyError(key)
        return self._python_value[dict_key].new_with_dependencies((self, key))

    def eq(self, value: "CaMeLValue") -> "CaMeLBool":
        if not isinstance(value, type(self)):
            return CaMeLFalse(Capabilities.camel(), (self, value))
        for (self_k, self_v), (value_k, value_v) in zip(self._python_value.items(), value._python_value.items()):
            if not self_k.eq(value_k).raw:
                return CaMeLFalse(Capabilities.camel(), (self, value))
            if not self_v.eq(value_v).raw:
                return CaMeLFalse(Capabilities.camel(), (self, value))
        return CaMeLTrue(Capabilities.camel(), (self, value))

    def items(self) -> "CaMeLList[CaMeLTuple]":
        items = []
        for k, v in self._python_value.items():
            items.append(CaMeLTuple((k, v), Capabilities.camel(), (self,)))
        return CaMeLList(items, self._metadata, (self,))

    def iterate_python(self) -> Iterator[_KV]:
        return iter(self._python_value)

    def iterate(self) -> "CaMeLIterator[_KV]":
        return CaMeLIterator(iter(self._python_value), Capabilities.camel(), (self,))

    def contains(self, other: CaMeLValue) -> "CaMeLBool":
        dependencies = [self, other]
        inner_element = next((el for el in self.iterate_python() if el.eq(other)), None)
        if inner_element is not None:
            return CaMeLTrue(Capabilities.camel(), (*dependencies, inner_element))
        keys_dependencies = tuple(dep for k in self.iterate_python() for dep in k.get_dependencies()[0])
        return CaMeLFalse(Capabilities.camel(), (*dependencies, *keys_dependencies))


_MMT = TypeVar("_MMT", bound=MutableMapping)


class CaMeLMutableMapping(Generic[_MMT, _KV, _VV], CaMeLMapping[_MMT, _KV, _VV]):
    def set_key(self, key: _KV, value: _VV) -> "CaMeLNone":
        dict_key = next((el for el in self.iterate_python() if el.eq(key).raw), None)
        if dict_key is None:
            dict_key = key
        if key is not dict_key:
            new_dict_key = dict_key.new_with_dependencies((key,))
            # Remove key value pair with key with old dependencies
            del self._python_value[dict_key]
        else:
            new_dict_key = dict_key
        self._python_value[new_dict_key] = value
        return CaMeLNone(Capabilities.camel(), (self,))


@runtime_checkable
class HasUnary(Protocol):
    def unary(self, op: ast.unaryop) -> Self: ...


@runtime_checkable
class HasAttrs(Generic[_T], CaMeLValue[_T], Protocol):
    def attr(self, name: str) -> CaMeLValue | None: ...
    def attr_names(self) -> set[str]: ...


@runtime_checkable
class HasSetField(Generic[_T], HasAttrs[_T], Protocol):
    def set_field(self, name: str, value: CaMeLValue) -> "CaMeLNone": ...


class CaMeLNone(CaMeLValue[None]):
    _python_value = None

    def __init__(self, metadata: Capabilities, dependencies: tuple[CaMeLValue, ...]) -> None:
        self._metadata = metadata
        self._dependencies = dependencies

    def freeze(self) -> "CaMeLNone":
        return self


class _CaMeLBool(TotallyOrdered[bool]):
    _python_value: bool

    def __bool__(self):
        return self._python_value

    def __init__(self, metadata: Capabilities, dependencies: tuple[CaMeLValue, ...]) -> None:
        self._metadata = metadata
        self._dependencies = dependencies

    def freeze(self) -> "CaMeLNone":
        return CaMeLNone(Capabilities.camel(), (self,))


class CaMeLTrue(_CaMeLBool):
    _python_value = True


class CaMeLFalse(_CaMeLBool):
    _python_value = False


CaMeLBool = CaMeLTrue | CaMeLFalse


class CaMeLFloat(
    TotallyOrdered[float],
    HasUnary,
    op_protocols.SupportsAdd["CaMeLFloat"],
    op_protocols.SupportsRAdd["CaMeLFloat"],
    op_protocols.SupportsSub["CaMeLFloat"],
    op_protocols.SupportsRSub["CaMeLFloat"],
    op_protocols.SupportsMult["CaMeLFloat"],
    op_protocols.SupportsRMult["CaMeLFloat"],
    op_protocols.SupportsTrueDiv["CaMeLFloat"],
    op_protocols.SupportsRTrueDiv["CaMeLFloat"],
    op_protocols.SupportsFloorDiv["CaMeLFloat"],
    op_protocols.SupportsRFloorDiv["CaMeLFloat"],
    op_protocols.SupportsMod["CaMeLFloat"],
    op_protocols.SupportsRMod["CaMeLFloat"],
    op_protocols.SupportsPow["CaMeLFloat"],
    op_protocols.SupportsRPow["CaMeLFloat"],
):
    def __init__(self, val: float, metadata: Capabilities, dependencies: tuple[CaMeLValue, ...]) -> None:
        self._python_value = val
        self._metadata = metadata
        self._dependencies = dependencies

    def freeze(self) -> CaMeLNone:
        return CaMeLNone(self._metadata, (self, *self._dependencies))

    def add(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLFloat | CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            self._python_value + other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def r_add(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLFloat | CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            self._python_value + other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def sub(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLFloat | CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            self._python_value - other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def r_sub(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLFloat | CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            other._python_value - self._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def mult(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLFloat | CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            self._python_value * other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def r_mult(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLFloat | CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            other._python_value * self._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def truediv(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLFloat | CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            self._python_value / other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def r_truediv(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLFloat | CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            other._python_value / self._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def floor_div(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLFloat | CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            self._python_value // other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def r_floor_div(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLFloat | CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            other._python_value // self._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def mod(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLFloat | CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            self._python_value % other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def r_mod(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLFloat | CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            other._python_value % self._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def pow(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLFloat | CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            self._python_value**other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def r_pow(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLFloat | CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            other._python_value**self._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def unary(self, op: ast.unaryop) -> "CaMeLFloat":
        match op:
            case ast.USub():
                r = -self._python_value
            case ast.UAdd():
                r = +self._python_value
            case ast.Invert():
                raise TypeError("bad operand type for unary ~: 'float'")
            case _:
                raise NotImplementedError(f"Unary operator {op} not supported.")
        return CaMeLFloat(r, Capabilities.camel(), (self,))


class CaMeLInt(
    TotallyOrdered[int],
    HasUnary,
    op_protocols.SupportsAdd["CaMeLInt"],
    op_protocols.SupportsSub["CaMeLInt"],
    op_protocols.SupportsMult["CaMeLInt"],
    op_protocols.SupportsTrueDiv["CaMeLFloat"],
    op_protocols.SupportsFloorDiv["CaMeLInt"],
    op_protocols.SupportsMod["CaMeLInt"],
    op_protocols.SupportsPow["CaMeLInt"],
    op_protocols.SupportsLShift["CaMeLInt"],
    op_protocols.SupportsRShift["CaMeLInt"],
    op_protocols.SupportsBitOr["CaMeLInt"],
    op_protocols.SupportsBitXor["CaMeLInt"],
    op_protocols.SupportsBitAnd["CaMeLInt"],
):
    def __init__(self, val: int, metadata: Capabilities, dependencies: tuple[CaMeLValue, ...]) -> None:
        self._python_value = val
        self._metadata = metadata
        self._dependencies = dependencies

    def freeze(self) -> CaMeLNone:
        return CaMeLNone(Capabilities.camel(), (self,))

    def unary(self, op: ast.unaryop) -> "CaMeLInt":
        match op:
            case ast.USub():
                r = -self._python_value
            case ast.UAdd():
                r = +self._python_value
            case ast.Invert():
                r = ~self._python_value
            case _:
                raise NotImplementedError(f"Unary operator {op} not supported.")
        return CaMeLInt(r, self._metadata, (self, *self._dependencies))

    def add(self, other: CaMeLValue) -> "CaMeLInt":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLInt(
            self._python_value + other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def sub(self, other: CaMeLValue) -> "CaMeLInt":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLInt(
            self._python_value - other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def mult(self, other: CaMeLValue) -> "CaMeLInt":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLInt(
            self._python_value * other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def truediv(self, other: CaMeLValue) -> "CaMeLFloat":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLFloat(
            self._python_value / other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def floor_div(self, other: CaMeLValue) -> "CaMeLInt":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLInt(
            self._python_value // other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def mod(self, other: CaMeLValue) -> "CaMeLInt":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLInt(
            self._python_value % other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def pow(self, other: CaMeLValue) -> "CaMeLInt":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLInt(
            self._python_value**other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def bit_or(self, other: CaMeLValue) -> "CaMeLInt":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLInt(
            self._python_value | other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def bit_and(self, other: CaMeLValue) -> "CaMeLInt":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLInt(
            self._python_value & other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def bit_xor(self, other: CaMeLValue) -> "CaMeLInt":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLInt(
            self._python_value ^ other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def r_shift(self, other: CaMeLValue) -> "CaMeLInt":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLInt(
            self._python_value >> other._python_value,
            Capabilities.camel(),
            (self, other),
        )

    def l_shift(self, other: CaMeLValue) -> "CaMeLInt":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLInt(
            self._python_value << other._python_value,
            Capabilities.camel(),
            (self, other),
        )


class _CaMeLChar(TotallyOrdered[str]):
    def __init__(self, val: str, metadata: Capabilities, dependencies: tuple[CaMeLValue, ...]) -> None:
        self._python_value = val
        self._metadata = metadata
        self._dependencies = dependencies

    def __gt__(self, other) -> bool:
        if not isinstance(other, _CaMeLChar):
            return NotImplemented
        return self._python_value > other._python_value

    def __lt__(self, other) -> bool:
        if not isinstance(other, _CaMeLChar):
            return NotImplemented
        return self._python_value < other._python_value

    def freeze(self) -> "CaMeLNone":
        return CaMeLNone(Capabilities.camel(), (self,))  # already immutable


class CaMeLStr(
    TotallyOrdered[tuple[_CaMeLChar, ...]],
    HasAttrs,
    CaMeLSequence[tuple, _CaMeLChar],
    op_protocols.SupportsAdd["CaMeLStr"],
    op_protocols.SupportsMult["CaMeLStr"],
    op_protocols.SupportsRMult["CaMeLStr"],
):
    def __init__(
        self,
        string: Sequence[_CaMeLChar],
        metadata: Capabilities,
        dependencies: tuple[CaMeLValue, ...],
    ) -> None:
        self._python_value = tuple(string)
        self._metadata = metadata
        self._dependencies = dependencies

    def contains(self, other: CaMeLValue) -> "CaMeLBool":
        if not isinstance(other, CaMeLStr | _CaMeLChar):
            raise TypeError(f"in <string>' requires string as left operand, not {other.raw_type}")
        if other.raw in self.raw:
            return CaMeLTrue(Capabilities.camel(), (self, *other.get_dependencies()[0]))
        # Add metadata from elements as well as False reveal something about all of them
        # (i.e., that none of them is `other`).
        return CaMeLFalse(Capabilities.camel(), (*self.get_dependencies()[0], other))

    @classmethod
    def from_raw(cls, string: str, metadata: Capabilities, dependencies: tuple[CaMeLValue, ...]) -> Self:
        return cls(
            tuple(_CaMeLChar(c, metadata, dependencies) for c in string),
            metadata,
            dependencies,
        )

    def attr(self, name) -> CaMeLValue | None:
        # need local import because of otherwise circular import
        from camel.interpreter import library

        attr = library.SUPPORTED_BUILT_IN_METHODS[self.raw_type].get(name)
        if attr is not None:
            return attr.new_with_dependencies((self,))
        return attr

    def attr_names(self) -> set[str]:
        # need local import because of otherwise circular import
        from camel.interpreter import library

        return set(library.SUPPORTED_BUILT_IN_METHODS[self.raw_type].keys())

    def string(self) -> "CaMeLStr":
        return self

    def freeze(self) -> "CaMeLNone":
        return CaMeLNone(Capabilities.camel(), (self,))  # already immutable

    def iterate(self) -> CaMeLIterator[_CaMeLChar]:
        return super().iterate()

    @property
    def raw(self) -> str:
        s = ""
        for c in self._python_value:
            s += c._python_value
        return s

    @property
    def raw_type(self) -> str:
        return "str"

    def add(self, other: CaMeLValue) -> "CaMeLStr":
        if not isinstance(other, CaMeLStr):
            return NotImplemented
        return CaMeLStr(self._python_value + other._python_value, Capabilities.camel(), (self, other))

    def mult(self, other: CaMeLValue) -> "CaMeLStr":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLStr(self._python_value * other._python_value, Capabilities.camel(), (self, other))

    r_mult = mult


class CaMeLTuple(
    Generic[_V],
    TotallyOrdered[tuple[_V, ...]],
    CaMeLSequence[tuple[_V, ...], _V],
    op_protocols.SupportsAdd["CaMeLTuple"],
    op_protocols.SupportsMult["CaMeLTuple"],
    op_protocols.SupportsRMult["CaMeLTuple"],
):
    def __init__(
        self,
        it: Iterable[_V],
        metadata: Capabilities,
        dependencies: tuple[CaMeLValue, ...],
    ) -> None:
        self._metadata = metadata
        self._python_value = tuple(it)
        self._dependencies = dependencies

    @property
    def raw(self) -> tuple:
        return tuple(v.raw for v in self._python_value)

    def freeze(self) -> "CaMeLNone":
        return CaMeLNone(Capabilities.camel(), (self,))  # already immutable

    def add(self, other: CaMeLValue) -> "CaMeLTuple":
        if not isinstance(other, CaMeLTuple):
            return NotImplemented
        return CaMeLTuple(self._python_value + other._python_value, Capabilities.camel(), (self, other))

    def mult(self, other: CaMeLValue) -> "CaMeLTuple":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLTuple(self._python_value * other._python_value, Capabilities.camel(), (self, other))

    r_mult = mult


class CaMeLList(
    Generic[_V],
    TotallyOrdered[list[_V]],
    CaMeLMutableSequence[list[_V], _V],
    op_protocols.SupportsAdd["CaMeLList"],
    op_protocols.SupportsMult["CaMeLList"],
    op_protocols.SupportsRMult["CaMeLList"],
):
    def __init__(
        self,
        it: Iterable[_V],
        metadata: Capabilities,
        dependencies: tuple[CaMeLValue, ...],
    ) -> None:
        self._python_value = list(it)
        self._frozen = False
        self._metadata = metadata
        self._dependencies = dependencies

    @property
    def raw(self) -> list:
        return list(v.raw for v in self._python_value)

    def attr(self, name) -> CaMeLValue | None:
        # need local import because of otherwise circular import
        from camel.interpreter import library

        attr = library.SUPPORTED_BUILT_IN_METHODS[self.raw_type].get(name)
        if attr is not None:
            return attr.new_with_dependencies((self,))
        return attr

    def attr_names(self) -> set[str]:
        # need local import because of otherwise circular import
        from camel.interpreter import library

        return set(library.SUPPORTED_BUILT_IN_METHODS[self.raw_type].keys())

    def freeze(self) -> "CaMeLNone":
        [el.freeze() for el in self._python_value]
        self._frozen = True
        return CaMeLNone(Capabilities.camel(), (self, *self.iterate_python()))

    def add(self, other: CaMeLValue) -> "CaMeLList":
        if not isinstance(other, CaMeLList):
            return NotImplemented

        return CaMeLList(self._python_value + other._python_value, Capabilities.camel(), (self, other))

    def mult(self, other: CaMeLValue) -> "CaMeLList":
        if not isinstance(other, CaMeLInt):
            return NotImplemented
        return CaMeLList(self._python_value * other._python_value, Capabilities.camel(), (self, other))

    r_mult = mult


class CaMeLSet(
    Generic[_V],
    TotallyOrdered[set[_V]],
    CaMeLIterable[set[_V], _V],
    op_protocols.SupportsBitOr["CaMeLSet"],
    op_protocols.SupportsBitAnd["CaMeLSet"],
    op_protocols.SupportsSub["CaMeLSet"],
):
    def __init__(
        self,
        it: Iterable[_V],
        metadata: Capabilities,
        dependencies: tuple[CaMeLValue, ...],
    ) -> None:
        self._python_value = set(it)
        self._metadata = metadata
        self._dependencies = dependencies

    @property
    def raw(self) -> set:
        return set(v.raw for v in self._python_value)

    def freeze(self) -> "CaMeLNone":
        [el.freeze() for el in self._python_value]
        self._frozen = True
        return CaMeLNone(Capabilities.camel(), (self, *self.iterate_python()))

    def sub(self, other: CaMeLValue) -> "CaMeLSet":
        if not isinstance(other, CaMeLSet):
            return NotImplemented
        return CaMeLSet(self._python_value - other._python_value, Capabilities.camel(), (self, other))

    def bit_or(self, other: CaMeLValue) -> "CaMeLSet":
        if not isinstance(other, CaMeLSet):
            return NotImplemented
        return CaMeLSet(self._python_value | other._python_value, Capabilities.camel(), (self, other))

    def bit_xor(self, other: CaMeLValue) -> "CaMeLSet":
        if not isinstance(other, CaMeLSet):
            return NotImplemented
        return CaMeLSet(self._python_value ^ other._python_value, Capabilities.camel(), (self, other))

    def bit_and(self, other: CaMeLValue) -> "CaMeLSet":
        if not isinstance(other, CaMeLSet):
            raise TypeError(f"unsupported operand type(s) for &: '{self.raw_type}' and '{other.raw_type}'")
        return CaMeLSet(self._python_value & other._python_value, Capabilities.camel(), (self, other))


class CaMeLDict(
    Generic[_KV, _VV],
    CaMeLMutableMapping[dict[_KV, _VV], _KV, _VV],
    op_protocols.SupportsBitOr["CaMeLDict[_KV, _VV]"],
    HasAttrs,
):
    def __init__(
        self,
        it: Mapping[_KV, _VV],
        metadata: Capabilities,
        dependencies: tuple[CaMeLValue, ...],
    ) -> None:
        self._python_value = dict(it)
        self._frozen = False
        self._metadata = metadata
        self._dependencies = dependencies

    @property
    def raw(self) -> dict:
        return {k.raw: v.raw for k, v in self._python_value.items()}

    def attr(self, name) -> CaMeLValue | None:
        # need local import because of otherwise circular import
        from camel.interpreter import library

        attr = library.SUPPORTED_BUILT_IN_METHODS[self.raw_type].get(name)
        if attr is not None:
            return attr.new_with_dependencies((self,))
        return attr

    def attr_names(self) -> set[str]:
        # need local import because of otherwise circular import
        from camel.interpreter import library

        return set(library.SUPPORTED_BUILT_IN_METHODS[self.raw_type].keys())

    def freeze(self) -> "CaMeLNone":
        [(k.freeze(), v.freeze()) for k, v in self._python_value.items()]
        self._frozen = True
        return CaMeLNone(
            Capabilities.camel(),
            (self, *self._python_value.keys(), *self._python_value.values()),
        )

    def bit_or(
        self,
        other: CaMeLValue,
    ) -> "CaMeLDict[_KV, _VV]":
        if not isinstance(other, CaMeLDict):
            return NotImplemented
        return CaMeLDict(self._python_value | other._python_value, Capabilities.camel(), (self, other))


def _get_class_attr_names(instance: Any | type) -> set[str]:
    if dataclasses.is_dataclass(instance):
        return {f.name for f in dataclasses.fields(instance)}

    if isinstance(instance, pydantic.BaseModel):
        return {f for f in type(instance).model_fields}
    if isinstance(instance, type) and issubclass(instance, pydantic.BaseModel):
        return {f for f in instance.model_fields}

    return {
        attr
        for attr in dir(instance)
        if not (attr.startswith("__") and attr.endswith("__")) and not isinstance(getattr(instance, attr), Callable)
    }


class CaMeLClass(Generic[_T], CaMeLCallable[_T], HasAttrs):
    def __init__(
        self,
        name: str,
        callable: type[_T],
        metadata: Capabilities,
        dependencies: tuple[CaMeLValue, ...],
        methods: dict[str, CaMeLCallable],
        base_classes: tuple["CaMeLClass", ...] = (),
        is_totally_ordered: bool = False,
        is_builtin: bool = False,
    ):
        self._python_value = callable  # type: ignore
        self._metadata = metadata
        self._name = name
        self._base_classes = base_classes
        self._recv: CaMeLValue | None = None
        self._methods = methods | {k: v for d in [c._methods for c in base_classes] for k, v in d.items()}
        self._is_totally_ordered = is_totally_ordered
        self._dependencies = dependencies
        self.is_builtin = is_builtin

    def __hash__(self) -> int:
        return super().__hash__()

    def get_dependencies(
        self, visited_objects: frozenset[int] = frozenset()
    ) -> tuple[tuple["CaMeLValue", ...], frozenset[int]]:
        dependencies = self._dependencies
        if id(self) in visited_objects:
            return dependencies, visited_objects
        for method in self._methods.values():
            new_dependencies, visited_objects = method.get_dependencies(visited_objects | {id(self)})
            dependencies += new_dependencies
        return dependencies, visited_objects

    def init(
        self,
        namespace: "ns.Namespace",
        *args: Any,
        **kwargs: dict[str, Any],
    ) -> "CaMeLClassInstance[_T]":
        return CaMeLClassInstance(
            self._python_value(*args, **kwargs),
            self,
            self._metadata,
            namespace,
            (self,),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CaMeLClass):
            return False
        return (
            self._python_value.__name__ == other._python_value.__name__
            and dir(self._python_value) == dir(other._python_value)
            and self._metadata == other._metadata
        )

    def call(
        self, args: CaMeLTuple, kwargs: CaMeLDict[CaMeLStr, CaMeLValue], namespace: "ns.Namespace"
    ) -> tuple["CaMeLClassInstance[_T]", dict[str, Any]]:
        return self.init(namespace, *args.raw, **kwargs.raw), self._make_args_by_keyword(args, kwargs)

    def make_args_by_keyword_preserve_values(
        self, args: "CaMeLTuple", kwargs: "CaMeLDict[CaMeLStr, CaMeLValue]"
    ) -> dict[str, CaMeLValue]:
        return {str(i): arg for i, arg in enumerate(args.iterate_python())} | {
            k.raw: v for k, v in kwargs._python_value.items()
        }

    def freeze(self) -> "CaMeLNone":
        return CaMeLNone(Capabilities.camel(), (self,))  # already immutable

    def attr(self, name: str) -> CaMeLValue | None:
        assert isinstance(self._python_value, type)
        if issubclass(self._python_value, Enum):
            try:
                val: Enum = getattr(self._python_value, name)
            except AttributeError:
                return None
            if isinstance(val, str):
                return CaMeLStr.from_raw(val, self._metadata, (self,))
            if isinstance(val, int):
                return CaMeLInt(val, self._metadata, (self,))
            raise TypeError(f"invalid enum type: {type(val)}")
        if name not in self._methods:
            return None
        return self._methods[name].new_with_dependencies((self,))

    def attr_names(self) -> set[str]:
        class_attrs = _get_class_attr_names(self._python_value) | set(self._methods.keys())
        return class_attrs | (set.union(*[c.attr_names() for c in self._base_classes]) if self._base_classes else set())


class CaMeLClassInstance(Generic[_T], HasSetField[_T]):
    def __init__(
        self,
        value: _T,
        _class: CaMeLClass[_T],
        metadata: Capabilities,
        namespace: "ns.Namespace",
        dependencies: tuple[CaMeLValue, ...],
    ):
        self._python_value = value
        self._class = _class
        self._metadata = metadata
        self._namespace = namespace
        self._dependencies = dependencies
        self._frozen = False

        if self._class._is_totally_ordered:
            self.cmp = self._cmp

    def __hash__(self) -> int:
        return super().__hash__()

    def get_dependencies(
        self, visited_objects: frozenset[int] = frozenset()
    ) -> tuple[tuple["CaMeLValue", ...], frozenset[int]]:
        dependencies = self._dependencies
        if id(self) in visited_objects:
            return dependencies, visited_objects
        for attr_name in self.attr_names():
            attr = self.attr(attr_name)
            if attr is not None and attr_name not in self._class._methods:
                new_dependencies, visited_objects = attr.get_dependencies(visited_objects | {id(self)})
                dependencies += new_dependencies
        return dependencies, visited_objects

    def _cmp(self, y: Self) -> "CaMeLInt":
        if self.raw > y.raw:  # type: ignore -- this is hardcoded
            return CaMeLInt(1, Capabilities.camel(), (self, y))
        if self.raw < y.raw:  # type: ignore -- this is hardcoded
            return CaMeLInt(-1, Capabilities.camel(), (self, y))
        return CaMeLInt(0, Capabilities.camel(), (self, y))

    def __eq__(self, other) -> bool:
        if not isinstance(other, CaMeLClassInstance):
            return False
        return (
            self._python_value == other._python_value
            and self._class == other._class
            and self._metadata == other._metadata
            and self._frozen == other._frozen
        )

    @property
    def raw(self) -> _T:
        # create a copy of the Python instance
        instance = copy.copy(self._python_value)
        # replace all `CaMeLValue` attributes with their `raw` respective
        for attr_name in self.attr_names():
            attr_value = getattr(self._python_value, attr_name)
            if isinstance(attr_value, CaMeLValue):
                setattr(instance, attr_name, attr_value.raw)
        return instance

    def freeze(self) -> "CaMeLNone":
        self._frozen = True
        return CaMeLNone(Capabilities.camel(), (self,))  # already immutable

    def set_field(self, name: str, value: CaMeLValue) -> "CaMeLNone":
        if self._frozen:
            raise ValueError("instance is frozen")
        setattr(self._python_value, name, value)
        return CaMeLNone(Capabilities.default(), ())

    def attr(self, name: str) -> CaMeLValue | None:
        if name not in self.attr_names():
            return None
        if name in self._class._methods:
            return self._class._methods[name]
        attr = getattr(self._python_value, name)
        if not isinstance(attr, CaMeLValue):
            return value_from_raw(attr, Capabilities.camel(), self._namespace, ())
        return attr.new_with_dependencies((self,))

    def attr_names(self) -> set[str]:
        return self._class.attr_names() | _get_class_attr_names(self._python_value)


class CaMeLValueAsWrapper(Generic[_T], CaMeLClassInstance[_T]):
    def __init__(
        self,
        value: _T,
        metadata: Capabilities,
        namespace: "ns.Namespace",
        dependencies: tuple[CaMeLValue, ...],
    ):
        _class = namespace.get(type(value).__name__)
        if _class is None:
            raise NameError(f"name {type(value).__name__} is not defined")
        if not isinstance(_class, CaMeLClass):
            raise TypeError(f"{type(value).__name__} is not callable")
        super().__init__(value, _class, metadata, namespace, dependencies)

    @property
    def raw(self) -> _T:
        return self._python_value

    def get_dependencies(
        self, visited_objects: frozenset[int] = frozenset()
    ) -> tuple[tuple["CaMeLValue", ...], frozenset[int]]:
        return self._dependencies, visited_objects | {id(self)}

    def attr(self, name: str) -> CaMeLValue | None:
        if name not in self.attr_names():
            return None
        return value_from_raw(
            getattr(self._python_value, name),
            Capabilities.camel(),
            self._namespace,
            self._dependencies,
        )

    def set_field(self, name: str, value: CaMeLValue) -> "CaMeLNone":
        if self._frozen:
            raise ValueError("instance is frozen")
        setattr(self._python_value, name, value.raw)
        return CaMeLNone(Capabilities.default(), ())

    def freeze(self) -> CaMeLNone:
        return CaMeLNone(Capabilities.camel(), (self,))  # already immutable


class CaMeLFunction(Generic[_T], CaMeLCallable[_T]):
    def __init__(
        self,
        name: str,
        callable: Callable[..., _T],
        metadata: Capabilities,
        dependencies: tuple[CaMeLValue, ...],
    ):
        self._python_value = callable
        self._metadata = metadata
        self._dependencies = dependencies
        self._name = name
        self._recv: CaMeLValue | None = None

    def make_args_by_keyword_preserve_values(
        self, args: "CaMeLTuple", kwargs: "CaMeLDict[CaMeLStr, CaMeLValue]"
    ) -> dict[str, CaMeLValue]:
        return {str(i): arg for i, arg in enumerate(args.iterate_python())} | {
            k.raw: v for k, v in kwargs._python_value.items()
        }


class UndefinedClassError(Exception): ...


@conditional_lru_cache()
def value_from_raw(
    raw_value: Any,
    metadata: Capabilities,
    namespace: "ns.Namespace",
    dependencies: tuple[CaMeLValue, ...],
) -> CaMeLValue:
    match raw_value:
        # Extremely important to keep the order because in Python `bool` subclasses `int`
        case bool():
            return CaMeLTrue(metadata, dependencies) if raw_value is True else CaMeLFalse(metadata, dependencies)
        case int():
            return CaMeLInt(raw_value, metadata, dependencies)
        case str():
            return CaMeLStr.from_raw(raw_value, metadata, dependencies)
        case float():
            return CaMeLFloat(raw_value, metadata, dependencies)
        case None:
            return CaMeLNone(metadata, dependencies)
        case list():
            return CaMeLList(
                [value_from_raw(val, Capabilities.camel(), namespace, ()) for val in raw_value],
                metadata,
                dependencies,
            )
        case dict():
            return CaMeLDict(
                {
                    value_from_raw(k, Capabilities.camel(), namespace, ()): value_from_raw(v, metadata, namespace, ())
                    for k, v in raw_value.items()
                },
                metadata,
                dependencies,
            )
        case set():
            return CaMeLSet(
                {value_from_raw(val, Capabilities.camel(), namespace, ()) for val in raw_value},
                metadata,
                dependencies,
            )
        case tuple():
            return CaMeLTuple(
                tuple(value_from_raw(val, Capabilities.camel(), namespace, ()) for val in raw_value),
                metadata,
                dependencies,
            )
        case type():
            return CaMeLClass(raw_value.__name__, raw_value, metadata, dependencies, {})
        case Callable():
            return CaMeLFunction(raw_value.__name__, raw_value, metadata, dependencies)
        case _ if (value_class := namespace.get(type(raw_value).__name__)) is not None and isinstance(
            value_class, CaMeLClass
        ):
            raw_value_copy = copy.copy(raw_value)
            for attr in value_class.attr_names():
                if attr in value_class._methods:
                    continue
                try:
                    converted_attr = value_from_raw(getattr(raw_value, attr), Capabilities.camel(), namespace, ())
                    setattr(raw_value_copy, attr, converted_attr)
                except (AttributeError, RecursionError):
                    # Some built-in classes do not allow writing some attributes
                    # and/or have weird self references (which causes RecursionErrors)
                    # so we wrap the entire class for consistency
                    return CaMeLValueAsWrapper(raw_value, metadata, namespace, dependencies)
            return CaMeLClassInstance(raw_value_copy, value_class, metadata, namespace, dependencies)
        case _:
            # CaMeLValue of unknown class, raise exception
            raise UndefinedClassError(f"Undefined class {type(raw_value).__name__}")


class CaMeLBuiltin(Generic[_T], CaMeLCallable[_T]):
    is_builtin: bool = True

    def __init__(
        self,
        name: str,
        callable: Callable[..., _T],
        metadata: Capabilities,
        dependencies: tuple[CaMeLValue, ...],
        is_class_method: bool = False,
    ):
        self._python_value = callable
        self._metadata = metadata
        self._name = name
        self._recv: CaMeLValue | None = None
        self.is_class_method = is_class_method
        self._dependencies = dependencies

    def freeze(self) -> CaMeLNone:
        if self._recv is not None:
            self._recv.freeze()
            dependencies = (self, self._recv)
        else:
            dependencies = (self,)
        return CaMeLNone(Capabilities.camel(), dependencies)

    def string(self) -> CaMeLStr:
        if self._recv is None:
            return CaMeLStr.from_raw(f"built-in function {self._name}", Capabilities.camel(), (self,))
        return CaMeLStr.from_raw(
            f"built-in method '{self._name}' of {self._recv.raw_type} object",
            Capabilities.camel(),
            (self,),
        )

    def type(self) -> CaMeLStr:
        return CaMeLStr.from_raw("builtin_function_or_method", Capabilities.camel(), (self,))

    def make_args_by_keyword_preserve_values(
        self, args: "CaMeLTuple", kwargs: "CaMeLDict[CaMeLStr, CaMeLValue]"
    ) -> dict[str, CaMeLValue]:
        return {str(i): arg for i, arg in enumerate(args.iterate_python())} | {
            k.raw: v for k, v in kwargs._python_value.items()
        }


def make_camel_builtin(name: str, fn: Callable[..., _T], is_class_method: bool = False) -> CaMeLBuiltin[_T]:
    return CaMeLBuiltin(name, fn, Capabilities.camel(), (), is_class_method)

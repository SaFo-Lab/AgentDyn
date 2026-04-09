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

"""Standard library implementation

From https://github.com/google/starlark-go/blob/master/starlark/library.go#L157
"""

import datetime
import enum
from collections.abc import Iterable, Reversible
from typing import Any, TypeVar

import pydantic
import pydantic.fields

from camel.capabilities import Capabilities
from camel.interpreter import value
from camel.quarantined_llm import NotEnoughInformationError


# This can't be typed in a more narrow way bc of limitations of the Python type system.
def camel_zip(
    *x: Iterable[Any],
) -> list[tuple[Any, ...]]:
    return list(zip(*x))


_T = TypeVar("_T")


def camel_enumerate(x: Iterable[_T], start: int = 0) -> list[tuple[int, _T]]:
    return list(enumerate(x, start))


def camel_reversed(x: Reversible[_T]) -> list[_T]:
    return list(reversed(x))


def camel_bool(x: object) -> bool:
    return bool(x)


def camel_dir(x: object) -> list[str]:
    return dir(x)


def camel_range(start: int, stop: int | None = None, step: int | None = None, /) -> list[int]:
    match (stop, step):
        case None, None:
            return list(range(start))
        case (_, None):
            return list(range(start, stop))
        case (None, _):
            raise TypeError("'NoneType' object cannot be interpreted as an integer")
        case (_, _):
            return list(range(start, stop, step))


def camel_print(
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n",
    file: Any | None = None,
    flush: bool = False,
) -> None:
    return None


BUILT_IN_FUNCTIONS: dict[str, value.CaMeLBuiltin] = {
    "abs": value.make_camel_builtin("abs", abs),
    "any": value.make_camel_builtin("any", any),
    "all": value.make_camel_builtin("all", all),
    "bool": value.make_camel_builtin("bool", camel_bool),
    "dir": value.make_camel_builtin("dir", camel_dir),
    "divmod": value.make_camel_builtin("divmod", divmod),
    # We don't want lazy objects, so `enumerate` must return a list
    "enumerate": value.make_camel_builtin("enumerate", camel_enumerate),
    "float": value.make_camel_builtin("float", float),
    "hash": value.make_camel_builtin("hash", hash),
    "int": value.make_camel_builtin("int", int),
    "len": value.make_camel_builtin("len", len),
    "list": value.make_camel_builtin("list", list),
    "max": value.make_camel_builtin("max", max),
    "min": value.make_camel_builtin("min", min),
    "print": value.make_camel_builtin("print", camel_print),
    "range": value.make_camel_builtin("range", camel_range),
    "repr": value.make_camel_builtin("repr", repr),
    # We don't want lazy objects, so `reversed` must return a list of tuples
    "reversed": value.make_camel_builtin("reversed", camel_reversed),
    "set": value.make_camel_builtin("set", set),
    "sorted": value.make_camel_builtin("sorted", sorted),
    "str": value.make_camel_builtin("str", str),
    "tuple": value.make_camel_builtin("tuple", tuple),
    "type": value.make_camel_builtin("type", lambda x: type(x).__name__),
    # We don't want lazy objects, so `zip` must return a list of tuples
    "zip": value.make_camel_builtin("zip", camel_zip),
    "sum": value.make_camel_builtin("sum", sum),
}
"""Built-in functions supported in Starlark (plus or minus some extra).

See https://github.com/bazelbuild/starlark/blob/master/spec.md#built-in-constants-and-functions"""


SUPPORTED_BUILT_IN_METHODS: dict[str, dict[str, value.CaMeLBuiltin]] = {
    "dict": {
        # "clear": value.make_builtin("clear", dict.clear),
        "get": value.make_camel_builtin("get", dict.get),
        "items": value.make_camel_builtin("items", lambda d: list(d.items())),
        "keys": value.make_camel_builtin("keys", lambda d: list(d.keys())),
        # "pop": value.make_builtin("pop", dict.pop),
        # "popitem": value.make_builtin("popitem", dict.popitem),
        # "setdefault": value.make_builtin("setdefault", dict.setdefault),
        # "update": value.make_builtin("update", dict.update),
        "values": value.make_camel_builtin("values", lambda d: list(d.values())),
    },
    "list": {
        # "append": value.make_builtin("append", list.append),
        # "clear": value.make_builtin("clear", list.clear),
        # "extend": value.make_builtin("extend", list.extend),
        "index": value.make_camel_builtin("index", list.index),
        # "insert": value.make_builtin("insert", list.insert),
        # "pop": value.make_builtin("pop", list.pop),
        # "remove": value.make_builtin("remove", list.remove),
    },
    "str": {
        "capitalize": value.make_camel_builtin("capitalize", str.capitalize),
        "count": value.make_camel_builtin("count", str.count),
        "endswith": value.make_camel_builtin("endswith", str.endswith),
        "find": value.make_camel_builtin("find", str.find),
        "format": value.make_camel_builtin("format", str.format),
        "index": value.make_camel_builtin("index", str.index),
        "isalnum": value.make_camel_builtin("isalnum", str.isalnum),
        "isalpha": value.make_camel_builtin("isalpha", str.isalpha),
        "isdigit": value.make_camel_builtin("isdigit", str.isdigit),
        "islower": value.make_camel_builtin("islower", str.islower),
        "isspace": value.make_camel_builtin("isspace", str.isspace),
        "istitle": value.make_camel_builtin("istitle", str.istitle),
        "isupper": value.make_camel_builtin("isupper", str.isupper),
        "join": value.make_camel_builtin("join", str.join),
        "lower": value.make_camel_builtin("lower", str.lower),
        "lstrip": value.make_camel_builtin("lstrip", str.lstrip),
        "partition": value.make_camel_builtin("partition", str.partition),
        "removeprefix": value.make_camel_builtin("removeprefix", str.removeprefix),
        "removesuffix": value.make_camel_builtin("removesuffix", str.removesuffix),
        "replace": value.make_camel_builtin("replace", str.replace),
        "rfind": value.make_camel_builtin("rfind", str.rfind),
        "rindex": value.make_camel_builtin("rindex", str.rindex),
        "rpartition": value.make_camel_builtin("rpartition", str.rpartition),
        "rsplit": value.make_camel_builtin("rsplit", str.rsplit),
        "rstrip": value.make_camel_builtin("rstrip", str.rstrip),
        "split": value.make_camel_builtin("split", str.split),
        "splitlines": value.make_camel_builtin("splitlines", str.splitlines),
        "startswith": value.make_camel_builtin("startswith", str.startswith),
        "strip": value.make_camel_builtin("strip", str.strip),
        "title": value.make_camel_builtin("title", str.title),
        "upper": value.make_camel_builtin("upper", str.upper),
    },
}
"""Built-in methods supported in Starlark (plus or minus some extra).

See https://github.com/bazelbuild/starlark/blob/master/spec.md#built-in-constants-and-functions"""

BUILT_IN_CLASSES: dict[str, value.CaMeLClass] = {
    "ValueError": value.CaMeLClass("ValueError", ValueError, Capabilities.camel(), (), {}, is_builtin=True),
    "NotEnoughInformationError": value.CaMeLClass(
        "NotEnoughInformationError",
        NotEnoughInformationError,
        Capabilities.camel(),
        (),
        {},
        is_builtin=True,
    ),
    "Enum": value.CaMeLClass("Enum", enum.Enum, Capabilities.camel(), (), {}, is_builtin=True),
    "datetime": value.CaMeLClass(
        "datetime",
        datetime.datetime,
        Capabilities.camel(),
        (),
        {
            "strftime": value.make_camel_builtin("strftime", datetime.datetime.strftime),
            "replace": value.make_camel_builtin("replace", datetime.datetime.replace),
            "isoformat": value.make_camel_builtin("isoformat", datetime.datetime.isoformat),
            "utcoffset": value.make_camel_builtin("utcoffset", datetime.datetime.utcoffset),
            "strptime": value.make_camel_builtin("strptime", datetime.datetime.strptime, is_class_method=True),
            "fromisoformat": value.make_camel_builtin(
                "fromisoformat", datetime.datetime.fromisoformat, is_class_method=True
            ),
            "date": value.make_camel_builtin("date", datetime.datetime.date, is_class_method=False),
            "time": value.make_camel_builtin("time", datetime.datetime.time, is_class_method=False),
            "weekday": value.make_camel_builtin("weekday", datetime.datetime.weekday, is_class_method=False),
            "combine": value.make_camel_builtin("combine", datetime.datetime.combine, is_class_method=True),
            "__add__": value.make_camel_builtin("__add__", datetime.datetime.__add__),  # Operator method in methods
            "__sub__": value.make_camel_builtin("__sub__", datetime.datetime.__sub__),  # Operator method in methods
        },
        is_totally_ordered=True,
        is_builtin=True,
    ),
    "timedelta": value.CaMeLClass(
        "timedelta",
        datetime.timedelta,
        Capabilities.camel(),
        (),
        {
            "total_seconds": value.make_camel_builtin("total_seconds", datetime.timedelta.total_seconds),
            "__add__": value.make_camel_builtin("__add__", datetime.timedelta.__add__),  # Operator method in methods
            "__sub__": value.make_camel_builtin("__sub__", datetime.timedelta.__sub__),  # Operator method in methods
            "__mul__": value.make_camel_builtin("__mul__", datetime.timedelta.__mul__),
            "__truediv__": value.make_camel_builtin("__truediv__", datetime.timedelta.__truediv__),
            "__radd__": value.make_camel_builtin("__radd__", datetime.timedelta.__radd__),
            "__rsub__": value.make_camel_builtin("__rsub__", datetime.timedelta.__rsub__),
            "__rmul__": value.make_camel_builtin("__rmul__", datetime.timedelta.__rmul__),
        },
        is_totally_ordered=True,
        is_builtin=True,
    ),
    "date": value.CaMeLClass(
        "date",
        datetime.date,
        Capabilities.camel(),
        (),
        {
            "replace": value.make_camel_builtin("replace", datetime.date.replace),
            "isoformat": value.make_camel_builtin("isoformat", datetime.date.isoformat),
            "strftime": value.make_camel_builtin("strftime", datetime.date.strftime),
            "fromisoformat": value.make_camel_builtin(
                "fromisoformat", datetime.date.fromisoformat, is_class_method=True
            ),
            "__add__": value.make_camel_builtin("__add__", datetime.date.__add__),
            "__radd__": value.make_camel_builtin("__radd__", datetime.date.__radd__),
            "__sub__": value.make_camel_builtin("__sub__", datetime.date.__sub__),
        },
        is_totally_ordered=True,
        is_builtin=True,
    ),
    "time": value.CaMeLClass(
        "time",
        datetime.time,
        Capabilities.camel(),
        (),
        {
            "replace": value.make_camel_builtin("replace", datetime.time.replace),
            "isoformat": value.make_camel_builtin("isoformat", datetime.time.isoformat),
            "strftime": value.make_camel_builtin("strftime", datetime.time.strftime),
            "fromisoformat": value.make_camel_builtin(
                "fromisoformat", datetime.date.fromisoformat, is_class_method=True
            ),
        },
        is_totally_ordered=True,
        is_builtin=True,
    ),
    "timezone": value.CaMeLClass(
        "timezone",
        datetime.timezone,
        Capabilities.camel(),
        (),
        {
            "utcoffset": value.make_camel_builtin("utcoffset", datetime.timezone.utcoffset),
            "tzname": value.make_camel_builtin("tzname", datetime.timezone.tzname),
            "dst": value.make_camel_builtin("dst", datetime.timezone.dst),
        },
        is_totally_ordered=False,
        is_builtin=True,
    ),
    "BaseModel": value.CaMeLClass(
        "BaseModel",
        pydantic.BaseModel,
        Capabilities.camel(),
        (),
        {
            "model_construct": value.make_camel_builtin("model_construct", pydantic.BaseModel.model_construct),
            "model_copy": value.make_camel_builtin("model_copy", pydantic.BaseModel.model_copy),
            "model_dump": value.make_camel_builtin("model_dump", pydantic.BaseModel.model_dump),
            "model_dump_json": value.make_camel_builtin("model_dump_json", pydantic.BaseModel.model_dump_json),
            "model_json_schema": value.make_camel_builtin("model_json_schema", pydantic.BaseModel.model_json_schema),
            "model_parametrized_name": value.make_camel_builtin(
                "model_parametrized_name", pydantic.BaseModel.model_parametrized_name
            ),
            "model_validate": value.make_camel_builtin("model_validate", pydantic.BaseModel.model_validate),
            "model_validate_json": value.make_camel_builtin(
                "model_validate_json", pydantic.BaseModel.model_validate_json
            ),
            "model_validate_strings": value.make_camel_builtin(
                "model_validate_strings", pydantic.BaseModel.model_validate_strings
            ),
        },
        is_builtin=True,
    ),
    "FieldInfo": value.CaMeLClass(
        "FieldInfo",
        pydantic.fields.FieldInfo,
        Capabilities.camel(),
        (),
        {},
        is_builtin=True,
    ),
    "EmailStr": value.CaMeLClass(
        "EmailStr",
        pydantic.EmailStr,  # type: ignore
        Capabilities.camel(),
        (),
        {},
        is_builtin=True,
    ),
    "NaiveDatetime": value.CaMeLClass(
        "NaiveDatetime",
        pydantic.NaiveDatetime,  # type: ignore
        Capabilities.camel(),
        (),
        {},
        is_builtin=True,
    ),
}

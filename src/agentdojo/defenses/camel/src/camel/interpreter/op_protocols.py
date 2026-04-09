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

from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from camel.interpreter.value import CaMeLValue

_RT = TypeVar("_RT", bound="CaMeLValue", covariant=True)


@runtime_checkable
class SupportsAdd(Generic[_RT], Protocol):
    def add(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsSub(Generic[_RT], Protocol):
    def sub(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsMult(Generic[_RT], Protocol):
    def mult(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsTrueDiv(Generic[_RT], Protocol):
    def truediv(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsFloorDiv(Generic[_RT], Protocol):
    def floor_div(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsMod(Generic[_RT], Protocol):
    def mod(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsPow(Generic[_RT], Protocol):
    def pow(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsLShift(Generic[_RT], Protocol):
    def l_shift(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsRShift(Generic[_RT], Protocol):
    def r_shift(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsBitOr(Generic[_RT], Protocol):
    def bit_or(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsBitXor(Generic[_RT], Protocol):
    def bit_xor(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsBitAnd(Generic[_RT], Protocol):
    def bit_and(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsRAdd(Generic[_RT], Protocol):
    def r_add(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsRSub(Generic[_RT], Protocol):
    def r_sub(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsRMult(Generic[_RT], Protocol):
    def r_mult(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsRTrueDiv(Generic[_RT], Protocol):
    def r_truediv(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsRFloorDiv(Generic[_RT], Protocol):
    def r_floor_div(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsRMod(Generic[_RT], Protocol):
    def r_mod(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsRPow(Generic[_RT], Protocol):
    def r_pow(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsRLShift(Generic[_RT], Protocol):
    def r_l_shift(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsRRShift(Generic[_RT], Protocol):
    def r_r_shift(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsRBitOr(Generic[_RT], Protocol):
    def r_bit_or(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsRBitXor(Generic[_RT], Protocol):
    def r_bit_xor(self, other: "CaMeLValue") -> _RT: ...


@runtime_checkable
class SupportsRBitAnd(Generic[_RT], Protocol):
    def r_bit_and(self, other: "CaMeLValue") -> _RT: ...

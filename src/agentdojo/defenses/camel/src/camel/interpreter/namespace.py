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

import dataclasses
import functools

from typing_extensions import Self

from camel.interpreter import library, value


@dataclasses.dataclass(frozen=True)
class Namespace:
    variables: dict[str, value.CaMeLValue] = dataclasses.field(default_factory=dict)

    @classmethod
    def with_builtins(cls, variables: dict[str, value.CaMeLValue] | None = None) -> Self:
        return cls(variables=library.BUILT_IN_FUNCTIONS | library.BUILT_IN_CLASSES | (variables or {}))

    def add_variables(self, variables: dict[str, value.CaMeLValue]) -> Self:
        """Creates a copy of this adding the variables passed as argument."""
        return dataclasses.replace(self, variables=self.variables | variables)

    @functools.lru_cache
    def _all_namespace_elements(
        self,
    ) -> dict[str, value.CaMeLValue]:
        return self.variables

    def set_variable(self, name: str, value: value.CaMeLValue) -> None:
        self.variables[name] = value

    def get(self, name: str) -> value.CaMeLValue | None:
        return self.variables.get(name)

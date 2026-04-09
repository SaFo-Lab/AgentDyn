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

"""Module containing definition for data sources."""

import dataclasses
from enum import Enum, auto
from typing import TypeAlias


class SourceEnum(Enum):
    CaMeL = auto()
    User = auto()
    Assistant = auto()
    TrustedToolSource = auto()


@dataclasses.dataclass(frozen=True)
class Tool:
    """Tool source."""

    tool_name: str
    """Name of the tool."""
    inner_sources: frozenset[str | SourceEnum] = dataclasses.field(default_factory=frozenset)
    """Sources within the tool (e.g., email addresses)."""

    def __hash__(self) -> int:
        return hash(self.tool_name) ^ hash(tuple(self.inner_sources))


Source: TypeAlias = SourceEnum | Tool

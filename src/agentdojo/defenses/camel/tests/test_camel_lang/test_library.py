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

from camel.interpreter.library import camel_range


def test_range_only_start():
    res = camel_range(3)
    assert res == [0, 1, 2]


def test_range_start_end():
    res = camel_range(1, 4)
    assert res == [1, 2, 3]


def test_range_start_end_step():
    res = camel_range(0, 10, 2)
    assert res == [0, 2, 4, 6, 8]

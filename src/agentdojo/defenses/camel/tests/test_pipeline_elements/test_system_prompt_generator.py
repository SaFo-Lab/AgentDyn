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

import enum

import pydantic
from agentdojo import functions_runtime

from camel import system_prompt_generator


def test_function_to_python_definition():
    def sum(a: int, b: int) -> int:
        """Sums two integers.

        Args:
            a: the first integer
            b: the second integer

        Returns:
            the sum of the two integers
        """
        return a + b

    function = functions_runtime.make_function(sum)
    python_definition = system_prompt_generator.function_to_python_definition(function)

    expected_function = """
def sum(a: int, b: int) -> int:
    \"\"\"Sums two integers.

    Args:
        a: the first integer
        b: the second integer

    Returns:
        the sum of the two integers
    \"\"\"
    ...
  """

    assert python_definition.strip() == expected_function.strip()


def test_function_to_python_definition_generic_classes():
    def sum(a: list[int], b: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, list[float | None]]]:
        """Sums two integers.

        Args:
            a: the first integer
            b: the second integer

        Returns:
            the sum of the two integers
        """
        ...

    function = functions_runtime.make_function(sum)

    python_definition = system_prompt_generator.function_to_python_definition(function)

    expected_function = """
def sum(a: list[int], b: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, list[float | None]]]:
    \"\"\"Sums two integers.

    Args:
        a: the first integer
        b: the second integer

    Returns:
        the sum of the two integers
    \"\"\"
    ...
    """

    assert python_definition.strip() == expected_function.strip()


def test_function_to_python_definition_special_class():
    class A(pydantic.BaseModel): ...

    def sum(a: int, b: A) -> int:
        """Sums two integers.

        Args:
            a: the first integer
            b: the second integer

        Returns:
            the sum of the two integers
        """
        return a

    function = functions_runtime.make_function(sum)
    python_definition = system_prompt_generator.function_to_python_definition(function)

    expected_function = """
def sum(a: int, b: A) -> int:
    \"\"\"Sums two integers.

    Args:
        a: the first integer
        b: the second integer

    Returns:
        the sum of the two integers
    \"\"\"
    ...
  """

    assert python_definition.strip() == expected_function.strip()


def test_function_to_python_definition_no_return_type():
    def sum(a: int, b: int):
        """Sums two integers.

        Args:
            a: the first integer
            b: the second integer

        Returns:
            the sum of the two integers
        """
        return a + b

    function = functions_runtime.make_function(sum)
    python_definition = system_prompt_generator.function_to_python_definition(function)

    expected_function = """
def sum(a: int, b: int) -> Any:
    \"\"\"Sums two integers.

    Args:
        a: the first integer
        b: the second integer

    Returns:
        the sum of the two integers
    \"\"\"
    ...
  """

    assert python_definition.strip() == expected_function.strip()


def test_get_pydantic_model_code():
    class TestModelA(pydantic.BaseModel):
        a: int
        b: float = pydantic.Field(..., title="Some field")

    code, _ = system_prompt_generator._get_code_and_dependencies(TestModelA)
    expected = """\
class TestModelA(BaseModel):
    a: int = Field()
    b: float = Field(title='Some field')"""
    assert code == expected


def test_get_pydantic_model_code_metadata():
    class TestModelA(pydantic.BaseModel):
        a: int
        b: float = pydantic.Field(title="A number", gt=4)
        c: str = pydantic.Field(title="A string", pattern=".*")

    code, _ = system_prompt_generator._get_code_and_dependencies(TestModelA)
    expected = """\
class TestModelA(BaseModel):
    a: int = Field()
    b: float = Field(title='A number', gt=4)
    c: str = Field(title='A string', pattern='.*')"""
    assert code == expected


def test_get_enum_model_code():
    class TestEnumA(enum.Enum):
        a = 1
        b = 2

    code, _ = system_prompt_generator._get_code_and_dependencies(TestEnumA)
    expected = """\
class TestEnumA(enum.Enum):
    a = 1
    b = 2"""
    assert code == expected


def test_get_pydantic_model_code_enum_basemodel_dependency():
    class TestEnumA(enum.Enum):
        a = 1
        b = 2

    class TestModelC(pydantic.BaseModel):
        a: int

    class TestModelD(pydantic.BaseModel):
        a: int

    class TestModelA(pydantic.BaseModel):
        a: int
        b: float = pydantic.Field(..., title="Some field")
        c: TestEnumA

    class TestModelB(pydantic.BaseModel):
        a: int
        b: float = pydantic.Field(..., title="Some field")
        c: TestEnumA
        d: dict[TestModelA, TestModelC | TestModelD]

    code, dependencies = system_prompt_generator._get_code_and_dependencies(TestModelB)
    expected = """\
class TestModelB(BaseModel):
    a: int = Field()
    b: float = Field(title='Some field')
    c: TestEnumA = Field()
    d: dict[TestModelA, TestModelC | TestModelD] = Field()"""
    assert code == expected
    assert dependencies == {TestEnumA, TestModelA, TestModelC, TestModelD}


def test_get_code_recursive_simple():
    class SimpleEnum(enum.Enum):
        A = "a"
        B = "b"

    class SimpleModel(pydantic.BaseModel):
        name: str
        value: SimpleEnum

    expected_code = {
        "SimpleModel": """\
class SimpleModel(BaseModel):
    name: str = Field()
    value: SimpleEnum = Field()""",
        "SimpleEnum": """\
class SimpleEnum(enum.Enum):
    A = 'a'
    B = 'b'""",
    }

    code = system_prompt_generator.get_code_recursive(SimpleModel)
    # Normalize code string by removing leading/trailing spaces and new lines
    for name in code:
        code[name] = code[name].strip()
    assert code == expected_code


def test_get_code_recursive_complex():
    class NestedEnum(enum.Enum):
        X = "x"
        Y = "y"

    class NestedModel(pydantic.BaseModel):
        nested_value: NestedEnum

    class ComplexModel(pydantic.BaseModel):
        name: str
        nested: NestedModel
        optional_nested: NestedModel | None
        list_of_nested: list[NestedModel]

    code = system_prompt_generator.get_code_recursive(ComplexModel)
    assert len(code) == 3
    assert "ComplexModel" in code
    assert "NestedModel" in code
    assert "NestedEnum" in code


def test_get_code_recursive_circular():
    class ModelA(pydantic.BaseModel):
        b: "ModelB"  # Note the string referencex here

    class ModelB(pydantic.BaseModel):
        a: ModelA

    code = system_prompt_generator.get_code_recursive(ModelA)
    assert len(code) == 2
    assert "ModelA" in code
    assert "ModelB" in code


MyStr = str


def test_get_code_recursive_with_type_alias():
    class ModelWithAlias(pydantic.BaseModel):
        s: MyStr
        i: int

    code = system_prompt_generator.get_code_recursive(ModelWithAlias)
    assert code == {
        "ModelWithAlias": """\
class ModelWithAlias(BaseModel):
    s: str = Field()
    i: int = Field()"""
    }

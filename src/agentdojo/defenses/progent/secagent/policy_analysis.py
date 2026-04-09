#!/usr/bin/env python3
from z3 import *
from role_analyzer import regex_to_z3_expr
import sre_parse

# Define a global Z3 datatype for null.
NullType = Datatype('NullType')
NullType.declare('null')
NullType = NullType.create()

# Define a Z3 datatype for arrays.
# ArrayWrapper has two fields:
#   a: an array from Int to Type
#   len: an integer representing the length of the array.

# Array wrapper for arrays of strings.
ArrayWrapper_string = Datatype('ArrayWrapper_string')
ArrayWrapper_string.declare('mkArrayWrapper', ('a', ArraySort(IntSort(), StringSort())), ('len', IntSort()))
ArrayWrapper_string = ArrayWrapper_string.create()

# Array wrapper for arrays of integers.
ArrayWrapper_integer = Datatype('ArrayWrapper_integer')
ArrayWrapper_integer.declare('mkArrayWrapper', ('a', ArraySort(IntSort(), IntSort())), ('len', IntSort()))
ArrayWrapper_integer = ArrayWrapper_integer.create()

# Array wrapper for arrays of reals (numbers).
ArrayWrapper_number = Datatype('ArrayWrapper_number')
ArrayWrapper_number.declare('mkArrayWrapper', ('a', ArraySort(IntSort(), RealSort())), ('len', IntSort()))
ArrayWrapper_number = ArrayWrapper_number.create()

# Array wrapper for arrays of booleans.
ArrayWrapper_boolean = Datatype('ArrayWrapper_boolean')
ArrayWrapper_boolean.declare('mkArrayWrapper', ('a', ArraySort(IntSort(), BoolSort())), ('len', IntSort()))
ArrayWrapper_boolean = ArrayWrapper_boolean.create()

# Array wrapper for arrays of null.
ArrayWrapper_null = Datatype('ArrayWrapper_null')
ArrayWrapper_null.declare('mkArrayWrapper', ('a', ArraySort(IntSort(), NullType)), ('len', IntSort()))
ArrayWrapper_null = ArrayWrapper_null.create()

# A list of all array wrappers.
ARRAY_WRAPPERS = [ArrayWrapper_string, ArrayWrapper_integer, ArrayWrapper_number, ArrayWrapper_boolean, ArrayWrapper_null]

# Map each array wrapper sort to its accessors (the 'a' field and the 'len' field).
ARRAY_WRAPPER_ACCESSORS = {
    ArrayWrapper_string: (ArrayWrapper_string.a, ArrayWrapper_string.len),
    ArrayWrapper_integer: (ArrayWrapper_integer.a, ArrayWrapper_integer.len),
    ArrayWrapper_number: (ArrayWrapper_number.a, ArrayWrapper_number.len),
    ArrayWrapper_boolean: (ArrayWrapper_boolean.a, ArrayWrapper_boolean.len),
    ArrayWrapper_null: (ArrayWrapper_null.a, ArrayWrapper_null.len)
}


def make_array_from_list(lst, ArrayWrapper):
    """
    Given a Python list and an element type (one of "string", "integer", "number", "boolean", "null"),
    return a Z3 array (of type Array(Int, T)) mapping indices 0..len(lst)-1 to the corresponding values.
    """
    if ArrayWrapper == ArrayWrapper_string:
        default_val = StringVal("")
        def convert(x): return StringVal(x)
    elif ArrayWrapper == ArrayWrapper_integer:
        default_val = IntVal(0)
        def convert(x): return IntVal(x)
    elif ArrayWrapper == ArrayWrapper_number:
        default_val = RealVal(0)
        def convert(x): return RealVal(x)
    elif ArrayWrapper == ArrayWrapper_boolean:
        default_val = BoolVal(False)
        def convert(x): return BoolVal(x)
    elif ArrayWrapper == ArrayWrapper_null:
        default_val = NullType.null
        def convert(x): return NullType.null
    else:
        raise NotImplementedError
    default_arr = K(IntSort(), default_val)  # default value
    arr_term = default_arr
    for i, elem in enumerate(lst):
        arr_term = Store(arr_term, i, convert(elem))
    return arr_term


def build_constraints(var, schema):
    """
    Recursively build Z3 constraints from a JSON Schema.
    Supported keywords include:
      - "enum"
      - "anyOf", "allOf", "oneOf"
      - "not"
      - "if", "then", "else"
      - Type-specific keywords:
            * string: {"pattern", "minLength", "maxLength"}
            * number: {"multipleOf", "minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"}
            * integer: {"multipleOf", "minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"}
            * array: {"items", "minItems", "maxItems"}
            * boolean, null
    """
    constraint = True

    # Global enum constraint.
    if "enum" in schema:
        enums = schema["enum"]
        if enums:
            first = enums[0]
            if isinstance(first, list):
                # For arrays, convert each Python list to an ArrayWrapper instance.
                enum_consts = []
                ArrayWrapper = var.sort()
                for arr in enums:
                    arr_term = make_array_from_list(arr, ArrayWrapper)
                    array_value = ArrayWrapper.mkArrayWrapper(arr_term, len(arr))
                    enum_consts.append(var == array_value)
                constraint = And(constraint, Or(enum_consts))
            elif isinstance(first, str):
                constraint = And(constraint, Or([var == StringVal(e) for e in enums]))
            elif isinstance(first, int):
                constraint = And(constraint, Or([var == IntVal(e) for e in enums]))
            elif isinstance(first, float):
                constraint = And(constraint, Or([var == RealVal(e) for e in enums]))
            elif isinstance(first, bool):
                constraint = And(constraint, Or([var == BoolVal(e) for e in enums]))
            elif first is None:
                constraint = And(constraint, var == NullType.null)

    # anyOf: disjunction of subschemas.
    if "anyOf" in schema:
        branch_constraints = [build_constraints(var, subschema) for subschema in schema["anyOf"]]
        constraint = And(constraint, Or(branch_constraints))

    # allOf: all subschemas must hold.
    if "allOf" in schema:
        all_constraints = [build_constraints(var, subschema) for subschema in schema["allOf"]]
        constraint = And(constraint, And(all_constraints))

    # oneOf: exactly one subschema must hold.
    if "oneOf" in schema:
        oneof_constraints = [build_constraints(var, subschema) for subschema in schema["oneOf"]]
        constraint = And(constraint, Sum([If(c, 1, 0) for c in oneof_constraints]) == 1)

    # not: the subschema must not hold.
    if "not" in schema:
        constraint = And(constraint, Not(build_constraints(var, schema["not"])))

    # if/then/else: conditional constraints.
    if "if" in schema:
        if_constraint = build_constraints(var, schema["if"])
        if "then" in schema:
            then_constraint = build_constraints(var, schema["then"])
            constraint = And(constraint, Implies(if_constraint, then_constraint))
        if "else" in schema:
            else_constraint = build_constraints(var, schema["else"])
            constraint = And(constraint, Implies(Not(if_constraint), else_constraint))

    # Type check.
    if "type" in schema:
        t = schema["type"]
        t_list = []
        if isinstance(t, list):
            t_list = t
        else:
            t_list.append(t)
        type_constraint = []
        for t in t_list:
            if t == "string":
                type_constraint.append(var.sort() == StringSort())
            elif t == "array":
                type_constraint.append(Or([var.sort() == aw for aw in ARRAY_WRAPPERS]))
            elif t == "integer":
                type_constraint.append(var.sort() == IntSort())
            elif t == "number":
                type_constraint.append(var.sort() == RealSort())
            elif t == "boolean":
                type_constraint.append(var.sort() == BoolSort())
            elif t == "null":
                type_constraint.append(var.sort() == NullType)
        constraint = And(constraint, Or(type_constraint))

    # Type-specific constraints.
    if var.sort() == StringSort():
        if "pattern" in schema:
            pat = schema["pattern"]
            regex = regex_to_z3_expr(sre_parse.parse(pat))
            constraint = And(constraint, InRe(var, regex))
        if "minLength" in schema:
            constraint = And(constraint, Length(var) >= schema["minLength"])
        if "maxLength" in schema:
            constraint = And(constraint, Length(var) <= schema["maxLength"])
    elif var.sort() in ARRAY_WRAPPERS:
        a_accessor, len_accessor = ARRAY_WRAPPER_ACCESSORS[var.sort()]
        constraint = And(constraint, len_accessor(var) >= 0)
        if "minItems" in schema:
            constraint = And(constraint, len_accessor(var) >= schema["minItems"])
        if "maxItems" in schema:
            constraint = And(constraint, len_accessor(var) <= schema["maxItems"])
        if "items" in schema:
            items_schema = schema["items"]
            i = Int('i')
            constraint = And(constraint, ForAll(i, Implies(And(i >= 0, i < len_accessor(var)),
                                                           build_constraints(Select(a_accessor(var), i), items_schema))))
    elif var.sort() == IntSort():
        if "multipleOf" in schema:
            m = schema["multipleOf"]
            constraint = And(constraint, var % m == 0)
        if "exclusiveMinimum" in schema:
            constraint = And(constraint, var > schema["exclusiveMinimum"])
        elif "minimum" in schema:
            constraint = And(constraint, var >= schema["minimum"])
        if "exclusiveMaximum" in schema:
            constraint = And(constraint, var < schema["exclusiveMaximum"])
        elif "maximum" in schema:
            constraint = And(constraint, var <= schema["maximum"])
    elif var.sort() == RealSort():
        if "multipleOf" in schema:
            m = schema["multipleOf"]
            k = Int('k_mult')
            constraint = And(constraint, Exists(k, var == RealVal(m) * k))
        if "exclusiveMinimum" in schema:
            constraint = And(constraint, var > schema["exclusiveMinimum"])
        elif "minimum" in schema:
            constraint = And(constraint, var >= schema["minimum"])
        if "exclusiveMaximum" in schema:
            constraint = And(constraint, var < schema["exclusiveMaximum"])
        elif "maximum" in schema:
            constraint = And(constraint, var <= schema["maximum"])
    elif var.sort() == BoolSort():
        pass
    elif var.sort() == NullType:
        constraint = And(constraint, var == NullType.null)

    return constraint


def solve_schema(schema):
    """
    Given a JSON Schema (as a dict), determine if there is a value that satisfies it.
    Chooses a top-level Z3 variable based on the schema's "type" or by inferring from an "enum".
    Returns a tuple (model, var) if satisfiable, or (None, var) if unsat.
    """
    var_list = []
    # Determine the top-level type.
    if "type" in schema:
        t = schema["type"]
        t_list = []
        if isinstance(t, list):
            t_list = t
        else:
            t_list.append(t)
        for t in t_list:
            if t == "string":
                var_list.append(String('x'))
            elif t == "array":
                # Choose an array wrapper based on the "items" type if provided.
                if "items" in schema and "type" in schema["items"]:
                    item_type = schema["items"]["type"]
                    if item_type == "string":
                        var_list.append(Const('x', ArrayWrapper_string))
                    elif item_type == "integer":
                        var_list.append(Const('x', ArrayWrapper_integer))
                    elif item_type == "number":
                        var_list.append(Const('x', ArrayWrapper_number))
                    elif item_type == "boolean":
                        var_list.append(Const('x', ArrayWrapper_boolean))
                    elif item_type == "null":
                        var_list.append(Const('x', ArrayWrapper_null))
                    else:
                        for aw in ARRAY_WRAPPERS:
                            var_list.append(Const('x', aw))
                else:
                    for aw in ARRAY_WRAPPERS:
                        var_list.append(Const('x', aw))
            elif t == "integer":
                var_list.append(Int('x'))
            elif t == "number":
                var_list.append(Real('x'))
            elif t == "boolean":
                var_list.append(Bool('x'))
            elif t == "null":
                var_list.append(Const('x', NullType))
    else:
        # Infer type from enum if provided.
        if "enum" in schema and schema["enum"]:
            first = schema["enum"][0]
            if isinstance(first, list):
                # Infer element type.
                if len(first) > 0:
                    sample = first[0]
                    if isinstance(sample, str):
                        var_list.append(Const('x', ArrayWrapper_string))
                    elif isinstance(sample, int):
                        var_list.append(Const('x', ArrayWrapper_integer))
                    elif isinstance(sample, float):
                        var_list.append(Const('x', ArrayWrapper_number))
                    elif isinstance(sample, bool):
                        var_list.append(Const('x', ArrayWrapper_boolean))
                    elif sample is None:
                        var_list.append(Const('x', ArrayWrapper_null))
                    else:
                        for aw in ARRAY_WRAPPERS:
                            var_list.append(Const('x', aw))
                else:
                    for aw in ARRAY_WRAPPERS:
                        var_list.append(Const('x', aw))
            elif isinstance(first, str):
                var_list.append(String('x'))
            elif isinstance(first, int):
                var_list.append(Int('x'))
            elif isinstance(first, float):
                var_list.append(Real('x'))
            elif isinstance(first, bool):
                var_list.append(Bool('x'))
            elif first is None:
                var_list.append(Const('x', NullType))
        else:
            var_list = [String('x'), Int('x'), Real('x'), Bool('x'), Const('x', NullType)]
            for aw in ARRAY_WRAPPERS:
                var_list.append(Const('x', aw))

    for var in var_list:
        cons = build_constraints(var, schema)
        s = Solver()
        s.add(cons)
        if s.check() == sat:
            model = s.model()
            return model, var
    return None, var


def security_policy_analysis(security_policy):
    warnings = []
    if security_policy is None:
        return warnings
    for tool_name, policies in security_policy.items():
        n = len(policies)
        for i in range(n):
            for j in range(i+1, n):
                for arg_name, restriction_1 in policies[i][2].items():
                    if arg_name in policies[j][2]:
                        restriction_2 = policies[j][2][arg_name]
                        # print(restriction_1, restriction_2)
                        if isinstance(restriction_1, dict) and isinstance(restriction_2, dict):  # json schema
                            schema = {
                                "allOf": [
                                    restriction_1,
                                    restriction_2
                                ]
                            }
                            model, var = solve_schema(schema)
                            if model:
                                warnings.append(f"Policy Warning: {restriction_1} and {restriction_2} overlap. Possible value: {model[var]}.")
    return warnings


if __name__ == "__main__":
    warnings = security_policy_analysis(
        security_policy={
            "send_money": [
                (1, 0, {"recipient": {"type": "string", "pattern": "UK12345678901234567890"}}, 0),
                (2, 1, {"recipient": {"type": "string", "pattern": "SE3550000000054910000003"}}, 0),
            ],
            "send_email": [
                (1, 0, {"recipient": {"type": "string", "pattern": "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}}, 0),
                (2, 1, {"recipient": {"type": "string", "pattern": "agent@example.com"}}, 0),
            ]
        }
    )
    print(warnings)

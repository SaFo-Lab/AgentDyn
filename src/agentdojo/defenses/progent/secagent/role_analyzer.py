# https://github.com/gravitational/rbac-linter/blob/main/role_analyzer.py 
from dataclasses import dataclass
from enum import Enum
import logging
import re
import sre_constants
import sre_parse
import typing
import z3  # type: ignore


@dataclass
class EntityTypeInfo:
    """
    Information about an entity type (node, k8s cluster, etc.)

    ----------------------------------------------------------------------
    Attributes defined here:

    name: str
        The short human-readable name identifying the entity.

    labels_name: str
        The name used for the entity's labels in role YAML files.

    ----------------------------------------------------------------------
    """

    name: str
    labels_name: str


class EntityType(Enum):
    """
    An enumeration of all supported entity types. New entity types should be
    added here.
    """

    APP: EntityTypeInfo = EntityTypeInfo("app", "app_labels")
    NODE: EntityTypeInfo = EntityTypeInfo("node", "node_labels")
    K8S: EntityTypeInfo = EntityTypeInfo("k8s", "kubernetes_labels")
    DB: EntityTypeInfo = EntityTypeInfo("db", "db_labels")


def other_entity_types(entity_type: EntityType) -> list[EntityType]:
    """
    Gets an enumeration of all entity types except the given one.
    """
    return list(filter(lambda e: e != entity_type, EntityType))


@dataclass
class EntityAttributes:
    """
    The Z3 variables modeling entities, on which constraints are placed.

    ----------------------------------------------------------------------
    Attributes defined here:

    keys: z3.FuncDeclRef
        The set of all label keys which must be possessed by an entity.
        Should be a Z3 function from string to bool.

    labels: z3.FuncDeclRef
        The actual key/value labels which must be possessed by an entity.
        Should be a Z3 function from string to string.

    ----------------------------------------------------------------------
    """

    keys: z3.FuncDeclRef
    labels: z3.FuncDeclRef


class UserType(Enum):
    INTERNAL: str = "internal"
    EXTERNAL: str = "external"


def get_other_user_type(user_type: UserType) -> UserType:
    if UserType.INTERNAL == user_type:
        return UserType.EXTERNAL
    elif UserType.EXTERNAL == user_type:
        return UserType.INTERNAL
    else:
        raise ValueError(f"Invalid user type {user_type}")


def get_user_type(user_type_str: str) -> UserType:
    for user_type in UserType:
        if user_type.value == user_type_str:
            return user_type
    raise ValueError(f"Invalid user type {user_type_str}")


@dataclass
class AuthzContext:
    """
    The context for a given authorization analysis. Used to encapsulate the
    variables on which constraints are placed.

    ----------------------------------------------------------------------
    Attributes defined here:

    entities: dict[EntityType, EntityAttributes]
        A map from all entity types to their Z3 variables.

    users: dict[UserType, z3.FuncDeclRef]
        A map from all user types to their Z3 variables.
    ----------------------------------------------------------------------
    """

    entities: dict[EntityType, EntityAttributes]

    users: dict[UserType, z3.FuncDeclRef]

    def __init__(self, uninterpreted: bool):
        """
        Initializes a new instance of the AuthzContext object.

        ------------------------------------------------------------------
        Parameters:

        uninterpreted: bool
            If true, construct the Z3 variables as uninterpreted functions
            on which constraints are placed; if false, construct the Z3
            variables as functions with actual definitions (to be provided
            later) against which constraints are checked. Generally
            uninterpreted functions are used when comparing two roles in
            the abstract and defined functions are used when determining
            whether a user has access to a resource through a role.
        """
        z3_func_constructor = z3.Function if uninterpreted else z3.RecFunction
        self.entities = {}
        self.users = {}
        for entity_type in EntityType:
            self.entities[entity_type] = EntityAttributes(
                z3_func_constructor(
                    f"{entity_type.value.name}_attribute_keys",
                    z3.StringSort(),
                    z3.BoolSort(),
                ),
                z3_func_constructor(
                    f"{entity_type.value.name}_attribute_labels",
                    z3.StringSort(),
                    z3.StringSort(),
                ),
            )
        for user_type in UserType:
            self.users[user_type] = z3_func_constructor(
                f"{user_type.value}_traits",
                z3.StringSort(),
                z3.StringSort(),
                z3.BoolSort(),
            )


@dataclass
class AnyValueConstraint:
    value: str


@dataclass
class StringConstraint:
    value: str


@dataclass
class RegexConstraint:
    regex: sre_parse.SubPattern


@dataclass
class UserTraitConstraint:
    trait_type: UserType
    trait_key: str
    inner_trait_key: str


@dataclass
class InterpolationConstraint:
    prefix: str
    trait_type: UserType
    trait_key: str
    inner_trait_key: str
    suffix: str


@dataclass
class EmailFunctionConstraint:
    trait_type: UserType
    trait_key: str
    inner_trait_key: str


@dataclass
class RegexReplaceFunctionConstraint:
    trait_type: UserType
    trait_key: str
    inner_trait_key: str
    pattern: str
    replace: str


def try_parse_regex(value: str) -> typing.Optional[RegexConstraint]:
    """
    Attempts to parse the given value as a regex.
    """
    try:
        parsed_regex = sre_parse.parse(value)
        is_regex = any(
            [sre_constants.LITERAL != node_type for node_type, _ in parsed_regex.data]
        )
        return RegexConstraint(parsed_regex) if is_regex else None
    except Exception as e:
        logging.debug(f"Cannot parse regex {value} - {e}")
        return None


# Regex pattern for {{internal.logins}} or {{external.email}} type template values.
template_value_pattern = re.compile(
    r'\{\{(?P<type>internal|external)\.(?P<key>[\w]+)(\["(?P<inner_key>[\w]+)"\])?\}\}'
)


def try_parse_template(value: str) -> typing.Optional[UserTraitConstraint]:
    """
    Attempts to parse template constraints of type {{internal.logins}}
    """
    match = template_value_pattern.match(value)
    if isinstance(match, re.Match):
        user_type = get_user_type(match.group("type"))
        trait_key = match.group("key")
        inner_trait_key = match.group("inner_key")
        return UserTraitConstraint(user_type, trait_key, inner_trait_key)
    else:
        return None


# Regex pattern for IAM#{{internal.logins}}#user type interpolation values.
interpolation_value_pattern = re.compile(
    r'(?P<prefix>.*)\{\{(?P<type>internal|external)\.(?P<key>[\w]+)(\["(?P<inner_key>[\w]+)"\])?\}\}(?P<suffix>.*)'
)


def try_parse_interpolation(value: str) -> typing.Optional[InterpolationConstraint]:
    """
    Attempts to parse interpolation constraints of type IAM#{external.foo}
    """
    match = interpolation_value_pattern.match(value)
    if isinstance(match, re.Match):
        prefix = match.group("prefix")
        user_type = get_user_type(match.group("type"))
        trait_key = match.group("key")
        inner_trait_key = match.group("inner_key")
        suffix = match.group("suffix")
        return InterpolationConstraint(
            prefix, user_type, trait_key, inner_trait_key, suffix
        )
    else:
        return None


# Regex pattern for {{email.local(external.email)}}
email_function_value_pattern = re.compile(
    r'\{\{email\.local\([\s]*(?P<type>internal|external)\.(?P<key>[\w]+)(\["(?P<inner_key>[\w]+)"\])?[\s]*\)\}\}'
)


def try_parse_email_function(value: str) -> typing.Optional[EmailFunctionConstraint]:
    """
    Attempts to parse email function constraints of type
    {{email.local(external.email)}}
    """
    match = email_function_value_pattern.match(value)
    if isinstance(match, re.Match):
        user_type = get_user_type(match.group("type"))
        trait_key = match.group("key")
        inner_trait_key = match.group("inner_key")
        return EmailFunctionConstraint(user_type, trait_key, inner_trait_key)
    else:
        return None


# Regex pattern for {{regexp.replace(external.access["env"], "^(staging)$", "$1")}}
regex_function_value_pattern = re.compile(
    r'\{\{regexp\.replace\([\s]*(?P<type>internal|external)\.(?P<key>[\w]+)(\["(?P<inner_key>[\w]+)"\])?[\s]*,[\s]*"(?P<pattern>.*)"[\s]*,[\s]*"(?P<replace>.*)"[\s]*\)\}\}'
)


def try_parse_regexp_replace_function(
    value: str,
) -> typing.Optional[RegexReplaceFunctionConstraint]:
    """
    Attempts to parse regexp replace function constraints of type
    {{regexp.replace(external.access, "foo", "bar")}}
    """
    match = regex_function_value_pattern.match(value)
    if isinstance(match, re.Match):
        user_type = get_user_type(match.group("type"))
        trait_key = match.group("key")
        inner_trait_key = match.group("inner_key")
        pattern = match.group("pattern")
        replace = match.group("replace")
        return RegexReplaceFunctionConstraint(
            user_type, trait_key, inner_trait_key, pattern, replace
        )
    else:
        return None


def requires_user_traits(values: typing.Union[str, list[str]]) -> bool:
    """
    Determines whether the given constraint requires user traits to specify.
    """
    if not isinstance(values, list):
        values = [values]
    for value in values:
        is_template = try_parse_template(value) != None
        is_interpolation = try_parse_interpolation(value) != None
        is_email_function = try_parse_email_function(value) != None
        is_regexp_replace_function = try_parse_regexp_replace_function(value) != None
        if (
            is_template
            or is_interpolation
            or is_email_function
            or is_regexp_replace_function
        ):
            return True
    return False


def parse_constraint(
    value: str,
) -> typing.Union[
    AnyValueConstraint,
    StringConstraint,
    RegexConstraint,
    UserTraitConstraint,
    InterpolationConstraint,
    EmailFunctionConstraint,
    RegexReplaceFunctionConstraint,
]:
    """
    Determines the category of the constraint value and parses it appropriately.
    """

    if "*" == value:
        return AnyValueConstraint(value)

    parsed_trait_constraint = try_parse_template(value)
    if isinstance(parsed_trait_constraint, UserTraitConstraint):
        return parsed_trait_constraint

    parsed_interpolation_constraint = try_parse_interpolation(value)
    if isinstance(parsed_interpolation_constraint, InterpolationConstraint):
        return parsed_interpolation_constraint

    parsed_email_constraint = try_parse_email_function(value)
    if isinstance(parsed_email_constraint, EmailFunctionConstraint):
        return parsed_email_constraint

    parsed_regex_function_constraint = try_parse_regexp_replace_function(value)
    if isinstance(parsed_regex_function_constraint, RegexReplaceFunctionConstraint):
        return parsed_regex_function_constraint

    parsed_regex_constraint = try_parse_regex(value)
    if isinstance(parsed_regex_constraint, RegexConstraint):
        return parsed_regex_constraint

    return StringConstraint(value)


def Minus(re1: z3.ReRef, re2: z3.ReRef) -> z3.ReRef:
    """
    The Z3 regex matching all strings accepted by re1 but not re2.
    Formatted in camelcase to mimic Z3 regex API.
    """
    return z3.Intersect(re1, z3.Complement(re2))


def AnyChar() -> z3.ReRef:
    """
    The Z3 regex matching any character (currently only ASCII supported).
    Formatted in camelcase to mimic Z3 regex API.
    """
    return z3.Range(chr(0), chr(127))
    # return z3.AllChar(z3.StringSort())


def category_regex(category: sre_constants._NamedIntConstant) -> z3.ReRef:
    """
    Defines regex categories in Z3.
    """
    if sre_constants.CATEGORY_DIGIT == category:
        return z3.Range("0", "9")
    elif sre_constants.CATEGORY_SPACE == category:
        return z3.Union(
            z3.Re(" "), z3.Re("\t"), z3.Re("\n"), z3.Re("\r"), z3.Re("\f"), z3.Re("\v")
        )
    elif sre_constants.CATEGORY_WORD == category:
        return z3.Union(
            z3.Range("a", "z"), z3.Range("A", "Z"), z3.Range("0", "9"), z3.Re("_")
        )
    else:
        raise NotImplementedError(
            f"ERROR: regex category {category} not yet implemented"
        )


def regex_construct_to_z3_expr(regex_construct) -> z3.ReRef:
    """
    Translates a specific regex construct into its Z3 equivalent.
    """
    node_type, node_value = regex_construct
    if sre_constants.LITERAL == node_type:  # a
        return z3.Re(chr(node_value))
    if sre_constants.NOT_LITERAL == node_type:  # [^a]
        return Minus(AnyChar(), z3.Re(chr(node_value)))
    if sre_constants.SUBPATTERN == node_type:
        _, _, _, value = node_value
        return regex_to_z3_expr(value)
    elif sre_constants.ANY == node_type:  # .
        return AnyChar()
    elif sre_constants.MAX_REPEAT == node_type:
        low, high, value = node_value
        if (0, 1) == (low, high):  # a?
            return z3.Option(regex_to_z3_expr(value))
        elif (0, sre_constants.MAXREPEAT) == (low, high):  # a*
            return z3.Star(regex_to_z3_expr(value))
        elif (1, sre_constants.MAXREPEAT) == (low, high):  # a+
            return z3.Plus(regex_to_z3_expr(value))
        else:  # a{3,5}, a{3}
            return z3.Loop(regex_to_z3_expr(value), low, high)
    elif sre_constants.IN == node_type:  # [abc]
        first_subnode_type, _ = node_value[0]
        if sre_constants.NEGATE == first_subnode_type:  # [^abc]
            return Minus(
                AnyChar(),
                z3.Union(
                    [regex_construct_to_z3_expr(value) for value in node_value[1:]]
                ),
            )
        else:
            return z3.Union([regex_construct_to_z3_expr(value) for value in node_value])
    elif sre_constants.BRANCH == node_type:  # ab|cd
        _, value = node_value
        return z3.Union([regex_to_z3_expr(v) for v in value])
    elif sre_constants.RANGE == node_type:  # [a-z]
        low, high = node_value
        return z3.Range(chr(low), chr(high))
    elif sre_constants.CATEGORY == node_type:  # \d, \s, \w
        if sre_constants.CATEGORY_DIGIT == node_value:  # \d
            return category_regex(node_value)
        elif sre_constants.CATEGORY_NOT_DIGIT == node_value:  # \D
            return Minus(AnyChar(), category_regex(sre_constants.CATEGORY_DIGIT))
        elif sre_constants.CATEGORY_SPACE == node_value:  # \s
            return category_regex(node_value)
        elif sre_constants.CATEGORY_NOT_SPACE == node_value:  # \S
            return Minus(AnyChar(), category_regex(sre_constants.CATEGORY_SPACE))
        elif sre_constants.CATEGORY_WORD == node_value:  # \w
            return category_regex(node_value)
        elif sre_constants.CATEGORY_NOT_WORD == node_value:  # \W
            return Minus(AnyChar(), category_regex(sre_constants.CATEGORY_WORD))
        else:
            raise NotImplementedError(
                f"ERROR: regex category {node_value} not implemented"
            )
    elif sre_constants.AT == node_type:
        if node_value in {
            sre_constants.AT_BEGINNING,
            sre_constants.AT_BEGINNING_STRING,
        }:  # ^a, \A
            raise NotImplementedError(
                f"ERROR: regex position {node_value} not implemented"
            )
        elif sre_constants.AT_BOUNDARY == node_value:  # \b
            raise NotImplementedError(
                f"ERROR: regex position {node_value} not implemented"
            )
        elif sre_constants.AT_NON_BOUNDARY == node_value:  # \B
            raise NotImplementedError(
                f"ERROR: regex position {node_value} not implemented"
            )
        elif node_value in {
            sre_constants.AT_END,
            sre_constants.AT_END_STRING,
        }:  # a$, \Z
            raise NotImplementedError(
                f"ERROR: regex position {node_value} not implemented"
            )
        else:
            raise NotImplementedError(
                f"ERROR: regex position {node_value} not implemented"
            )
    else:
        raise NotImplementedError(
            f"ERROR: regex construct {regex_construct} not implemented"
        )


def regex_to_z3_expr(regex: sre_parse.SubPattern) -> z3.ReRef:
    """
    Translates a parsed regex into its Z3 equivalent.
    The parsed regex is a sequence of regex constructs (literals, *, +, etc.)
    """
    if 0 == len(regex.data):
        raise ValueError("ERROR: regex is empty")
    elif 1 == len(regex.data):
        return regex_construct_to_z3_expr(regex.data[0])
    else:
        return z3.Concat(
            [regex_construct_to_z3_expr(construct) for construct in regex.data]
        )


def matches_value(
    authz_context: AuthzContext, entity_type: EntityType, key: z3.SeqRef, value: str
) -> z3.BoolRef:
    """
    Constructs an expression evaluating whether a specific label constraint
    is satisfied by a given node, database, or k8s cluster.
    Example value for key : value parameters:

    'location' : 'us-east-[\\d]+'
    'owner' : {{external.email}}
    """
    labels = authz_context.entities[entity_type].labels
    constraint = parse_constraint(value)
    # 'key' : '*'
    if isinstance(constraint, AnyValueConstraint):
        return z3.BoolVal(True)
    # 'key' : 'value'
    elif isinstance(constraint, StringConstraint):
        return labels(key) == z3.StringVal(constraint.value)
    # 'key' : '(ab)*a
    elif isinstance(constraint, RegexConstraint):
        logging.debug(f"Uncompiled regex {constraint.regex}")
        regex = regex_to_z3_expr(constraint.regex)
        logging.debug(f"Compiled regex {regex}")
        return z3.InRe(labels(key), regex)
    # 'key' : '{internal.trait_key}'
    elif isinstance(constraint, UserTraitConstraint):
        logging.debug(
            f"User trait constraint of type {constraint.trait_type} on key {constraint.trait_key}[{constraint.inner_trait_key}]"
        )
        if None != constraint.inner_trait_key:
            raise NotImplementedError(f"Nested trait maps are not supported: {value}")
        user_traits = authz_context.users[constraint.trait_type]
        user_trait_key = z3.StringVal(constraint.trait_key)
        return user_traits(user_trait_key, labels(key))
    # 'key' : 'prefix#{internal.trait_key}#suffix'
    elif isinstance(constraint, InterpolationConstraint):
        logging.debug(
            f"User interpolation constraint of type {constraint.trait_type} on key {constraint.trait_key}[{constraint.inner_trait_key}] with prefix {constraint.prefix} and suffix {constraint.suffix}"
        )
        if None != constraint.inner_trait_key:
            raise NotImplementedError(f"Nested trait maps are not supported: {value}")
        prefix = z3.StringVal(constraint.prefix)
        suffix = z3.StringVal(constraint.suffix)
        user_traits = authz_context.users[constraint.trait_type]
        user_trait_key = z3.StringVal(constraint.trait_key)
        user_trait_value = z3.String(f"{constraint.trait_type}_{constraint.trait_key}")
        is_user_trait_value = user_traits(user_trait_key, user_trait_value)
        label_equals_interpolation = labels(key) == z3.Concat(
            prefix, user_trait_value, suffix
        )
        return z3.Exists(
            user_trait_value, z3.And(is_user_trait_value, label_equals_interpolation)
        )
    # 'key' : '{{email.local(external.email)}}'
    elif isinstance(constraint, EmailFunctionConstraint):
        logging.debug(
            f"Email function constraint of type {constraint.trait_type} on key {constraint.trait_key}[{constraint.inner_trait_key}]"
        )
        if None != constraint.inner_trait_key:
            raise NotImplementedError(f"Nested trait maps are not supported: {value}")
        user_traits = authz_context.users[constraint.trait_type]
        user_trait_key = z3.StringVal(constraint.trait_key)
        user_trait_value = z3.String(
            f"{constraint.trait_type}_{constraint.trait_key}_email"
        )
        is_user_trait_value = user_traits(user_trait_key, user_trait_value)
        index_end_of_local = z3.IndexOf(user_trait_value, z3.StringVal("@"))
        label_equals_email_local = labels(key) == z3.SubString(
            user_trait_value, z3.IntVal(0), index_end_of_local
        )
        return z3.Exists(
            user_trait_value, z3.And(is_user_trait_value, label_equals_email_local)
        )
    # 'key' : '{{regexp.replace(external.access["env"], "^(staging)$", "$1")}}'
    elif isinstance(constraint, RegexReplaceFunctionConstraint):
        logging.debug(
            f"Regexp replace function constraint of type {constraint.trait_type} on key {constraint.trait_key}[{constraint.inner_trait_key}], replacing {constraint.pattern} with {constraint.replace}"
        )
        raise NotImplementedError(
            f"Regexp replace function constraint not yet supported given {key} : {value}"
        )
    else:
        raise NotImplementedError(
            f"Unknown constraint value type {value}; not supported."
        )


def matches_constraint(
    authz_context: AuthzContext,
    entity_type: EntityType,
    key: str,
    value: typing.Union[str, list[str]],
) -> z3.BoolRef:
    """
    Constructs an expression evaluating whether a specific label constraint
    is satisfied by a given node, database, or k8s cluster; constraint can
    take the form of a list of permissible values.
    Example value for key : value parameters:

    'env' : ['test', 'prod']
    """
    logging.debug(f"Compiling {key} : {value} constraint")
    if "*" == key:
        if "*" == value:
            return z3.BoolVal(True)
        else:
            raise ValueError(f"Constraint of type '*' : {value} is not valid")

    key = z3.StringVal(key)
    label_keys = authz_context.entities[entity_type].keys
    if isinstance(value, list):
        return z3.And(
            label_keys(key),
            z3.Or([matches_value(authz_context, entity_type, key, v) for v in value]),
        )
    else:
        return z3.And(
            label_keys(key), matches_value(authz_context, entity_type, key, value)
        )


def matches_constraints(
    authz_context: AuthzContext,
    entity_type: EntityType,
    constraints: dict[str, typing.Union[str, list[str]]],
    constraint_fold: typing.Callable,
) -> z3.BoolRef:
    """
    Constructs an expression evaluating to whether a given set of label
    requirements are satisfied by a given node, database, or k8s cluster.
    Example value for constraints parameter:

    {'env' : ['test', 'prod'], 'location' : 'us-east-[\\d]+' }

    The constraint_fold parameter is itself a function determining how the
    sub-constraints should be combined (conjunction or disjunction).
    """
    logging.debug(f"Compiling {entity_type} constraints")
    return constraint_fold(
        [
            matches_constraint(authz_context, entity_type, key, value)
            for key, value in constraints.items()
        ]
    )


def matches_constraint_group(
    authz_context: AuthzContext,
    group: dict[str, dict[str, typing.Union[str, list[str]]]],
    constraint_fold: typing.Callable,
) -> z3.BoolRef:
    """
    Constructs an expression evaluating to whether a given constraint group
    (either Allow or Deny) matches the labels of a given node, database, or
    k8s cluster.
    Example value for group parameter:

    node_labels:
      'env' : 'test'
      'owner' : '.*@email.com'
    database_labels:
      'contains_PII' : 'no'

    The constraint_fold parameter is itself a function determining how the
    sub-constraints should be combined (conjunction or disjunction).
    """
    return z3.Or(
        [
            (entity_type_name := entity_type.value.labels_name) in group
            and matches_constraints(
                authz_context,
                entity_type,
                group[entity_type_name],
                constraint_fold,
            )
            for entity_type in EntityType
        ]
    )


def allows(authz_context: AuthzContext, role: typing.Any) -> z3.BoolRef:
    """
    Constructs an expression evaluating to whether a given role
    gives access to a specific node, database, or k8s cluster.
    Example value for role parameter:

    spec:
     allow:
       node_labels:
         'env' : 'test'
       kubernetes_labels:
         'service' : 'company_app'
     deny:
       node_labels:
         'env' : 'prod'
    """
    role_name = role["metadata"]["name"]
    logging.debug(f"Compiling role template {role_name}")
    spec = role["spec"]
    logging.debug("Compiling allow constraints")
    allow_expr = "allow" in spec and matches_constraint_group(
        authz_context, spec["allow"], z3.And
    )
    logging.debug("Compiling deny constraints")
    deny_expr = "deny" in spec and matches_constraint_group(
        authz_context, spec["deny"], z3.Or
    )
    return z3.And(allow_expr, z3.Not(deny_expr))


def is_role_template(role) -> bool:
    """
    Determines whether the given role is a role template, filled in by user
    traits.
    """
    spec = role["spec"]
    if "allow" in spec:
        allow = spec["allow"]
        groups = [
            allow[constraint_type].values()
            for constraint_type in [
                entity_type.value.labels_name for entity_type in EntityType
            ]
            if constraint_type in allow
        ]
        if any([requires_user_traits(value) for values in groups for value in values]):
            return True

    if "deny" in spec:
        deny = spec["deny"]
        groups = [
            deny[constraint_type]
            for constraint_type in [
                entity_type.value.labels_name for entity_type in EntityType
            ]
            if constraint_type in deny
        ]
        if any([requires_user_traits(value) for values in groups for value in values]):
            return True

    return False


def Case(
    key: z3.SeqRef,
    cases: list[tuple[str, typing.Any]],
    case_transform: typing.Callable[[typing.Any], z3.BoolRef],
    other: z3.BoolRef,
) -> z3.BoolRef:
    """
    Builds a case-type expression up out of a list of string tuples using Z3's
    If expression. Terminates the case expression in the other parameter.
    Transforms the case values with the case_transform parameter.
    """
    if [] == cases:
        return other
    else:
        head, *tail = cases
        if_key, then_value = head
        return z3.If(
            key == z3.StringVal(if_key),
            case_transform(then_value),
            Case(key, tail, case_transform, other),
        )


def labels_as_z3_map(
    authz_context: AuthzContext,
    concrete_labels: typing.Optional[dict[str, str]],
    entity_type: EntityType,
):
    """
    Compiles the labels of a given app, node, k8s cluster, or database into a
    form understood by Z3 that can be checked against a compiled set of role
    constraints.
    """
    logging.debug(f"Compiling labels {concrete_labels} of type {entity_type.name}")

    # Add definition of required keys function.
    required_key = z3.String(f"{entity_type.value.name}_required_key")
    z3.RecAddDefinition(
        authz_context.entities[entity_type].keys,
        [required_key],
        z3.BoolVal(False)
        if concrete_labels is None
        else z3.Or(
            [required_key == z3.StringVal(key) for key in concrete_labels.keys()]
        ),
    )

    # Add definition of required labels function.
    label_key = z3.String(f"{entity_type.value.name}_label_key")
    z3.RecAddDefinition(
        authz_context.entities[entity_type].labels,
        [label_key],
        Case(
            label_key,
            [] if concrete_labels is None else list(concrete_labels.items()),
            z3.StringVal,
            z3.Empty(z3.StringSort()),
        ),
    )

    # Specify unused entity types have no required keys and no defined labels.
    for other_entity_type in other_entity_types(entity_type):
        required_key = z3.String(f"{other_entity_type.value.name}_required_key")
        z3.RecAddDefinition(
            authz_context.entities[other_entity_type].keys,
            [required_key],
            z3.BoolVal(False),
        )
        label_key = z3.String(f"{other_entity_type.value.name}_label_key")
        z3.RecAddDefinition(
            authz_context.entities[other_entity_type].labels,
            [label_key],
            z3.Empty(z3.StringSort()),
        )


def traits_as_z3_map(
    authz_context: AuthzContext,
    concrete_traits: dict[str, list[str]],
    user_type: UserType,
):
    """
    Compiles the traits of a given internal or external user into a form
    understood by Z3 that can be checked against a compiled set of role
    constraints.
    """
    logging.debug(f"Compiling user traits {concrete_traits} of type {user_type.name}")

    # Add definition of required user traits.
    user_trait_key = z3.String(f"{user_type.value}_trait_key")
    user_trait_value = z3.String(f"{user_type.value}_trait_value")
    z3.RecAddDefinition(
        authz_context.users[user_type],
        [user_trait_key, user_trait_value],
        Case(
            user_trait_key,
            list(concrete_traits.items()),
            lambda traits: z3.Or(
                [user_trait_value == z3.StringVal(trait) for trait in traits]
            ),
            z3.BoolVal(False),
        ),
    )

    # Specify unused user type has no trait values.
    other_user_type = get_other_user_type(user_type)
    other_user_trait_key = z3.String(f"{other_user_type.value}_trait_key")
    other_user_trait_value = z3.String(f"{other_user_type.value}_trait_value")
    z3.RecAddDefinition(
        authz_context.users[other_user_type],
        [other_user_trait_key, other_user_trait_value],
        z3.BoolVal(False),
    )


def role_allows_user_access_to_entity(
    role: typing.Any,
    user_traits: typing.Optional[dict[str, list[str]]],
    user_type: typing.Optional[UserType],
    entity_labels: dict[str, str],
    entity_type: EntityType,
    solver: z3.Solver,
) -> bool:
    """
    Determines whether the given role provides the user access to the entity.
    Does not check whether the user actually possesses that role.
    """
    authz_context = AuthzContext(False)
    if user_traits is not None and user_type is not None:
        traits_as_z3_map(authz_context, user_traits, user_type)
    labels_as_z3_map(authz_context, entity_labels, entity_type)
    allows_expr = allows(authz_context, role)
    logging.debug(allows_expr)
    solver.add(allows_expr)
    result = solver.check()
    if z3.sat == result:
        logging.debug(solver.model())
        return True
    else:
        return False

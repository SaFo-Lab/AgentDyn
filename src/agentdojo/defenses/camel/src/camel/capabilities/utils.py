from typing import Protocol

from camel.capabilities import Capabilities, readers, sources
from camel.interpreter.value import CaMeLValue


class HasDependenciesAndMetadata(Protocol):
    def get_dependencies(
        self, visited_objects: frozenset[int] = frozenset()
    ) -> tuple[tuple[CaMeLValue, ...], frozenset[int]]: ...
    @property
    def metadata(self) -> Capabilities: ...


def get_all_readers(
    value: HasDependenciesAndMetadata,
    visited_objects: frozenset[int] = frozenset(),
) -> tuple[readers.Readers, frozenset[int]]:
    # Catch circular dependencies that might come up
    if id(value) in visited_objects:
        return value.metadata.readers_set, visited_objects
    value_readers = value.metadata.readers_set
    for dependency in value.get_dependencies()[0]:
        new_value_readers, visited_objects = get_all_readers(dependency, visited_objects | {id(value)})
        value_readers &= new_value_readers
    return value_readers, visited_objects | {id(value)}


def is_public(value: HasDependenciesAndMetadata):
    return isinstance(get_all_readers(value)[0], readers.Public)


def can_readers_read_value(potential_readers: set, value: CaMeLValue) -> bool:
    value_readers, _ = get_all_readers(value)
    if isinstance(value_readers, readers.Public):
        return True
    return potential_readers.issubset(value_readers)


def get_all_sources(
    value: HasDependenciesAndMetadata,
    visited_objects: frozenset[int] = frozenset(),
) -> tuple[frozenset[sources.Source], frozenset[int]]:
    # Catch circular dependencies that might come up
    if id(value) in visited_objects:
        return frozenset(), visited_objects
    value_sources: frozenset[sources.Source]
    value_sources = value.metadata.sources_set
    for dependency in value.get_dependencies()[0]:
        new_value_sources, visited_objects = get_all_sources(dependency, visited_objects | {id(value)})
        value_sources |= new_value_sources
    return value_sources, visited_objects


_TRUSTED_SET = {
    sources.SourceEnum.User,
    sources.SourceEnum.CaMeL,
    sources.SourceEnum.Assistant,
    sources.SourceEnum.TrustedToolSource,
}


def _source_is_trusted(s: sources.Source) -> bool:
    if s in _TRUSTED_SET:
        return True
    if isinstance(s, sources.Tool) and not len(s.inner_sources) == 0 and s.inner_sources.issubset(_TRUSTED_SET):
        return True
    return False


def is_trusted(value: HasDependenciesAndMetadata, trusted_set: set | None = None) -> bool:
    trusted_set = trusted_set or _TRUSTED_SET
    return all(_source_is_trusted(source) for source in get_all_sources(value)[0])

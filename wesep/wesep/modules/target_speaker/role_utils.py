from __future__ import annotations

import re


_ROLE_RE = re.compile(r"^spk([1-9][0-9]*)$")


def validate_role(role: str) -> str:
    if not isinstance(role, str):
        raise TypeError(f"target role must be str, got {type(role)!r}")
    role = role.strip()
    if not _ROLE_RE.fullmatch(role):
        raise ValueError(f"Unsupported target role: {role!r}")
    return role


def role_to_index(role: str) -> int:
    role = validate_role(role)
    return int(role[3:])


def role_from_index(index: int) -> str:
    value = int(index)
    if value <= 0:
        raise ValueError(f"role index must be positive, got {value}")
    return f"spk{value}"


def role_sort_key(role: str) -> int:
    return role_to_index(role)


def sorted_roles(roles: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    return sorted((validate_role(role) for role in roles), key=role_sort_key)


def iter_standard_roles(max_speakers: int) -> list[str]:
    count = int(max_speakers)
    if count <= 0:
        return []
    return [role_from_index(index) for index in range(1, count + 1)]

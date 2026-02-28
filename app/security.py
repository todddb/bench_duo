from __future__ import annotations

import ipaddress
import re


_HOSTNAME_PATTERN = re.compile(r"^[A-Za-z0-9.-]+$")


def sanitize_text_input(value: object, field_name: str, *, max_length: int = 8000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    cleaned = value.replace("\x00", "").strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(cleaned) > max_length:
        raise ValueError(f"{field_name} is too long")
    return cleaned


def validate_host(host: str) -> bool:
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True

    if not _HOSTNAME_PATTERN.match(host):
        return False

    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass

    if host.startswith("-") or host.endswith("-") or ".." in host:
        return False

    return True

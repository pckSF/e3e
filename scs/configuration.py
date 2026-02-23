from __future__ import annotations

import hashlib
import json
from typing import (
    Callable,
    Hashable,
    Literal,
    overload,
)

from ml_collections import config_dict


@overload
def make_config(
    config: dict[str, Hashable], frozen: Literal[True] = True
) -> config_dict.FrozenConfigDict: ...


@overload
def make_config(
    config: dict[str, Hashable], frozen: Literal[False]
) -> config_dict.ConfigDict: ...


def make_config(
    config: dict[str, Hashable], frozen: bool = True
) -> config_dict.FrozenConfigDict | config_dict.ConfigDict:
    """Creates a config dict from a built-in python dictionary.

    This function converts a standard Python dictionary into an ``ml_collections``
    config dict, which allows for attribute-style access to keys.

    Args:
        config: The input dictionary to be converted. Its keys must be strings, and
            the values can be any hashable type.
        frozen: If ``True`` (the default), creates an immutable ``FrozenConfigDict``.
            If ``False``, creates a mutable ``ConfigDict``.

    Returns:
        An instance of ``config_dict.FrozenConfigDict`` or ``config_dict.ConfigDict``.
    """
    if frozen:
        return config_dict.FrozenConfigDict(config)
    return config_dict.ConfigDict(config)


@overload
def get_config[ConfigT](
    config_file: str,
    create_config: Callable[..., ConfigT],
    *,
    seed: int | None = None,
    with_hash: Literal[False] = False,
) -> ConfigT: ...


@overload
def get_config[ConfigT](
    config_file: str,
    create_config: Callable[..., ConfigT],
    *,
    seed: int | None = None,
    with_hash: Literal[True],
) -> tuple[ConfigT, str]: ...


def get_config[ConfigT](
    config_file: str,
    create_config: Callable[..., ConfigT],
    *,
    seed: int | None = None,
    with_hash: bool = False,
) -> tuple[ConfigT, str] | ConfigT:
    """Loads a configuration from a JSON file.

    Note:
        JSON arrays (lists) in the config file are automatically converted to
        tuples by ``FrozenConfigDict`` to ensure immutability.

    Args:
        config_file: Path to the JSON configuration file.
        create_config: A callable that creates the config object from keyword
            arguments (e.g., ``create_ppo_config`` or ``create_appo_config``).
        seed: Optional seed value that overrides the seed in the JSON file.
            Useful for running multiple experiments with different seeds.
        with_hash: If ``True``, also returns an MD5 hash of the config data.

    Returns:
        The config object, or a tuple of (config, hash) if ``with_hash=True``.
    """
    with open(config_file, "r") as file:
        config_data = json.load(file)
    if seed is not None:
        config_data["seed"] = seed

    if with_hash:
        config_hash = hashlib.md5(
            json.dumps(config_data, sort_keys=True).encode()
        ).hexdigest()
        return create_config(**config_data), config_hash
    else:
        return create_config(**config_data)

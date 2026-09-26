from typing import Any, Optional, List
import argparse

import yaml

def flatten(xs: list) -> list:
    """Flattens a nested list of arbitrary depth."""
    ys = []
    for x in xs:
        if type(x) is list:
            ys = ys + flatten(x)
        else:
            ys.append(x)
    return ys


def nest_dict(x: dict) -> dict:
    y = {}

    def deep_insert(y: dict, key: Any, val: Any) -> None:
        split = key.split("/")
        if len(split) == 1:
            y[key] = val
        else:
            prefix = split[0]
            suffix = "/".join(split[1:])
            if prefix not in y:
                y[prefix] = {}
            deep_insert(y[prefix], suffix, val)

    for key, val in x.items():
        deep_insert(y, key, val)
    return y


def extract_argument_keys(cfg: dict, prefix: Optional[str] = None) -> list:
    """Extract argument keys and their default values from a given config dict."""
    if prefix is None:
        prefix = ""
    if prefix != "":
        prefix = f"{prefix}/"

    def f(key, val):
        if isinstance(val, dict):
            return extract_argument_keys(val, prefix=f"{prefix}{key}")
        else:
            return f"{prefix}{key}", val

    return [f(key, val) for key, val in cfg.items()]


def merge_dicts(*dicts: List[dict]) -> dict:
    """Merge dictionaries with ascending priority."""
    y = {}

    def deep_update(a, b):
        """Dict a is updated with the values of dict b."""
        for key, val in b.items():
            if key not in a:
                a[key] = {}
            if isinstance(val, dict):
                deep_update(a[key], val)
            else:
                a[key] = val

    for x in dicts:
        deep_update(y, x)
    return y


def get_config(path_to_default_cfg: str) -> dict:
    """
    cfg_default < cfg_special < cfg_args
    """
    # load defaul config file (default.yaml)
    cfg_default = load_yaml_config(path_to_default_cfg)

    # extract argument keys and their default values from the default config
    arguments = dict(flatten(extract_argument_keys(cfg_default)))

    # build parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="Path where the config yaml file is located.")
    for argument_key, default_value in arguments.items():
        parser.add_argument(
            f"--{argument_key}",
            help=f"Default: {default_value}",
            type=type(default_value),
        )

    # parse arguments
    args = parser.parse_args()
    cfg_args = nest_dict(
        {key: val for key, val in vars(args).items() if val is not None}
    )

    # load special config file
    path_to_special_cfg = args.cfg
    if path_to_special_cfg is not None:
        cfg_special = load_yaml_config(path_to_special_cfg)
    else:
        cfg_special = {}

    # return the final configuration
    return merge_dicts(cfg_default, cfg_special, cfg_args)

def load_yaml_config(config_filename: str) -> dict:
    """Load yaml config.

    Args:
        config_filename: Filename to config.

    Returns:
        Loaded config.
    """
    with open(config_filename) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg
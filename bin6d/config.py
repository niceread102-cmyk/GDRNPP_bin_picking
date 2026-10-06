"""YAML config with attribute access and `--set a.b.c=value` CLI overrides."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


class Cfg(dict):
    """dict with attribute access; nested dicts become Cfg."""

    def __init__(self, d: dict | None = None):
        super().__init__()
        for k, v in (d or {}).items():
            self[k] = Cfg(v) if isinstance(v, dict) else v

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Cfg) else v for k, v in self.items()}


def apply_override(cfg: Cfg, expr: str) -> None:
    key, _, raw = expr.partition("=")
    if not _:
        raise ValueError(f"override must look like a.b=value, got {expr!r}")
    *parents, leaf = key.strip().split(".")
    node = cfg
    for p in parents:
        if p not in node:
            raise KeyError(f"unknown config key {key!r}")
        node = node[p]
    if leaf not in node:
        raise KeyError(f"unknown config key {key!r}")
    value = yaml.safe_load(raw)
    node[leaf] = Cfg(value) if isinstance(value, dict) else value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Cfg:
    with open(path) as f:
        cfg = Cfg(yaml.safe_load(f))
    for o in overrides or []:
        apply_override(cfg, o)
    return cfg


def parse_args(description: str, extra: callable | None = None) -> tuple[argparse.Namespace, Cfg]:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    if extra:
        extra(ap)
    args = ap.parse_args()
    return args, load_config(args.config, args.set)


def save_config(cfg: Cfg, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

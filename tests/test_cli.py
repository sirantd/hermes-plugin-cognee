import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for flat import

from cli import register_cli, _dispatch  # noqa: E402


def _build_parser():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    cognee_parser = subparsers.add_parser("cognee")
    register_cli(cognee_parser)
    return parser


def test_status_subcommand_parses_and_routes_to_dispatch():
    parser = _build_parser()
    args = parser.parse_args(["cognee", "status"])
    assert args.cognee_command == "status"
    assert args.func is _dispatch


def test_recall_subcommand_parses_query_and_routes_to_dispatch():
    parser = _build_parser()
    args = parser.parse_args(["cognee", "recall", "what do you know about me"])
    assert args.cognee_command == "recall"
    assert args.query == "what do you know about me"
    assert args.func is _dispatch

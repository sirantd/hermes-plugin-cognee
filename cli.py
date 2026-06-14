"""CLI helpers for the cognee memory provider: `hermes cognee ...`."""

from __future__ import annotations


def register_cli(subparser) -> None:
    """Build the ``hermes cognee`` subcommand tree.

    ``subparser`` is the argparse parser for ``hermes cognee`` itself.
    """
    subs = subparser.add_subparsers(dest="cognee_command")
    subs.add_parser("status", help="Show cognee server reachability + config")
    recall = subs.add_parser("recall", help="Run a one-off cognee recall query")
    recall.add_argument("query")
    subparser.set_defaults(func=_dispatch)


def _dispatch(args) -> None:
    command = getattr(args, "cognee_command", None)
    if command == "status":
        _cmd_status(args)
    elif command == "recall":
        _cmd_recall(args)
    else:
        print("usage: hermes cognee {status,recall}")


def _build_client():
    try:
        from .client import CogneeClient, CogneeConfig
    except ImportError:
        from client import CogneeClient, CogneeConfig
    cfg = CogneeConfig.from_hermes_config()
    return CogneeClient(cfg), cfg


def _cmd_status(args=None) -> None:
    client, cfg = _build_client()
    try:
        datasets = client.list_datasets()
        print(f"cognee OK @ {cfg.base_url} — dataset={cfg.dataset} node_set={cfg.node_set} "
              f"({len(datasets)} datasets visible)")
    except Exception as exc:
        print(f"cognee UNREACHABLE @ {cfg.base_url}: {exc}")
    finally:
        client.close()


def _cmd_recall(args) -> None:
    client, cfg = _build_client()
    try:
        results = client.search(args.query, search_type=cfg.tool_search_type, top_k=10)
        for r in results:
            print("-", r)
    except Exception as exc:
        print(f"recall failed: {exc}")
    finally:
        client.close()

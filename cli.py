"""CLI helpers for the cognee memory provider: `hermes cognee ...`."""

from __future__ import annotations


def register_cli(subparser) -> None:
    sub = subparser.add_parser("cognee", help="cognee memory provider utilities")
    actions = sub.add_subparsers(dest="cognee_action")

    status = actions.add_parser("status", help="Show cognee server reachability + config")
    status.set_defaults(func=_cmd_status)

    recall = actions.add_parser("recall", help="Run a one-off cognee recall query")
    recall.add_argument("query")
    recall.set_defaults(func=_cmd_recall)


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

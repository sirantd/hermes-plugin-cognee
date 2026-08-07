"""CLI helpers for the two-tier memory provider: `hermes cognee ...`."""

from __future__ import annotations


def register_cli(subparser) -> None:
    """Build the ``hermes cognee`` subcommand tree.

    ``subparser`` is the argparse parser for ``hermes cognee`` itself.
    """
    subs = subparser.add_subparsers(dest="cognee_command")
    subs.add_parser("status", help="Show both memory tiers' reachability + config")
    recall = subs.add_parser("recall", help="Run a one-off recall across both tiers")
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


def _build_clients():
    """Both tiers, constructed independently so one being down never hides the other."""
    try:
        from .client import CogneeClient, CogneeConfig
        from .graphiti_client import GraphitiClient, GraphitiConfig
    except ImportError:
        from client import CogneeClient, CogneeConfig
        from graphiti_client import GraphitiClient, GraphitiConfig
    cognee_cfg = CogneeConfig.from_hermes_config()
    graphiti_cfg = GraphitiConfig.from_hermes_config()
    group_id = graphiti_cfg.resolve_group_id(cognee_cfg.node_set)
    return (
        CogneeClient(cognee_cfg),
        cognee_cfg,
        GraphitiClient(graphiti_cfg),
        graphiti_cfg,
        group_id,
    )


def _cmd_status(args=None) -> None:
    cognee, cognee_cfg, graphiti, graphiti_cfg, group_id = _build_clients()
    try:
        try:
            status = graphiti.status()
            print(
                f"graphiti OK @ {graphiti_cfg.base_url} — group_id={group_id} "
                f"({status.get('message', status.get('status', 'ok'))})"
            )
        except Exception as exc:
            print(f"graphiti UNREACHABLE @ {graphiti_cfg.base_url}: {exc}")
        try:
            datasets = cognee.list_datasets()
            print(
                f"cognee OK @ {cognee_cfg.base_url} — dataset={cognee_cfg.dataset} "
                f"node_set={cognee_cfg.node_set} ({len(datasets)} datasets visible)"
            )
        except Exception as exc:
            print(f"cognee UNREACHABLE @ {cognee_cfg.base_url}: {exc}")
    finally:
        cognee.close()
        graphiti.close()


def _cmd_recall(args) -> None:
    cognee, cognee_cfg, graphiti, _graphiti_cfg, group_id = _build_clients()
    try:
        print(f"# working tier (graphiti, group_id={group_id})")
        try:
            facts = graphiti.search_facts(args.query, max_facts=10, group_ids=[group_id])
            for fact in facts:
                print("-", fact.get("fact", fact))
        except Exception as exc:
            print(f"  recall failed: {exc}")
        print(f"# archive tier (cognee, dataset={cognee_cfg.dataset})")
        try:
            for result in cognee.search(args.query, search_type=cognee_cfg.tool_search_type, top_k=10):
                print("-", result)
        except Exception as exc:
            print(f"  recall failed: {exc}")
    finally:
        cognee.close()
        graphiti.close()

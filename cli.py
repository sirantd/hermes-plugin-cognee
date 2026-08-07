"""CLI helpers for the two-tier memory provider: `hermes cognee ...`."""

from __future__ import annotations

from typing import Any, NamedTuple


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


class _Tiers(NamedTuple):
    # Untyped: the clients are imported lazily so this module stays importable
    # both as a package and flat.
    working: Any          # GraphitiClient
    archive: Any          # CogneeClient
    archive_config: Any   # CogneeConfig
    group_id: str

    def close(self) -> None:
        self.working.close()
        self.archive.close()


def _build_tiers() -> _Tiers:
    """Both tiers, constructed independently so one being down never hides the other."""
    try:
        from .client import CogneeClient, CogneeConfig
        from .graphiti_client import GraphitiClient, GraphitiConfig
    except ImportError:
        from client import CogneeClient, CogneeConfig
        from graphiti_client import GraphitiClient, GraphitiConfig
    archive_config = CogneeConfig.from_hermes_config()
    working_config = GraphitiConfig.from_hermes_config()
    return _Tiers(
        working=GraphitiClient(working_config),
        archive=CogneeClient(archive_config),
        archive_config=archive_config,
        group_id=working_config.resolve_group_id(archive_config.node_set),
    )


def _report(label: str, describe) -> None:
    """Print one tier's line, turning an outage into a message rather than a traceback."""
    try:
        print(f"{label} OK — {describe()}")
    except Exception as exc:
        print(f"{label} UNREACHABLE: {exc}")


def _cmd_status(args=None) -> None:
    tiers = _build_tiers()
    try:
        def working():
            status = tiers.working.status()
            # Episode count is the cheapest answer to "is the write path
            # actually landing?", which is the thing most worth knowing here.
            episodes = tiers.working.get_episodes(max_episodes=5, group_ids=[tiers.group_id])
            detail = status.get("message") or status.get("status", "ok")
            return f"group_id={tiers.group_id} ({detail}; {len(episodes)} recent episodes)"

        def archive():
            datasets = tiers.archive.list_datasets()
            return (
                f"dataset={tiers.archive_config.dataset} "
                f"node_set={tiers.archive_config.node_set} "
                f"({len(datasets)} datasets visible)"
            )

        _report(f"working @ {tiers.working.config.base_url}", working)
        _report(f"archive @ {tiers.archive_config.base_url}", archive)
    finally:
        tiers.close()


def _cmd_recall(args) -> None:
    tiers = _build_tiers()
    try:
        def working():
            facts = tiers.working.search_facts(args.query, max_facts=10, group_ids=[tiers.group_id])
            return "\n".join(f"  - {f.get('fact', f)}" for f in facts) or "  (nothing)"

        def archive():
            results = tiers.archive.search(
                args.query, search_type=tiers.archive_config.tool_search_type, top_k=10
            )
            return "\n".join(f"  - {r}" for r in results) or "  (nothing)"

        _report(f"working (group_id={tiers.group_id})\n", working)
        _report(f"archive (dataset={tiers.archive_config.dataset})\n", archive)
    finally:
        tiers.close()

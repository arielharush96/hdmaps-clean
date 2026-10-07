from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_eval = sub.add_parser("evaluate", add_help=False)
    p_eval.add_argument("--protocol", choices=("simultaneous", "staggered", "both"), default="both")
    p_eval.add_argument("--out", type=Path, default=None)
    p_eval.add_argument("--agent", type=Path, default=None)
    p_eval.add_argument("--master", type=Path, default=None)
    p_train = sub.add_parser("train", add_help=False)
    p_train.add_argument("--out", type=Path, default=None)
    p_train.add_argument("--pipeline", action="store_true")
    p_train.add_argument("--episodes", type=int, default=None)
    p_train.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    if args.cmd == "evaluate":
        from hdmaps.evaluation.runner import evaluate_simultaneous, evaluate_staggered
        from hdmaps.statistics.tables import format_table
        out = args.out or Path("outputs") / "evaluate"
        sim_rows = []
        stag_rows = []
        if args.protocol in ("simultaneous", "both"):
            sim_rows = evaluate_simultaneous(out_dir=out, agent_path=args.agent, master_path=args.master)
        if args.protocol in ("staggered", "both"):
            stag_rows = evaluate_staggered(out_dir=out, agent_path=args.agent, master_path=args.master)
        print(format_table(sim_rows, stag_rows or None))
        return 0
    if args.cmd == "train":
        from hdmaps.training.hdmaps import train, train_pipeline
        if args.pipeline:
            print(train_pipeline(out_dir=args.out, seed=args.seed))
        else:
            print(train(out_dir=args.out, episodes=args.episodes, seed=args.seed))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

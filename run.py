"""
run.py  —  SHARED CLI glue (one entry point for the whole pipeline)

    python run.py split        # build seeded 5-partition split + materialize dataset/validation
    python run.py materialize  # build offline augmented dataset twins (dataset/train_aug/)
    python run.py baseline     # naive logistic-regression floor (P0 -> test P1)
    python run.py train        # progressive training -> submissions/my_team/weights.joblib
    python run.py eval         # provided evaluate.py leaderboard (needs dataset/validation)
    python run.py robust       # clean vs provided-OOD accuracy report
    python run.py errors --partition P1   # log misclassifications on one partition

Recommended order for a fresh clone: split -> materialize -> baseline -> train -> robust.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TEAM_DIR = PROJECT_ROOT / "submissions" / "my_team"


def cmd_split(args):
    from split_data import build_splits, materialize_validation
    build_splits(seed=args.seed)
    materialize_validation(args.val_partition)   # so the provided evaluate.py has a folder


def cmd_materialize(args):
    from make_augmented import build_augmented
    build_augmented(copies=args.copies, seed=args.seed)


def cmd_baseline(args):
    import baseline_naive
    baseline_naive.main()


def cmd_train(args):
    # import the team's train.py as a module and call main()
    spec = importlib.util.spec_from_file_location("team_train", TEAM_DIR / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(stage=args.stage, img_size=args.img_size, batch_size=args.batch,
             num_workers=args.workers, epochs=args.epochs)


def cmd_eval(args):
    import evaluate
    evaluate.main()


def cmd_robust(args):
    import robust_eval
    robust_eval.main()


def cmd_errors(args):
    from torch.utils.data import DataLoader
    from data import ManifestDataset
    from augment import build_eval_transform
    from engine import get_device, evaluate as eval_fn
    from error_analysis import log_errors
    from robust_eval import load_trained_model

    device = get_device()
    model = load_trained_model().to(device)
    ds = ManifestDataset([args.partition], build_eval_transform())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    acc, preds, labels, confs = eval_fn(model, loader, device)
    log_errors(ds.samples, preds, labels, confs, tag=args.partition)


def build_parser():
    p = argparse.ArgumentParser(description="IML Challenge 2 pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split"); s.add_argument("--seed", type=int, default=42)
    s.add_argument("--val-partition", default="P1"); s.set_defaults(func=cmd_split)

    m = sub.add_parser("materialize")
    m.add_argument("--copies", type=int, default=1, help="augmented twins per original image")
    m.add_argument("--seed", type=int, default=42)
    m.set_defaults(func=cmd_materialize)

    sub.add_parser("baseline").set_defaults(func=cmd_baseline)

    t = sub.add_parser("train")
    t.add_argument("--stage", type=int, default=1, help="1=P0(40%%) 2=+P1(55%%) 3=+P2(70%%) 4=+P3(85%%); test = next block")
    t.add_argument("--img-size", type=int, default=224)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--epochs", type=int, default=40, help="per-stage max epochs (early-stop usually triggers first)")
    t.set_defaults(func=cmd_train)

    sub.add_parser("eval").set_defaults(func=cmd_eval)
    sub.add_parser("robust").set_defaults(func=cmd_robust)

    e = sub.add_parser("errors"); e.add_argument("--partition", default="P1")
    e.set_defaults(func=cmd_errors)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)

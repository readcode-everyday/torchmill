"""CLI entry point for torchmill.

Commands:
  torchmill train    --config config.yaml          # built-in trainer
  torchmill validate --config config.yaml          # validate trainer config
  torchmill run      --config torchmill.yaml        # autopilot: spawn + watch
  torchmill watch    --log train.log [--pid PID]    # watch existing process
  torchmill history  --dir ./torchmill_runs         # show experiment history
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("torchmill")


# ============================================================ trainer commands


def _build_datasets(config):
    """Construct train/val/test datasets from config."""
    from .data.csv_dataset import CSVDataset
    from .data.relbench import RelBenchDataset

    dcfg = config.data
    dtype = dcfg.type

    if dtype == "csv":
        train_ds = CSVDataset(dcfg.train_path, target_column=dcfg.target_column)
        val_ds = CSVDataset(dcfg.val_path, target_column=dcfg.target_column) if dcfg.val_path else None
        test_ds = CSVDataset(dcfg.test_path, target_column=dcfg.target_column) if dcfg.test_path else None
        return train_ds, val_ds, test_ds

    if dtype == "relbench":
        train_ds = RelBenchDataset(dcfg.dataset_name, dcfg.task_name, split="train")
        val_ds = RelBenchDataset(dcfg.dataset_name, dcfg.task_name, split="val")
        test_ds = RelBenchDataset(dcfg.dataset_name, dcfg.task_name, split="test")
        return train_ds, val_ds, test_ds

    raise ValueError(
        f"Unsupported data type '{dtype}'. For custom datasets use type='custom' "
        "and register your dataset with @register_dataset."
    )


def cmd_train(args: argparse.Namespace) -> None:
    from .config import TorchmillConfig
    from .models.registry import get_model
    from .trainer import Trainer

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading config from {config_path}")
    config = TorchmillConfig.from_yaml(config_path)

    logger.info(
        f"Model: {config.model.class_path}  |  "
        f"Data: {config.data.type}  |  "
        f"Epochs: {config.training.epochs}"
    )

    # Build model
    model = get_model(config.model.class_path, config.model.args)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build datasets
    train_ds, val_ds, test_ds = _build_datasets(config)

    # Train
    trainer = Trainer(config)
    best = trainer.fit(model, train_ds, val_ds)
    logger.info(f"Training complete. Best metrics: {best}")

    # Test
    if test_ds is not None:
        test_metrics = trainer.evaluate(model, test_ds)
        logger.info(f"Test metrics: {test_metrics}")


def cmd_validate(args: argparse.Namespace) -> None:
    from .config import TorchmillConfig
    try:
        config = TorchmillConfig.from_yaml(args.config)
        logger.info("Config is valid.")
        import json
        print(json.dumps(config.model_dump(by_alias=True), indent=2))
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        sys.exit(1)


# ============================================================ autopilot commands


def cmd_run(args: argparse.Namespace) -> None:
    """Mode 1: full autopilot — spawn training process and iterate."""
    from .watcher_config import TorchmillWatcherConfig
    from .autopilot import Autopilot

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    config = TorchmillWatcherConfig.from_yaml(config_path)
    if not config.command:
        logger.error(
            "torchmill.yaml must contain a 'command' field for 'torchmill run'"
        )
        sys.exit(1)

    pilot = Autopilot(config)
    asyncio.run(pilot.run_autopilot())


def cmd_watch(args: argparse.Namespace) -> None:
    """Mode 2: attach to an existing training process / log file."""
    from .watcher_config import TorchmillWatcherConfig
    from .autopilot import Autopilot

    log_file = args.log
    if not log_file:
        logger.error("--log is required for 'torchmill watch'")
        sys.exit(1)

    # Load config if provided, else use defaults
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config not found: {config_path}")
            sys.exit(1)
        config = TorchmillWatcherConfig.from_yaml(config_path)
    else:
        config = TorchmillWatcherConfig()

    pid = args.pid
    pilot = Autopilot(config)
    asyncio.run(pilot.watch_mode(log_file=log_file, external_pid=pid))


def cmd_history(args: argparse.Namespace) -> None:
    """Show experiment history from the log directory."""
    from .history.experiment_log import ExperimentLog

    log_dir = args.dir or "./torchmill_runs"
    exp_log = ExperimentLog(log_dir=log_dir)
    print(exp_log.summary())


# ============================================================ main


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="torchmill",
        description="Autonomous ML training optimizer with LLM brain",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- existing trainer commands ----
    train_parser = subparsers.add_parser("train", help="Train a model (built-in trainer)")
    train_parser.add_argument("--config", "-c", required=True, help="Trainer YAML config")
    train_parser.set_defaults(func=cmd_train)

    val_parser = subparsers.add_parser("validate", help="Validate a trainer config file")
    val_parser.add_argument("--config", "-c", required=True, help="Trainer YAML config")
    val_parser.set_defaults(func=cmd_validate)

    # ---- autopilot commands ----
    run_parser = subparsers.add_parser(
        "run", help="Autopilot: spawn training process and iterate with LLM brain"
    )
    run_parser.add_argument(
        "--config", "-c", required=True, help="torchmill.yaml watcher config"
    )
    run_parser.set_defaults(func=cmd_run)

    watch_parser = subparsers.add_parser(
        "watch", help="Attach to an existing training process log"
    )
    watch_parser.add_argument("--log", "-l", required=True, help="Path to log file to watch")
    watch_parser.add_argument(
        "--pid", "-p", type=int, default=None,
        help="PID of training process (enables kill on trigger)"
    )
    watch_parser.add_argument(
        "--config", "-c", default=None, help="torchmill.yaml config (optional)"
    )
    watch_parser.set_defaults(func=cmd_watch)

    hist_parser = subparsers.add_parser("history", help="Show experiment history")
    hist_parser.add_argument(
        "--dir", "-d", default="./torchmill_runs",
        help="History directory (default: ./torchmill_runs)"
    )
    hist_parser.set_defaults(func=cmd_history)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

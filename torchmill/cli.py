"""CLI entry point: torchmill train --config config.yaml"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("torchmill")


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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="torchmill",
        description="Model-agnostic PyTorch training framework",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # torchmill train --config config.yaml
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--config", "-c", required=True, help="Path to YAML config file"
    )
    train_parser.set_defaults(func=cmd_train)

    # torchmill validate --config config.yaml
    val_parser = subparsers.add_parser(
        "validate", help="Validate a config file without training"
    )
    val_parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    val_parser.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

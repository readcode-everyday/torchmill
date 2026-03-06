#!/usr/bin/env python3
"""
Reusable script: Download Kaggle dataset → Convert to RelBench format.

RelBench format:
  ~/.cache/relbench/rel-<name>/
  ├── db/
  │   ├── <table1>.parquet      # entity/dimension tables
  │   ├── <table2>.parquet
  │   └── ...
  └── tasks/
      └── <task_name>/
          ├── train.parquet     # columns: [timestamp, entity_id, label]
          ├── val.parquet
          └── test.parquet

Usage:
  # With Kaggle API (needs ~/.kaggle/kaggle.json)
  python kaggle_to_relbench.py \
    --kaggle-dataset ziya07/manufacturing-production-data \
    --name manufacturing \
    --output-dir ~/.cache/relbench/rel-manufacturing

  # From local CSV
  python kaggle_to_relbench.py \
    --csv-path /path/to/data.csv \
    --name manufacturing \
    --output-dir ~/.cache/relbench/rel-manufacturing

  # Custom config for column mapping
  python kaggle_to_relbench.py \
    --csv-path /path/to/data.csv \
    --name manufacturing \
    --config dataset_config.yaml
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Default config template ───────────────────────────────────────────────

DEFAULT_CONFIG = {
    "name": "custom",
    "description": "Custom dataset converted to RelBench format",

    # Column mapping
    "columns": {
        "timestamp": None,         # auto-detect or specify column name
        "entity_id": None,         # auto-detect or specify
        "label": None,             # target column for prediction task
    },

    # If timestamp column doesn't exist, generate synthetic timestamps
    "synthetic_timestamp": {
        "enabled": True,
        "start": "2020-01-01",
        "freq": "1h",              # pandas freq string
    },

    # Table splitting: which columns become separate dimension tables
    # If empty, auto-detect categorical columns with low cardinality
    "dimension_tables": {},
    # Example:
    # dimension_tables:
    #   machines:
    #     id_column: machine_id
    #     columns: [machine_type, manufacturer, install_date]
    #   operators:
    #     id_column: operator_id  
    #     columns: [operator_name, shift, experience_years]

    # Task definition
    "task": {
        "name": "predict",
        "type": "classification",  # classification or regression
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "temporal_split": True,    # split by time (True) or random (False)
    },

    # Preprocessing
    "preprocessing": {
        "drop_columns": [],        # columns to drop entirely
        "fillna_strategy": "median",  # median, mean, zero, drop
    },
}


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load config from YAML or return defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}
        _deep_update(config, user_config)
        logger.info(f"Loaded config from {config_path}")
    return config


def _deep_update(base: dict, update: dict) -> dict:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


# ─── Download ──────────────────────────────────────────────────────────────

def download_kaggle_dataset(dataset_slug: str, download_dir: str) -> List[str]:
    """Download dataset from Kaggle API. Returns list of CSV file paths."""
    try:
        import kaggle
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    logger.info(f"Downloading {dataset_slug} from Kaggle...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_slug, path=download_dir, unzip=True)

    csv_files = list(Path(download_dir).rglob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {download_dir}")
        sys.exit(1)

    logger.info(f"Downloaded {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}")
    return [str(f) for f in csv_files]


# ─── Auto-detect columns ──────────────────────────────────────────────────

def auto_detect_columns(df: pd.DataFrame, config: Dict) -> Dict:
    """Auto-detect timestamp, entity_id, and label columns if not specified."""
    cols = config["columns"].copy()

    # Timestamp: look for datetime columns or columns with 'time/date' in name
    if cols["timestamp"] is None:
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        if datetime_cols:
            cols["timestamp"] = datetime_cols[0]
        else:
            for c in df.columns:
                if any(kw in c.lower() for kw in ["timestamp", "datetime", "date", "time"]):
                    try:
                        pd.to_datetime(df[c].head(100))
                        cols["timestamp"] = c
                        break
                    except (ValueError, TypeError):
                        continue

    # Entity ID: look for 'id' columns (prefer non-primary-key looking ones)
    if cols["entity_id"] is None:
        id_candidates = [c for c in df.columns if "id" in c.lower()]
        # Prefer columns like "machine_id", "product_id" over generic "id"
        named_ids = [c for c in id_candidates if c.lower() != "id" and len(c) > 2]
        if named_ids:
            cols["entity_id"] = named_ids[0]
        elif id_candidates:
            cols["entity_id"] = id_candidates[0]

    # Label: use config task label, or try to detect
    if cols["label"] is None:
        label_candidates = [c for c in df.columns
                          if any(kw in c.lower() for kw in
                                 ["target", "label", "class", "defect", "quality",
                                  "outcome", "result", "status", "failure", "predict"])]
        if label_candidates:
            cols["label"] = label_candidates[0]
        else:
            # Last numeric column as fallback
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                cols["label"] = numeric_cols[-1]

    logger.info(f"Column mapping: timestamp={cols['timestamp']}, "
                f"entity_id={cols['entity_id']}, label={cols['label']}")
    return cols


# ─── Dimension table extraction ───────────────────────────────────────────

def extract_dimension_tables(
    df: pd.DataFrame, config: Dict
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Extract dimension/entity tables from flat CSV.
    
    Returns: (fact_table, {table_name: dimension_df})
    """
    dim_tables = {}
    fact_df = df.copy()
    dim_config = config.get("dimension_tables", {})

    if dim_config:
        # User-specified dimension tables
        for table_name, table_cfg in dim_config.items():
            id_col = table_cfg["id_column"]
            dim_cols = [id_col] + [c for c in table_cfg.get("columns", []) if c in df.columns]
            dim_cols = list(dict.fromkeys(dim_cols))  # dedupe, preserve order

            dim_df = df[dim_cols].drop_duplicates(subset=[id_col]).reset_index(drop=True)
            dim_tables[table_name] = dim_df
            logger.info(f"Extracted dimension table '{table_name}': {dim_df.shape}")

            # Remove extracted columns from fact table (keep the ID)
            drop_cols = [c for c in dim_cols if c != id_col and c in fact_df.columns]
            fact_df = fact_df.drop(columns=drop_cols)
    else:
        # Auto-detect: categorical columns with low cardinality → dimension tables
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            nunique = df[col].nunique()
            if 2 < nunique < min(100, len(df) * 0.1):  # reasonable cardinality
                # Create a dimension table with integer IDs
                unique_vals = df[col].dropna().unique()
                dim_df = pd.DataFrame({
                    f"{col}_id": range(len(unique_vals)),
                    col: unique_vals,
                })
                dim_tables[col] = dim_df

                # Replace values in fact table with IDs
                val_to_id = dict(zip(unique_vals, range(len(unique_vals))))
                fact_df[f"{col}_id"] = fact_df[col].map(val_to_id)
                fact_df = fact_df.drop(columns=[col])
                logger.info(f"Auto-extracted dimension table '{col}': {dim_df.shape}")

    return fact_df, dim_tables


# ─── Build RelBench structure ─────────────────────────────────────────────

def build_relbench_structure(
    df: pd.DataFrame,
    config: Dict,
    output_dir: str,
) -> None:
    """Convert DataFrame to RelBench directory structure."""
    output = Path(output_dir)
    db_dir = output / "db"
    tasks_dir = output / "tasks"
    db_dir.mkdir(parents=True, exist_ok=True)

    # Detect columns
    cols = auto_detect_columns(df, config)

    # Preprocessing
    preproc = config.get("preprocessing", {})
    drop_cols = preproc.get("drop_columns", [])
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Handle missing values
    fillna = preproc.get("fillna_strategy", "median")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if fillna == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif fillna == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif fillna == "zero":
        df[numeric_cols] = df[numeric_cols].fillna(0)
    elif fillna == "drop":
        df = df.dropna()
    # String columns: fill with "unknown"
    str_cols = df.select_dtypes(include=["object"]).columns
    df[str_cols] = df[str_cols].fillna("unknown")

    # Ensure timestamp
    if cols["timestamp"] and cols["timestamp"] in df.columns:
        df[cols["timestamp"]] = pd.to_datetime(df[cols["timestamp"]])
    elif config.get("synthetic_timestamp", {}).get("enabled", True):
        synth = config["synthetic_timestamp"]
        start = synth.get("start", "2020-01-01")
        freq = synth.get("freq", "1h")
        df["timestamp"] = pd.date_range(start=start, periods=len(df), freq=freq)
        cols["timestamp"] = "timestamp"
        logger.info(f"Generated synthetic timestamps: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Ensure entity_id
    if not cols["entity_id"] or cols["entity_id"] not in df.columns:
        df["entity_id"] = range(len(df))
        cols["entity_id"] = "entity_id"
        logger.info("Generated synthetic entity_id column")

    # Extract dimension tables
    fact_df, dim_tables = extract_dimension_tables(df, config)

    # Save dimension tables as db/*.parquet
    for table_name, dim_df in dim_tables.items():
        path = db_dir / f"{table_name}.parquet"
        dim_df.to_parquet(path, index=False)
        logger.info(f"Saved db/{table_name}.parquet: {dim_df.shape}")

    # Save main fact table
    fact_path = db_dir / "main.parquet"
    fact_df.to_parquet(fact_path, index=False)
    logger.info(f"Saved db/main.parquet: {fact_df.shape}")

    # Build task splits
    task_cfg = config.get("task", {})
    task_name = task_cfg.get("name", "predict")
    label_col = cols["label"]
    ts_col = cols["timestamp"]
    entity_col = cols["entity_id"]

    if not label_col or label_col not in fact_df.columns:
        logger.error(f"Label column '{label_col}' not found. Available: {list(fact_df.columns)}")
        sys.exit(1)

    # Create task dataframe: [timestamp, entity_id, label]
    task_df = pd.DataFrame({
        "timestamp": fact_df[ts_col],
        entity_col: fact_df[entity_col],
        "label": fact_df[label_col],
    })

    # Encode label for classification
    task_type = task_cfg.get("type", "classification")
    if task_type == "classification" and task_df["label"].dtype == object:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        task_df["label"] = le.fit_transform(task_df["label"])
        # Save label mapping
        mapping = dict(zip(le.classes_.tolist(), range(len(le.classes_))))
        with open(output / "label_mapping.json", "w") as f:
            json.dump(mapping, f, indent=2, default=str)
        logger.info(f"Label encoding: {mapping}")

    task_df["label"] = task_df["label"].astype(
        "int32" if task_type == "classification" else "float32"
    )

    # Split
    train_r = task_cfg.get("train_ratio", 0.7)
    val_r = task_cfg.get("val_ratio", 0.15)
    temporal = task_cfg.get("temporal_split", True)

    if temporal and ts_col in task_df.columns:
        task_df = task_df.sort_values("timestamp").reset_index(drop=True)
    else:
        task_df = task_df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(task_df)
    train_end = int(n * train_r)
    val_end = int(n * (train_r + val_r))

    train_df = task_df.iloc[:train_end]
    val_df = task_df.iloc[train_end:val_end]
    test_df = task_df.iloc[val_end:]

    # Save task splits
    task_dir = tasks_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(task_dir / "train.parquet", index=False)
    val_df.to_parquet(task_dir / "val.parquet", index=False)
    test_df.to_parquet(task_dir / "test.parquet", index=False)

    logger.info(f"Task '{task_name}' splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Save metadata
    meta = {
        "name": config.get("name", "custom"),
        "description": config.get("description", ""),
        "created_at": datetime.now().isoformat(),
        "source": config.get("source", "unknown"),
        "columns": cols,
        "task": {
            "name": task_name,
            "type": task_type,
            "num_classes": int(task_df["label"].nunique()) if task_type == "classification" else None,
        },
        "tables": {
            "main": {"shape": list(fact_df.shape), "columns": list(fact_df.columns)},
            **{name: {"shape": list(d.shape), "columns": list(d.columns)}
               for name, d in dim_tables.items()},
        },
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    with open(output / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"✅ RelBench dataset saved to {output}")
    logger.info(f"   db tables: {list(dim_tables.keys()) + ['main']}")
    logger.info(f"   task: {task_name} ({task_type})")
    logger.info(f"   total samples: {n}")


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert Kaggle/CSV dataset to RelBench format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From Kaggle
  python kaggle_to_relbench.py \\
    --kaggle-dataset ziya07/manufacturing-production-data \\
    --name manufacturing \\
    --output-dir ~/.cache/relbench/rel-manufacturing

  # From local CSV
  python kaggle_to_relbench.py \\
    --csv-path data.csv \\
    --name manufacturing \\
    --output-dir ~/.cache/relbench/rel-manufacturing

  # With custom config
  python kaggle_to_relbench.py \\
    --csv-path data.csv \\
    --config manufacturing_config.yaml

  # Generate config template
  python kaggle_to_relbench.py --generate-config manufacturing_config.yaml
        """,
    )

    parser.add_argument("--kaggle-dataset", type=str, help="Kaggle dataset slug (e.g., user/dataset)")
    parser.add_argument("--csv-path", type=str, help="Path to local CSV file")
    parser.add_argument("--name", type=str, default="custom", help="Dataset name (used in rel-<name>)")
    parser.add_argument("--output-dir", type=str, help="Output directory (default: ~/.cache/relbench/rel-<name>)")
    parser.add_argument("--config", type=str, help="YAML config file for column mapping and options")
    parser.add_argument("--generate-config", type=str, help="Generate a config template YAML file and exit")

    # Quick overrides (no config file needed)
    parser.add_argument("--timestamp-col", type=str, help="Timestamp column name")
    parser.add_argument("--entity-col", type=str, help="Entity ID column name")
    parser.add_argument("--label-col", type=str, help="Label/target column name")
    parser.add_argument("--task-type", type=str, choices=["classification", "regression"],
                       help="Task type")
    parser.add_argument("--task-name", type=str, help="Task name")

    args = parser.parse_args()

    # Generate config template
    if args.generate_config:
        with open(args.generate_config, "w") as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
        print(f"Config template written to {args.generate_config}")
        return

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.name:
        config["name"] = args.name
    if args.timestamp_col:
        config["columns"]["timestamp"] = args.timestamp_col
    if args.entity_col:
        config["columns"]["entity_id"] = args.entity_col
    if args.label_col:
        config["columns"]["label"] = args.label_col
    if args.task_type:
        config["task"]["type"] = args.task_type
    if args.task_name:
        config["task"]["name"] = args.task_name

    # Determine output dir
    output_dir = args.output_dir or str(Path.home() / ".cache" / "relbench" / f"rel-{config['name']}")

    # Get CSV data
    if args.csv_path:
        csv_path = args.csv_path
        config["source"] = f"local:{csv_path}"
    elif args.kaggle_dataset:
        tmpdir = tempfile.mkdtemp(prefix="kaggle_")
        csv_files = download_kaggle_dataset(args.kaggle_dataset, tmpdir)
        csv_path = csv_files[0]  # Use first CSV
        if len(csv_files) > 1:
            logger.warning(f"Multiple CSV files found, using: {csv_path}")
            logger.warning(f"Other files: {csv_files[1:]}")
        config["source"] = f"kaggle:{args.kaggle_dataset}"
    else:
        parser.error("Provide either --kaggle-dataset or --csv-path")
        return

    # Load and convert
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Dtypes:\n{df.dtypes.to_string()}")

    build_relbench_structure(df, config, output_dir)


if __name__ == "__main__":
    main()

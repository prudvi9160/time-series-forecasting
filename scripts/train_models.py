from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline import clean_dates, load_raw_data
from app.training import train_all_states


def main() -> None:
    source_path = os.getenv("DATA_SOURCE_PATH", "data/raw.xlsx")
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")
    validation_days = int(os.getenv("VALIDATION_DAYS", "56"))
    bundle = clean_dates(load_raw_data(source_path))
    artifacts = train_all_states(bundle, validation_days=validation_days, artifacts_dir=artifacts_dir)
    for artifact in artifacts:
        print(f"{artifact.state}: best={artifact.best_model} saved={artifact.artifact_path}")


if __name__ == "__main__":
    main()
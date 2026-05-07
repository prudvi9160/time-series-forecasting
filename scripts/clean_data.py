from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline import clean_excel_file


def main() -> None:
    source_path = os.getenv("DATA_SOURCE_PATH", "data/raw.xlsx")
    output_path = os.getenv("CLEANED_DATA_PATH", "data/cleaned.xlsx")
    cleaned_path = clean_excel_file(source_path, output_path)
    print(f"Cleaned data saved to {cleaned_path}")


if __name__ == "__main__":
    main()
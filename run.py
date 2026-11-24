"""Run the local AIMPACT PCB inspection workbench."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--database", type=Path, default=Path("outputs/inspections.sqlite3"))
    parser.add_argument("--assets", type=Path, default=Path("data/pcb_demo"))
    parser.add_argument("--empty", action="store_true", help="Skip demonstration records for a new database")
    args = parser.parse_args()

    assets_path = ROOT / args.assets
    if not (assets_path / "model.joblib").is_file():
        parser.error("PCB assets are missing. Follow the preparation instructions in docs/PCB_BENCHMARK.md.")

    from pcb_inspection.web import create_app

    app = create_app(ROOT / args.database, assets_path, load_demo=not args.empty)
    app.run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

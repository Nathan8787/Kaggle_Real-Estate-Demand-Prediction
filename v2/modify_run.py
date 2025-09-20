# -*- coding: utf-8 -*-
from pathlib import Path
path = Path(r"d:\\User\\орн▒\\Nathan\\side_project\\Kaggle_Real Estate Demand Prediction\\v2\\scripts\\run_pipeline_v2.py")
text = path.read_text(encoding="utf-8")
text = text.replace("from src.automl_runner_v2 import run\n\n\n", "from src.automl_runner_v2 import run\nfrom walk_forward_train import walk_forward\n\n\n")
text = text.replace("def walk_forward_train(args: argparse.Namespace) -> None:\n    raise NotImplementedError(\"walk_forward_train CLI is not yet implemented in this script\")\n", "def walk_forward_train(args: argparse.Namespace) -> None:\n    walk_forward(args)\n")
path.write_text(text, encoding="utf-8")


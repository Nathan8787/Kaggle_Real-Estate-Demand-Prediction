# -*- coding: utf-8 -*-
from pathlib import Path
import sys

path = Path(r"d:\\User\\орн▒\\Nathan\\side_project\\Kaggle_Real Estate Demand Prediction\\v2\\scripts\\walk_forward_train.py")
text = path.read_text(encoding="utf-8")
if "sys.path.append" not in text:
    insertion = "import argparse\nimport json\nimport sys\nfrom pathlib import Path\n\nsys.path.append(str(Path(__file__).resolve().parents[1]))\n\nimport joblib\nimport numpy as np\nimport pandas as pd\n"
    text = text.replace("import argparse\nimport json\nfrom pathlib import Path\n\nimport joblib\nimport numpy as np\nimport pandas as pd\n", insertion)
    path.write_text(text, encoding="utf-8")


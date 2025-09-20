import pathlib
import re

text = pathlib.Path('kaggle_data_page.md').read_text(encoding='utf-8')
filenames = re.findall(r"\*\*(train/[^*]+)\*\*", text)
for name in filenames:
    print(name)

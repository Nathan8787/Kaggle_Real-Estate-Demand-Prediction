import pathlib
import re

text = pathlib.Path('kaggle_data_page.md').read_text(encoding='utf-8')
pattern = re.compile(r"\*\*(train/[^*]+)\*\*\n((?:\s*\*   \*\*[^\n]+\n?)+)")
for match in pattern.finditer(text):
    filename = match.group(1)
    items = [line.strip('* ').strip() for line in match.group(2).strip().split('\n') if line.strip()]
    print(filename)
    for item in items:
        print('  ', item)
    print()

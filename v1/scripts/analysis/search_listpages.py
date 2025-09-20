import pathlib
import re

text = pathlib.Path('kaggle_app.js').read_text(encoding='utf-8', errors='ignore')
for match in re.finditer(r"listPages\(\{[^\}]*\}\)", text):
    snippet = match.group(0)
    if 'selector' in snippet:
        print(snippet)
        break
else:
    print('not found')

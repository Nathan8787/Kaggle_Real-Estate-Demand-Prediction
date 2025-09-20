import pathlib

text = pathlib.Path('kaggle_app.js').read_text(encoding='utf-8', errors='ignore')
needle = 'competitions.PageService/ListPages'
idx = text.find(needle)
print(text[idx-500:idx+200])

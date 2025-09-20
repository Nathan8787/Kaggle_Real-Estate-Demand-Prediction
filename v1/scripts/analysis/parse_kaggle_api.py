import pathlib

text = pathlib.Path("kaggle_app.js").read_text(encoding="utf-8", errors="ignore")
idx = text.find('storage.googleapis.com')
print(idx)
if idx != -1:
    print(text[idx-200:idx+200])

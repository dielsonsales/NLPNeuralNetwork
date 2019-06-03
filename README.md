## Installing

Install the packages in `requirements.txt` with:

```
pip install -r requirements.txt
```

Install the pt_wikipedia_md module:

```
pip install --no-cache-dir --no-deps pt_wikipedia_md-1.0.0.tar.gz
```

Link the model to use it as the `pt` language:

```
python -m spacy link pt_wikipedia_md pt --force
```

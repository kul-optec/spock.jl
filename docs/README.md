Documenter has pretty URLs enabled, i.e. `src/foo.md` is turned into `src/foo/index.html` instead of `src/foo.html`.

This may not work when browsing the documentation locally. As a workaround, run
```
python3 -m http.server --bind localhost
```
and access the docs locally at `localhost:8000`.
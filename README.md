# Teachable Machine Lab: browser version

A static, client-side rewrite of the `test_02.py` teachable-machine idea. It is a single file, `index.html`, with no backend.



## Run locally

```bash
python3 -m http.server 8600      # run from this folder
# open http://localhost:8600
```

`getUserMedia` needs a secure context, so use `localhost` or https. It won't work from `file://`.

## Host on GitHub Pages

Pages serves only from the repo root or `/docs`, so keep `index.html` at the root (or move it to `docs/`).

Use a repo of your own. Free Pages needs it to be public.

```bash
git init
git add index.html
git commit -m "Add Teachable Machine Lab"
git branch -M main
git remote add origin https://github.com/<you>/teachable-machine-lab.git
git push -u origin main
```

Then enable Pages, either in the UI (Settings → Pages → Deploy from a branch → `main` / `/ (root)`) or with the CLI:

```bash
gh api -X POST repos/<you>/teachable-machine-lab/pages \
  -f "source[branch]=main" -f "source[path]=/"
```

After a minute or two it is live at `https://<you>.github.io/teachable-machine-lab/`.

## Notes

- Pages is https, so the camera works there.
- A locked-down network that blocks `cdn.jsdelivr.net` will stop the model loading. To remove that dependency, vendor the three scripts and self-host the MobileNet weights.
- Raw-pixel features (`test_02.py`) are replaced by MobileNet embeddings, which is why a handful of samples is enough.
- Browser smoke-tested only with Chrome's fake camera (`--use-fake-device-for-media-stream`), not with a real webcam.

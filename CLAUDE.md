# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A beginner-level scikit-learn course (supervised learning), not an application. There is no package, test suite, linter, or build step. Each lesson in `lessons/` is a standalone, runnable script. `IIT KGP Gen AI - Final Student Handbook.pdf` and `plan/*.png` are course reference material, not code inputs.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy<2, pandas, scikit-learn, matplotlib

python lessons/01_first_classifier.py  # run one lesson; run from any cwd
```

There are no tests; verify a lesson change by running the script and checking it exits cleanly.

## Lesson conventions

Lessons are numbered `NN_topic.py` and are meant to be read top to bottom, so keep new or edited lessons consistent with the existing ones:

- Module docstring first, explaining the concept being taught.
- Numbered `# --- N. Step name ---...` section comments with a comment explaining *why*, not just what.
- Built-in sklearn datasets only (iris, diabetes, wine, breast_cancer) — no external data files.
- Fixed `random_state=42` and `stratify=y` on classification splits so output is reproducible.
- Each file ends with a `# --- Exercises ---` block of numbered student exercises.
- Lessons build on each other (e.g. 01 points ahead to 08; 09 assumes pipelines/CV concepts), so the order in the README table matters. Update the README lesson table when adding or renaming a lesson.

### Plotting

Lessons that plot call `matplotlib.use("Agg")` *before* importing `pyplot` (headless, no window) and save into the repo-level `outputs/` directory via:

```python
OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)
```

Output PNGs are named after the lesson (e.g. `04_knn_scaling.png`). Only lessons 01, 04, and 06 currently produce plots.

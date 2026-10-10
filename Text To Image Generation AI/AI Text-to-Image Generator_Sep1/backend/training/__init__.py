"""
backend/training
-----------------
A separate, optional module for preparing a fine-tuning dataset.

Nothing in this package is imported by `backend/main.py` or
`backend/model.py`, and nothing in this package imports from them either.
Running (or not running) anything in here has zero effect on the live
SD-Turbo generation app — it's offline tooling you run yourself from the
command line. See `backend/training/README.md` for details.
"""

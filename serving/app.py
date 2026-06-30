"""Modal app for Surge — serves Llama-3-8B with vLLM and runs benchmarks on a GPU.

Implemented in milestone M1 (smoke test) and M3+ (sweep). This is the scaffold.
"""

# Entry points wired up here are referenced by the Makefile:
#   modal run serving/app.py::smoke
#   modal run serving/app.py::sweep

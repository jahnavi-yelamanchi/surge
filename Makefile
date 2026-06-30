.PHONY: help smoke sweep plots test

help:
	@echo "Surge — vLLM serving optimization"
	@echo ""
	@echo "  make smoke    Boot vLLM on a cheap GPU and send one request"
	@echo "  make sweep    Run the baseline-vs-tuned benchmark sweep on Modal"
	@echo "  make plots    Regenerate charts from results/raw/"
	@echo "  make test     Run local unit tests (no GPU needed)"

smoke:
	modal run serving/app.py::smoke

sweep:
	modal run serving/app.py::sweep

plots:
	python -m analyze.plot

test:
	python -m pytest -q

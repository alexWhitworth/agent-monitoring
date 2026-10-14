# agent_monitor — dev workflow gates (details in MUTATION_NOTES.md)

.PHONY: mutate

# F-013: restricted mutation run on the critical modules (joint.py,
# population.py, queue.py — scope set by [tool.mutmut] in pyproject.toml).
# mutmut caches verdicts under mutants/, so the cache is wiped first;
# mutmut>=3 is a dev extra, hence --extra dev.
mutate:
	rm -rf mutants
	uv run --extra dev mutmut run
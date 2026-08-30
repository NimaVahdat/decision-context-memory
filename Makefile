# Decision Context Memory — per-approach build tooling.
#
# Each approach is a self-contained uv project in its own folder. Every
# target loops over all of them so `make <target>` operates on the whole
# repo, while `make <target> APPROACHES=forge_dc` scopes to one folder.

APPROACHES ?= doc_naive doc_chunk hybrid_raw lucid_dc forge_dc

.PHONY: install
install:
	@for a in $(APPROACHES); do \
		echo "== uv sync $$a =="; \
		( cd $$a && uv sync ) || exit 1; \
	done

.PHONY: lint
lint:
	@for a in $(APPROACHES); do \
		echo "== format + autofix $$a =="; \
		( cd $$a && uv run ruff format . && uv run ruff check --fix . ) || exit 1; \
	done

.PHONY: lint-check
lint-check: decouple-check
	@for a in $(APPROACHES); do \
		echo "== ruff check $$a =="; \
		( cd $$a && uv run ruff check . ) || exit 1; \
	done

# Guard the decoupling contract: no folder may import `dc_approaches`
# (the legacy coupled package) or another approach's top-level package.
# Each approach must be copy-one-folder-and-go.
.PHONY: decouple-check
decouple-check:
	@echo "Checking approach folders are fully decoupled"
	@fail=0; \
	for a in $(APPROACHES); do \
		if grep -rEn "dc_approaches" "$$a/$$a" --include="*.py" >/dev/null 2>&1; then \
			echo "ERROR: $$a references legacy package dc_approaches"; fail=1; \
		fi; \
		for other in $(APPROACHES); do \
			[ "$$a" = "$$other" ] && continue; \
			if grep -rEn "^[[:space:]]*(from|import)[[:space:]]+$$other([. ]|$$)" "$$a/$$a" --include="*.py" >/dev/null 2>&1; then \
				echo "ERROR: $$a imports cross-folder package $$other"; fail=1; \
			fi; \
		done; \
	done; \
	if [ "$$fail" != "0" ]; then \
		echo "Decoupling violated: each approach must import only itself + third-party deps."; \
		exit 1; \
	fi; \
	echo "decouple-check: OK"

.PHONY: typing
typing:
	@for a in $(APPROACHES); do \
		echo "== mypy $$a =="; \
		( cd $$a && uv run mypy $$a ) || exit 1; \
	done

.PHONY: test
test:
	@for a in $(APPROACHES); do \
		echo "== pytest $$a =="; \
		( cd $$a && uv run pytest ) || exit 1; \
	done

.PHONY: clean
clean:
	@echo "Cleaning..."
	-find . -name __pycache__ -exec rm -rf {} +
	-find . -name .pytest_cache -exec rm -rf {} +
	-find . -name .mypy_cache -exec rm -rf {} +
	-find . -name .ruff_cache -exec rm -rf {} +
	-find . -name '*.egg-info' -exec rm -rf {} +

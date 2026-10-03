# Project instructions

- Read `README.md`, `docs/research_plan.md`, and `docs/decisions.md` before substantive work.
- Treat `data/raw/` as immutable unless explicitly instructed otherwise. Generate derived datasets reproducibly with scripts.
- In rolling forecasts, never use future information; do not use the holdout/test period to choose model specifications.
- Keep source code out of `outputs/` and generated figures/tables out of `data/`.
- Record important methodological decisions in `docs/decisions.md`.
- Run relevant tests before committing.
- Complete only the requested project stage; do not begin the next stage without explicit instruction.
- Do not modify the submitted proposal unless explicitly instructed.
- Resolve script paths from the repository root using `__file__`, not the caller's working directory.

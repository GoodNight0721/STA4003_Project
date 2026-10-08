# First-round research review and verification

Reviewed research code through commit `d40127703b9598a9781fc98a150bf6d85e9aac18` against the Stage 7 baseline `4b03bdc3b4602b4419478584b170da2e8b5dd927`, on 2026-10-08.

- Window implementation: specification and code-quality reviews approved. Real study completed 174/174 fits and 344 forecasts; checkpoint replay reused all successful entries.
- Statistical implementation: specification and scientific/code reviews approved. Saved point metrics, all 2,752 interval records, time eligibility and selected paired-block bootstrap values were independently reconciled with actual archived inputs. The nonempty partial-window scoring fixture was added after review.
- Whole-branch review identified one Important defect: stale completed execution metadata could hide interrupted current window exports in analysis-only mode. The fix now derives counts/completeness from current fit keys, frozen orders, forecast keys and analysis coverage; incomplete models are not scored, and incomplete CLI execution exits 2. Old execution metadata is retained only as trace evidence.
- A scoped re-review marked the finding ADDRESSED and found no new Critical or Important issues in the fix. Nine complete-study numeric tables were exactly unchanged in the regression comparison.
- Fresh whole-suite verification: **149 passed in 78.12s**. `python -m compileall -q scripts tests` and `git diff --check` passed. See `verification.txt` for test stdout.
- Baseline preservation: all 67 existing manifest artifacts and 8 other archived artifact files passed byte checks; Git comparison also found no changes in the original data, proposal, forecast/table/figure/report directories or `docs/final_report.md`.
- Cold Windows checkout verification with automatic line-ending conversion enabled also passed all 75 archive checks. Explicit Git attributes preserve the original mixed LF/CRLF serialization. Numerical optimization across library/BLAS environments is not claimed byte-portable.
- Four static research figures were visually inspected for units, scales, clipping and uncertainty presentation.

Nonblocking limitation retained: warnings occurring before an optimizer exception, or during deterministic starting-value construction outside the warning context, may be omitted from the attempt warning list. Failures remain explicit and successful-fit optimizer warnings remain recorded; no fit failed in this study.

The authorized research branch was pushed and its remote SHA verified. Main-model adoption, a new final forecast, and merging to main remain outside this round. Historical improvements retain exploratory status and do not constitute independent future validation.

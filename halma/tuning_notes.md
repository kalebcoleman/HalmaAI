# Halma AI Tuning Notes

- `arena.py` supports per-side depth/time, alpha-beta toggles, and alternating start player.
- Starting player:
  - `--start-player G|R|alternate` (alternate is recommended for fairness).
- Time per side:
  - Global: `--time-limit`
  - Per-side: `--time-limit-g`, `--time-limit-r`

Example runs:
- Baseline deterministic: `python arena.py --games 10 --start-player alternate --depth-g 3 --depth-r 3 --time-limit 10`
- Depth handicap: `python arena.py --games 20 --start-player alternate --depth-g 3 --depth-r 4 --time-limit 10`
- Time handicap: `python arena.py --games 20 --start-player alternate --time-limit-g 8 --time-limit-r 12 --depth-g 3 --depth-r 3`
- No pruning comparison: `python arena.py --games 10 --start-player alternate --no-alpha-beta`

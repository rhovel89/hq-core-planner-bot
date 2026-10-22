#!/bin/sh
# Explicit Railway worker entrypoint. Run only in the ORIGINAL verified bot service.
# DATA_FILE and its attached volume must have been recovered and backed up first.
set -eu
python -u scripts/preflight_storage.py --require-volume
exec python -u hq_core_planner_bot.py

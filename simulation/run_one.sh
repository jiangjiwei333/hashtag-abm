#!/bin/sh
# run_one.sh LIFE_TIME_TYPE MEMORY D0 THETA MU SIGMA NETWORK SEED
#
# One simulation run, invoked by parallel_sim.sh through xargs. Kept as a separate file
# rather than an inline `sh -c '...'`: BSD xargs (macOS) caps the length of a command line
# assembled with -I at ~255 bytes and fails with "command line cannot be assembled, too long".
#
# simulation.sh is copied to a private temp file first, so that editing simulation.sh while
# runs are in flight cannot change the behaviour of running copies (sh reads scripts lazily).

if [ $# -ne 8 ]; then
    echo "run_one.sh: expected 8 parameters, got $# ($*)" >&2
    exit 1
fi

mkdir -p temp_scripts simulation_logs

TEMP_SCRIPT="temp_scripts/sim_$$.sh"
cp simulation.sh "$TEMP_SCRIPT"
chmod +x "$TEMP_SCRIPT"

LOG_FILE="simulation_logs/sim_$1_$2_$3_$4_$5_$6_$7_$8.log"
echo "start: $* -> $LOG_FILE"
./"$TEMP_SCRIPT" "$@" > "$LOG_FILE" 2>&1
STATUS=$?
rm -f "$TEMP_SCRIPT"

if [ $STATUS -ne 0 ]; then
    echo "FAILED (exit $STATUS): $*  -- see $LOG_FILE" >&2
else
    echo "done : $*"
fi
exit $STATUS

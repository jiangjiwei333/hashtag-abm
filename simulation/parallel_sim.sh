#!/bin/bash
#
# parallel_sim.sh -- run the simulations behind Fig. 4, Fig. 6, Fig. 7 and Fig. 9(a) of the paper.
#
#   chmod +x parallel_sim.sh
#   ./parallel_sim.sh A        # Model 0: main run + memory-window sweep   (8 runs)
#   ./parallel_sim.sh B        # Model 0: network randomisations           (3 runs)
#   ./parallel_sim.sh C        # Model 1: main run                         (1 run)
#   ./parallel_sim.sh all      # A, B and C in sequence                    (12 runs)
#
# The three passes are kept separate on purpose: putting every parameter in one nested
# loop would produce the full Cartesian product (8 memory windows x 4 networks = 32 runs)
# instead of the 12 the paper needs.
#
# Argument order of simulation.sh:
#   LIFE_TIME_TYPE  MEMORY  D0  THETA  MU  SIGMA  NETWORK  SEED
# MODEL is read from the environment and only tags the output folder
# (simulation_results/model<MODEL>/); the C++ never reads it.
#
# Which run feeds which figure panel:
#   A, m=24            -> Fig. 4 (a) (b) (c) (d)
#   A, all 8 windows   -> Fig. 4 (f)
#   B, 3 networks      -> Fig. 4 (e)
#   C                  -> Fig. 6 (a)-(f), Fig. 7 (a) (b), Fig. 9 (a)
#
# Output per run (see simulation.sh):
#   simulation_results/model<M>/results/<FILE_NAME>.txt        daily usage counts x_k(t)
#   simulation_results/model<M>/new_users/<FILE_NAME>.txt      n_k^new (new adopters)
#   simulation_results/model<M>/xt/, diffu_value/, figs/, error/

set -u

DIR="./modules"
CPP="$DIR/simulation.cpp"
BIN="$DIR/simulation"

PARALLEL_JOBS=${PARALLEL_JOBS:-4}    # each full-size run needs ~14 GB of RAM; raise this to fit your machine
SEED=${SEED:-1}

PASS="${1:-all}"

# ------------------------------------------------------------------ compile once
# -march=native/-mtune=native are not accepted by Apple clang on arm64; fall back silently.
CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:-"-std=c++17 -O3 -mtune=native -march=native"}
if ! $CXX $CXXFLAGS "$CPP" -o "$BIN" 2>/dev/null; then
    echo "note: retrying without -march/-mtune=native"
    CXXFLAGS="-std=c++17 -O3"
    if ! $CXX $CXXFLAGS "$CPP" -o "$BIN"; then
        echo "Compilation failed! Exiting..."
        exit 1
    fi
fi
echo "compiled with: $CXX $CXXFLAGS"

mkdir -p simulation_logs temp_scripts

# ------------------------------------------------------------------ runner
# Reads parameter lines on stdin (8 fields each) and runs them PARALLEL_JOBS at a time.
# -n 8 rather than -I {}: BSD xargs (macOS) refuses long command lines assembled with -I.
run_lines() {
    MODEL="$1" xargs -n 8 -P "$PARALLEL_JOBS" ./run_one.sh
}

# ------------------------------------------------------------------ pass A
# Model 0 (const_dk, d_k fixed to 1): main run at m=24 plus the memory-window sweep.
pass_A() {
    echo "[A] Model 0: main run + memory sweep (8 runs)"
    for MEMORY in 24 36 48 72 84 96 108 120; do
        echo "const_dk ${MEMORY} 1 0 0 0.1 real ${SEED}"
    done | run_lines 0
}

# ------------------------------------------------------------------ pass B
# Model 0 at m=24 on randomised copies of the user relationship network (Fig. 4e).
pass_B() {
    echo "[B] Model 0: network randomisations (3 runs)"
    for NETWORK in real_to_random real_keep_in_degree real_keep_out_degree; do
        echo "const_dk 24 1 0 0 0.1 ${NETWORK} ${SEED}"
    done | run_lines 0
}

# ------------------------------------------------------------------ pass C
# Model 1: OU-decaying effective contagiousness, calibrated parameters of the paper
# (m = 84 h, d0 = 13, theta = 0.007, mu = 0, sigma = 0.1).
pass_C() {
    echo "[C] Model 1: main run (1 run)"
    echo "ou 84 13 0.007 0 0.1 real ${SEED}" | run_lines 1
}

case "$PASS" in
    A|a) pass_A ;;
    B|b) pass_B ;;
    C|c) pass_C ;;
    all) pass_A; pass_B; pass_C ;;
    *) echo "usage: $0 {A|B|C|all}"; exit 1 ;;
esac

wait
echo "All simulations completed (pass: $PASS, seed: $SEED)"

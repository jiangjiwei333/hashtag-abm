#!/bin/sh

# One simulation run: builds the output paths, runs the binary, post-processes.
# Argument order: LIFE_TIME_TYPE MEMORY D0 THETA MU SIGMA NETWORK SEED
DIR="./modules"
CPP="$DIR/simulation.cpp"
BIN="$DIR/simulation" # compile output file name

# create the parent directory of a path if needed
create_directory() {
    local dir_path=$(dirname "$1")
    if [ ! -d "$dir_path" ]; then
        mkdir -p "$dir_path"
        echo "Create folder: $dir_path"
    fi
}

if [ ! -e $CPP ]; then
    echo "$CPP not exists"
    exit 1
fi

# This script does NOT compile (so that many copies can run in parallel safely).
# parallel_sim.sh compiles before dispatching; when calling simulation.sh directly, run ./build.sh first.
if [ ! -x $BIN ]; then
    echo "error: $BIN not found -- run ./build.sh first (or use ./parallel_sim.sh, which compiles)."
    exit 1
fi

# ------------------- python result processor ---------------------
# Needs numpy + pandas. Set PYTHON explicitly if the default interpreter lacks them,
# e.g.  PYTHON=$(conda run -n torch which python) ./simulation.sh ...
PYTHON=${PYTHON:-$(command -v python >/dev/null 2>&1 && echo python || echo python3)}
if ! $PYTHON -c 'import numpy, pandas' >/dev/null 2>&1; then
    echo "warning: '$PYTHON' has no numpy/pandas -- the simulation will still run and write"
    echo "         results/, new_users/ and xt/, but the post-processing"
    echo "         (.npy/.pkl conversion and the check plots) will fail. Set PYTHON to fix."
fi
PY_TO_NPY="process_result/to_npy.py"
GET_TIME_SERIES="process_result/get_hashtagdf.py"
PLOT_RESULTS="process_result/plot_results.py"
SAVE_HIGH_RESO_XT="process_result/save_high_resolution_xt.py"

# ---------------------- model parameters -------------------------
MODEL=${MODEL:-1}           # output-folder tag only (config.model is never read by the C++);
                            # override from the environment, e.g. `MODEL=0 ./simulation.sh ...`
DIR_OUT=./simulation_results/model$MODEL
SHOW_PROGRESS=1             # 0/1
SEED=$8

N_AGENTS=397369             # users in the LCC of the retweet network
N_DAYS=30                   # model days simulated
N_SAVE_FROM_LAST_STEP=10    # write the last 10 days (the paper's statistics use the final 7)
ONE_HOUR=93626              # posting events per model hour (empirical mean); one day = 24 x this
P_NEW=0.0218                # probability of creating a new hashtag (empirical)
P_NEW_NAME=_pn"$P_NEW"

D0=$3
THETA=$4
MU=$5
SIGMA=$6

# ou       = Model 1 (OU-decaying effective contagiousness)
# const_dk = Model 0 (exposure only, d_k == 1)
LIFE_TIME_TYPE=$1
if [ "$LIFE_TIME_TYPE" = "ou" ]; then
    LIFE_TIME_NAME=_ou_sigma"$SIGMA"_theta"$THETA"_mu"$MU"_d0"$D0"
elif [ "$LIFE_TIME_TYPE" = "const_dk" ]; then
    LIFE_TIME_NAME=_constdk_c"$D0"
else
    echo "error: unknown LIFE_TIME_TYPE '$LIFE_TIME_TYPE' (expected 'ou' or 'const_dk')"
    exit 1
fi

# network: real / real_to_random / real_keep_in_degree / real_keep_out_degree
NETWORK=$7
NETWORK_NAME=_"$NETWORK"_net

CHANG_INTEREST=network      # positional argument kept for compatibility; only "network" is implemented
MEMORY=$2
MEMORY_NAME=_m"$MEMORY"_"steps"

FILE_NAME=N"$N_AGENTS"_T"$N_DAYS"_HOUR"$ONE_HOUR""$P_NEW_NAME""$LIFE_TIME_NAME""$NETWORK_NAME""$MEMORY_NAME"_SEED"$SEED".txt

PATH_RESULT="$DIR_OUT"/results/"$FILE_NAME"
PATH_FIG="$DIR_OUT"/figs/"$FILE_NAME"
PATH_ERROR_SAVE="$DIR_OUT"/error/"$FILE_NAME"
PATH_DIFFU_SAMPLE="$DIR_OUT"/diffu_value/sample"$FILE_NAME"
PATH_XT="$DIR_OUT"/xt/"$FILE_NAME"
PATH_NEW_USERS="$DIR_OUT"/new_users/"$FILE_NAME"   # [new-users] users who did not post the hashtag on the previous day

# output directories
create_directory "$PATH_RESULT"
create_directory "$PATH_FIG"
create_directory "$PATH_ERROR_SAVE"
create_directory "$PATH_XT"
create_directory "$PATH_DIFFU_SAMPLE"        # Fig. 9(a); was missing, so the file was never written
create_directory "$PATH_NEW_USERS"

# SKIP_SIM=1 re-runs only the python post-processing on an existing result file
# (use that instead of commenting the line out, which silently produces an empty run).
if [ "${SKIP_SIM:-0}" != "1" ]; then
$BIN $MODEL $N_AGENTS $N_DAYS $P_NEW $LIFE_TIME_TYPE $NETWORK $CHANG_INTEREST $MEMORY $PATH_RESULT $SIGMA $THETA $MU $D0 $ONE_HOUR $PATH_XT $PATH_DIFFU_SAMPLE $N_SAVE_FROM_LAST_STEP $SEED $PATH_NEW_USERS   # [new-users] 19th argument
fi
$PYTHON $PY_TO_NPY $PATH_RESULT int
$PYTHON $SAVE_HIGH_RESO_XT $PATH_XT $N_DAYS $N_SAVE_FROM_LAST_STEP
$PYTHON $GET_TIME_SERIES $PATH_RESULT $N_DAYS $N_SAVE_FROM_LAST_STEP

last_line=$($PYTHON -u $PLOT_RESULTS $PATH_RESULT $PATH_FIG $SHOW_PROGRESS $PATH_DIFFU_SAMPLE $PATH_XT $NETWORK | tail -n 1)
echo "$MEMORY $SIGMA $THETA $MU $D0 $last_line" >> $PATH_ERROR_SAVE
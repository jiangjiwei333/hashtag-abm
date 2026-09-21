#!/bin/sh
# build.sh -- compile the simulation binary.
#
#   ./build.sh
#
# parallel_sim.sh compiles on its own, so this is only needed when calling
# simulation.sh directly (e.g. for a single run or a memory measurement).
#
# -march=native/-mtune=native are not accepted by Apple clang on arm64; we fall back.

CXX=${CXX:-g++}
CPP="./modules/simulation.cpp"
BIN="./modules/simulation"

CXXFLAGS=${CXXFLAGS:-"-std=c++17 -O3 -mtune=native -march=native"}
if ! $CXX $CXXFLAGS "$CPP" -o "$BIN" 2>/dev/null; then
    echo "note: retrying without -march/-mtune=native"
    CXXFLAGS="-std=c++17 -O3"
    if ! $CXX $CXXFLAGS "$CPP" -o "$BIN"; then
        echo "Compilation failed."
        exit 1
    fi
fi
echo "built $BIN  ($CXX $CXXFLAGS)"

#! /bin/bash

# Usage: ./clean.sh [--move]

move_files=false
if [ "$1" == "--move" ]; then
    move_files=true
fi

timestamp=$(date +"%Y-%m-%d-%H:%M")

if $move_files; then
    mkdir -p "/xdisk/bethard/kbozler/ds-run-outputs/ds-run-outputs_$timestamp"
    mv output/* "/xdisk/bethard/kbozler/ds-run-outputs/ds-run-outputs_$timestamp" 
    cp .run_multi.sh "/xdisk/bethard/kbozler/ds-run-outputs/ds-run-outputs_$timestamp"
    mv .hpclogs "/xdisk/bethard/kbozler/ds-run-outputs/ds-run-outputs_$timestamp"
fi

# remove used directories after runs
rm -rf "output/unprocessed_outputs"
rm -rf "output/processed_outputs"
rm -rf "output/models"
rm -rf "output/run_args"
rm -rf "results"
rm -rf "config"
rm -rf ".hpclogs"

# make new directories
mkdir -p "output/unprocessed_outputs"
mkdir -p "output/processed_outputs"
mkdir -p "output/models"
mkdir -p "output/run_args"
mkdir -p "results"
mkdir -p "config"
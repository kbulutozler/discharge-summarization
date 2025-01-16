#! /bin/bash

# Usage: ./clean.sh [--move]

move_files=false
if [ "$1" == "--move" ]; then
    move_files=true
fi

timestamp=$(date +"%Y-%m-%d-%H:%M")

if $move_files; then
    mkdir -p "/media/networkdisk/bulut/ds-run-outputs/ds-run-outputs_$timestamp/logs"
    mv output/* "/media/networkdisk/bulut/ds-run-outputs/ds-run-outputs_$timestamp" 
    mv .run_pbs_files "/media/networkdisk/bulut/ds-run-outputs/ds-run-outputs_$timestamp"
    mv job.* "/media/networkdisk/bulut/ds-run-outputs/ds-run-outputs_$timestamp/logs"
fi

# remove used directories after runs
rm -rf "output/unprocessed_outputs"
rm -rf "output/processed_outputs"
rm -rf "output/models"
rm -rf "output/run_args"
rm -rf "results"
rm -rf "config"
rm -rf ".run_pbs_files"

# remove error and output logs 
rm -rf *.stderr.txt
rm -rf *.stdout.txt

# make new directories
mkdir -p "output/unprocessed_outputs"
mkdir -p "output/processed_outputs"
mkdir -p "output/models"
mkdir -p "output/run_args"
mkdir -p "results"
mkdir -p "config"
mkdir -p ".run_pbs_files"

# kill processes
pkill -u bulut python
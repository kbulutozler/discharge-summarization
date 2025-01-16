#! /bin/bash

# Usage: ./clean.sh --keep

keep_outputs=false
if [ "$1" == "--keep" ]; then
    keep_outputs=true
fi

# remove used directories after runs
if ! $keep_outputs; then
    rm -rf /media/networkdisk/bulut/run-outputs/discharge-summarization/qlora/*
fi

rm -rf "config"
rm -rf ".run_pbs_files"

# remove error and output logs 
rm -rf *.stderr.txt
rm -rf *.stdout.txt

# make new directories
mkdir -p "config"
mkdir -p ".run_pbs_files"

# kill processes
pkill -u bulut python 




#! /bin/bash

# remove used directories after runs
rm -rf "data/custom_split"
rm -rf "output/unprocessed_outputs"
rm -rf "output/processed_outputs"
rm -rf "output/models"
rm -rf "output/run_args"
rm -rf "results"
rm -rf "config"
rm -rf ".hpclogs"


mkdir -p "data/custom_split"
mkdir -p "output/unprocessed_outputs"
mkdir -p "output/processed_outputs"
mkdir -p "output/models"
mkdir -p "output/run_args"
mkdir -p "results"
mkdir -p "config"

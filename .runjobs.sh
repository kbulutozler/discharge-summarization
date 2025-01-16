#!/bin/bash

# for each file under .run_pbs_files, run the pbs script
for file in .run_pbs_files/*.pbs; do
    qsub $file
    # wait 10 seconds
done

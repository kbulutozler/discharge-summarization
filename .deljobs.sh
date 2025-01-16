#bin/bash

# get job ids from stderr files where each file is like job.id.stderr.txt
# and delete them

for file in job.*.stderr.txt; do
    job_id=$(echo $file | grep -oP 'job\.\K\d+');
    qdel $job_id;
done

pkill -u bulut python
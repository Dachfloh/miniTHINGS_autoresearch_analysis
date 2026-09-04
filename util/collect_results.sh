#!/bin/bash
read -p "Enter run tag: " RUN_TAG

for((i=1;i<=5;i++)); do
	echo $i
	mkdir "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch_analysis/past_runs/agent$i/$RUN_TAG"
	mv "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent$i/results.tsv" "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch_analysis/past_runs/agent$i/$RUN_TAG"
done

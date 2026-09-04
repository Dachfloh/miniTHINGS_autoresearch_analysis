#!/bin/bash
for((i=1;i<=5;i++)); do	
	echo $i
	rm "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent$i/run.log"
	rm -rf "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent$i/results/rnn_decoding/"
	git -C "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent$i" checkout "agent$i/main"
done

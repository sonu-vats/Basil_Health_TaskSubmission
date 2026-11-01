#!/bin/bash
# usage: run3.sh TAG [extra args]  -> runs seeds 0 1 2, prints result lines
TAG=$1; shift
for s in 0 1 2; do python3 train.py --seed $s --tag "$TAG" "$@" 2>&1 | grep -v Warn > ../logs/p1_${TAG}_s$s.log; tail -1 ../logs/p1_${TAG}_s$s.log; done

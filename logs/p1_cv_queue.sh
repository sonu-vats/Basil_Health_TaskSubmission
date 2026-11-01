#!/bin/bash
cd /home/claude/repo/part1_diagnose
for s in 0 1; do
  python3 train.py --cv --channels all     --seed $s --threads 1 --tag cv-all     2>&1 | grep -v Warn | grep -E "CV|subjects" 
  python3 train.py --cv --channels frontal --seed $s --threads 1 --tag cv-frontal 2>&1 | grep -v Warn | grep -E "CV|subjects"
  python3 train.py --cv --channels central --seed $s --threads 1 --tag cv-central 2>&1 | grep -v Warn | grep -E "CV|subjects"
  python3 train.py --cv --channels frontal --band 8 30 --seed $s --threads 1 --tag cv-frontal-8-30Hz 2>&1 | grep -v Warn | grep -E "CV|subjects"
  python3 train.py --cv --channels all     --band 8 30 --seed $s --threads 1 --tag cv-all-8-30Hz     2>&1 | grep -v Warn | grep -E "CV|subjects"
done
echo QUEUE_DONE

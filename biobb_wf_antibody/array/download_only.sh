#!/bin/bash
for ID in $(seq 0 15); do
    python launch_wf.py --index $ID --out-dir results -d
done
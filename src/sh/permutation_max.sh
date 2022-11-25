#!/bin/bash

# run a permutation test for the specified parameters

PROJECT_ROOT=$HOME/implied_motion

for design in compact expanded
do
    for target in lr fb
    do
        for condition in real implied cross
        do
            python "$PROJECT_ROOT"/src/Py/permutation_max.py \
                --workingdir "$PROJECT_ROOT" \
                --targetdecode $target \
                --design $design \
                --condition $condition
        done             
    done
done

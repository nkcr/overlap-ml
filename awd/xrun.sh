#!/bin/bash
# This script is used to store common runs

if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Usage: xrun.sh <id>"
fi

case $1 in
    1)
    # For simple local computation on CPU
    python3 main.py --batch_size 20 --data data/penn --dropouti 0.4 \
                    --dropouth 0.25 --seed 141 --epoch 1 \
                    --nhid 5 --emsize 5 --nlayers 1 --bptt 5
    ;;
    10)
    # For simple local computation on CPU
    python3 main_ds.py --batch_size 20 --data data/penn --dropouti 0.4 \
                    --dropouth 0.25 --seed 141 --epoch 7 \
                    --nhid 5 --emsize 4 --nlayers 1 --bptt 5 $2
    ;;
    6)
    # Vanilla execution
    python3 main.py --batch_size 20 --data data/penn --dropouti 0.4 \
                    --dropouth 0.25 --seed 141 --epoch 500 
    ;;
    60)
    # Vanilla execution with main_ds
    python3 main_ds.py --batch_size 20 --data data/penn --dropouti 0.4 \
                    --dropouth 0.25 --seed 141 --epoch 500 $2
    ;;
esac
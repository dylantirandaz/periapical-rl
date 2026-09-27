#!/bin/bash
# Publish every listed run to the site in turn, every 90 s, until the file STOP exists.
cd ~/periapical-rl
export HF_HUB_OFFLINE=1
while [ ! -f STOP ]; do
  while read -r dir id stage steps name; do
    [ -z "$dir" ] && continue
    [ -d "$dir" ] && ~/tinker-rl/.venv/bin/python sync.py --run-dir "$dir" --run-id "$id" --name "$name" --stage "$stage" --max-steps "$steps" --once >> sync.log 2>&1
  done < runs.txt
  sleep 90
done

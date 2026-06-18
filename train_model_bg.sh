#!/bin/bash

nohup stdbuf -oL -eL python3 -u heuristic_nn.py >> bot_output.log 2>&1 &
tail -f bot_output.log
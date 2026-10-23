#!/bin/bash

# Ensure script runs from the repository root
cd "$(dirname "$0")"

LOG_FILE="bot_output.log"

rm -f "$LOG_FILE"

nohup uv run python -u -m agent.train_rl > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training started in background with PID: $PID"
echo "Log file: $LOG_FILE"
echo "To monitor logs in real-time, run:"
echo "  tail -f $LOG_FILE"
echo "To stop training, run:"
echo "  kill $PID"

#!/bin/bash
cd "$(dirname "$0")"

case "${1:-all}" in
  supervised) CMD="uv run python -u -m agent.train_supervised" ;;
  rl)         CMD="uv run python -u -m agent.train_rl" ;;
  all)        CMD="uv run python -u -m agent.train_supervised && uv run python -u -m agent.train_rl" ;;
  *)          echo "Usage: $0 [all|supervised|rl]"; exit 1 ;;
esac

nohup bash -c "$CMD" > bot_output.log 2>&1 &
echo "Started ${1:-all} training (PID: $!). Logs: tail -f bot_output.log"

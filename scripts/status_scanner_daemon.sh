#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PID_FILE="$DIR/logs/daemon.pid"
LOG_FILE="$DIR/logs/daemon.log"

if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
    echo "🟢 Daemon is RUNNING (PID: $(cat "$PID_FILE"))"
    echo "Last 5 log entries:"
    tail -n 5 "$LOG_FILE"
else
    echo "🔴 Daemon is STOPPED."
fi

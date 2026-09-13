#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PID_FILE="$DIR/logs/daemon.pid"
LOG_FILE="$DIR/logs/daemon.log"

if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
    echo "Daemon is already running (PID: $(cat "$PID_FILE"))."
    exit 0
fi

echo "Starting Project Intelligence Background Daemon..."
nohup python3 "$DIR/scanner/daemon.py" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Daemon started successfully (PID: $(cat "$PID_FILE"))."
echo "Live logs available at: $LOG_FILE"

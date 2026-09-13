#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PID_FILE="$DIR/logs/daemon.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping daemon (PID: $PID)..."
        kill -15 "$PID"
        rm -f "$PID_FILE"
        echo "Daemon stopped."
    else
        echo "Daemon is not running. Cleaning up stale PID file."
        rm -f "$PID_FILE"
    fi
else
    echo "No daemon PID file found. Daemon is not running."
fi

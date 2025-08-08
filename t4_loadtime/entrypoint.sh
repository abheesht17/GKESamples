#!/bin/bash

# Get current time in seconds since epoch (with milliseconds)
now=$(date +%s.%3N)

# Compute seconds until next 5-minute mark
next_mark=$(date -d "now + $((300 - $(date +%s) % 300)) seconds" +%s)

# Calculate duration to sleep (float)
sleep_sec=$(awk "BEGIN { print $next_mark - $now }")

echo "Sleeping for $sleep_sec seconds until next 5-minute boundary..."
sleep $sleep_sec

echo "[$(date +"%H:%M:%S.%3N")] Starting gst-launch"
gst-launch-1.0 videotestsrc is-live=true ! nvh264enc ! fakesink

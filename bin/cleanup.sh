#!/bin/bash

# Command output from 'hare me'
output=$(hare me)

# Extract container names using grep and awk
container_names=$(echo "$output" | grep -Eo '^[[:space:]]+[a-zA-Z0-9_]+[[:space:]]+\|' | awk '{print $1}')

# Loop through container names and run 'hare rm' command
for container in $container_names; do
    echo "Removing container: $container"
    hare kill "$container"
    hare rm "$container"
done

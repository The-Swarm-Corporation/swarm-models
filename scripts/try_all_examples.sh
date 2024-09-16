#!/bin/bash

# Create a log file
LOG_FILE="execution_log.json"
echo "[]" > $LOG_FILE

# Function to execute a file and log the result
execute_file() {
    local file=$1
    local start_time=$(date +%s)
    
    # Execute the file
    if bash "$file"; then
        status="success"
    else
        status="failure"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Log the result in JSON format
    log_entry=$(jq -n \
        --arg file "$file" \
        --arg status "$status" \
        --arg duration "$duration" \
        '{file: $file, status: $status, duration: $duration}')
    
    # Append to the log file
    echo "$log_entry" >> $LOG_FILE
}

export -f execute_file

# Find all shell scripts in the examples folder and run them in parallel
find examples/ -type f -name "*.sh" | parallel -j 4 execute_file

# Output the final log
cat $LOG_FILE
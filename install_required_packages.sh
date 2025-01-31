#!/bin/bash

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "------------------------------------------------------------"
    echo "Please run this script as root or with sudo privileges."
    echo "------------------------------------------------------------"
    exit 1
fi

# Check if apt-get is available (specific to Debian/Ubuntu systems)
if ! command -v apt-get &> /dev/null; then
    echo "------------------------------------------------------------"
    echo "This script is intended for Debian/Ubuntu-based systems."
    echo "apt-get command not found. Exiting."
    echo "------------------------------------------------------------"
    exit 1
fi

# Check if portaudio19-dev is already installed
if dpkg -l | grep -q "portaudio19-dev"; then
    echo "------------------------------------------------------------"
    echo "portaudio19-dev is already installed."
    echo "------------------------------------------------------------"
else
    echo "------------------------------------------------------------"
    echo "Installing portaudio19-dev..."
    echo "------------------------------------------------------------"
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev
    if [ $? -eq 0 ]; then
        echo "------------------------------------------------------------"
        echo "portaudio19-dev installed successfully."
        echo "------------------------------------------------------------"
    else
        echo "------------------------------------------------------------"
        echo "Failed to install portaudio19-dev. Please check for errors."
        echo "------------------------------------------------------------"
        exit 1
    fi
fi

echo "------------------------------------------------------------"
echo "Dependencies for PyAudio installed successfully."
echo "------------------------------------------------------------"

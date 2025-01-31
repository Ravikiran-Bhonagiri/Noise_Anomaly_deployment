#!/bin/bash

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "------------------------------------------------------------"
    echo "Git is not installed. Please install Git and try again."
    echo "------------------------------------------------------------"
    exit 1
fi

# Check if the repository URL is provided
if [ -z "$1" ]; then
    echo "------------------------------------------------------------"
    echo "Usage: $0 <repository_url>"
    echo "Example: $0 https://github.com/user/repo.git"
    echo "------------------------------------------------------------"
    exit 1
fi

# Variables
REPO_URL=$1
REPO_NAME=$(basename -s .git "$REPO_URL")
MAIN_BRANCH="main"

# Clone the repository
echo "------------------------------------------------------------"
echo "Cloning the repository from $REPO_URL..."
echo "------------------------------------------------------------"
git clone "$REPO_URL"
if [ $? -ne 0 ]; then
    echo "------------------------------------------------------------"
    echo "Failed to clone the repository. Please check the URL and try again."
    echo "------------------------------------------------------------"
    exit 1
fi

# Navigate to the repository directory
cd "$REPO_NAME" || { 
    echo "------------------------------------------------------------"
    echo "Failed to enter the repository directory. Exiting."
    echo "------------------------------------------------------------"
    exit 1
}

# Check out the main branch
echo "------------------------------------------------------------"
echo "Switching to the $MAIN_BRANCH branch..."
echo "------------------------------------------------------------"
git checkout "$MAIN_BRANCH"
if [ $? -ne 0 ]; then
    echo "------------------------------------------------------------"
    echo "Failed to switch to the $MAIN_BRANCH branch. Please check if it exists."
    echo "------------------------------------------------------------"
    exit 1
fi

# Confirm success
echo "------------------------------------------------------------"
echo "Repository cloned and switched to the $MAIN_BRANCH branch successfully."
echo "Current directory: $(pwd)"
echo "------------------------------------------------------------"

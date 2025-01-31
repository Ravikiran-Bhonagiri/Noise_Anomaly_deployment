#!/bin/bash

# Define the virtual environment directory
VENV_DIR="venv"

# Function to install Python
install_python() {
    echo "------------------------------------------------------------"
    echo "Python3 is not installed. Attempting to install Python3..."
    echo "------------------------------------------------------------"

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux installation
        if command -v apt-get &> /dev/null; then
            echo "------------------------------------------------------------"
            echo "Using apt-get to install Python3..."
            echo "------------------------------------------------------------"
            sudo apt-get update
            sudo apt-get install -y python3 python3-venv python3-pip
        elif command -v yum &> /dev/null; then
            echo "------------------------------------------------------------"
            echo "Using yum to install Python3..."
            echo "------------------------------------------------------------"
            sudo yum install -y python3 python3-venv python3-pip
        else
            echo "------------------------------------------------------------"
            echo "Unsupported package manager. Please install Python3 manually."
            echo "------------------------------------------------------------"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS installation
        if ! command -v brew &> /dev/null; then
            echo "------------------------------------------------------------"
            echo "Homebrew is not installed. Installing Homebrew..."
            echo "------------------------------------------------------------"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        echo "------------------------------------------------------------"
        echo "Using Homebrew to install Python3..."
        echo "------------------------------------------------------------"
        brew install python
    else
        echo "------------------------------------------------------------"
        echo "Unsupported OS. Please install Python3 manually."
        echo "------------------------------------------------------------"
        exit 1
    fi

    if ! command -v python3 &> /dev/null; then
        echo "------------------------------------------------------------"
        echo "Python3 installation failed. Please install it manually."
        echo "------------------------------------------------------------"
        exit 1
    fi

    echo "------------------------------------------------------------"
    echo "Python3 installed successfully."
    echo "------------------------------------------------------------"
}

# Check if Python3 is installed
echo "------------------------------------------------------------"
if ! command -v python3 &> /dev/null; then
    install_python
else
    echo "Python3 is already installed."
fi
echo "------------------------------------------------------------"

# Check if the virtual environment directory exists
if [ -d "$VENV_DIR" ]; then
    echo "------------------------------------------------------------"
    echo "The virtual environment directory '$VENV_DIR' already exists."
    echo "Deleting the directory..."
    echo "------------------------------------------------------------"
    rm -rf "$VENV_DIR"
    echo "------------------------------------------------------------"
    echo "Directory '$VENV_DIR' deleted successfully."
    echo "------------------------------------------------------------"
fi

# Create the virtual environment
echo "------------------------------------------------------------"
echo "Creating virtual environment in the directory: $VENV_DIR..."
echo "------------------------------------------------------------"
python3 -m venv $VENV_DIR

# Activate the virtual environment
echo "------------------------------------------------------------"
echo "Activating the virtual environment..."
echo "------------------------------------------------------------"
source $VENV_DIR/bin/activate

# Check if requirements.txt exists
echo "------------------------------------------------------------"
if [ -f "requirements.txt" ]; then
    echo "Installing required packages from requirements.txt..."
    echo "------------------------------------------------------------"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "requirements.txt not found. No packages will be installed."
    echo "------------------------------------------------------------"
fi

echo "------------------------------------------------------------"
echo "Virtual environment setup completed."
echo "------------------------------------------------------------"

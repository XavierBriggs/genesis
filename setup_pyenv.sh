#!/bin/bash
set -e

echo "Setting up pyenv on Linux Mint with Python 3.7.17 environment..."

# Check if running on Linux Mint
check_os() {
    echo "Checking operating system..."
    if [ ! -f /etc/os-release ]; then
        echo "ERROR: Cannot determine OS. /etc/os-release not found."
        exit 1
    fi
    
    if ! grep -qi "mint\|ubuntu" /etc/os-release; then
        echo "WARNING: This script is designed for Linux Mint/Ubuntu. Your OS may not be supported."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    echo "OS check passed."
}

# Check system requirements
check_requirements() {
    echo "Checking system requirements..."
    
    # Check available disk space (need at least 2GB)
    available_space=$(df / | awk 'NR==2 {print $4}')
    required_space=2097152  # 2GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        echo "ERROR: Insufficient disk space. Need at least 2GB free."
        echo "Available: $(echo "scale=1; $available_space/1024/1024" | bc)GB"
        exit 1
    fi
    
    # Check if we have internet connectivity
    if ! ping -c 1 google.com &> /dev/null; then
        echo "ERROR: No internet connection detected. Internet required for downloads."
        exit 1
    fi
    
    # Check if we have sudo privileges
    if ! sudo -n true 2>/dev/null; then
        echo "Checking sudo privileges..."
        sudo -v
    fi
    
    echo "System requirements check passed."
}

# Check if Python dependencies can be compiled
check_build_deps() {
    echo "Checking for required build tools..."
    
    missing_tools=()
    for tool in gcc make git curl; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        echo "Missing build tools: ${missing_tools[*]}"
        echo "These will be installed automatically."
    fi
}

# Run all checks
check_os
check_requirements
check_build_deps

echo "Pre-installation checks completed successfully!"
echo ""

# Update system packages
echo "Updating system packages..."
sudo apt update

# Install dependencies for pyenv and Python compilation
echo "Installing dependencies for pyenv..."
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev git bc

# Install pyenv
echo "Installing pyenv..."
if [ ! -d "$HOME/.pyenv" ]; then
    curl https://pyenv.run | bash
else
    echo "pyenv already installed, updating..."
    cd ~/.pyenv && git pull
fi

# Add pyenv to PATH and shell initialization
echo "Configuring shell environment..."
if ! grep -q 'pyenv' ~/.bashrc; then
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
fi

# Source the changes for current session
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Install Python 3.7.17
echo "Installing Python 3.7.17..."
pyenv install -s 3.7.17

# Create virtual environment
echo "Creating virtual environment 'sonic-ppo'..."
pyenv virtualenv 3.7.17 sonic-ppo

# Set local Python version for this project
echo "Setting local Python version for project..."
pyenv local sonic-ppo

# Upgrade pip and install requirements
echo "Installing project requirements..."
pip install --upgrade pip
pip install -r sonic-ppo-requirements.txt

echo "Setup complete!"
echo "To activate this environment in new terminal sessions, run:"
echo "pyenv activate sonic-ppo"
echo ""
echo "Or if you're in the project directory, the environment should activate automatically."
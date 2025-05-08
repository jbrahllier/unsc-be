#!/bin/bash

# NOTE: run 'chmod +x install.sh' before running if running independently

# detect operating system
OS="$(uname -s)"

if [ "${OS}" != "Darwin" ]; then
    echo "This script is only supported on macOS."
    exit 1
fi

echo "Detected macOS."

# detect architecture
ARCH=$(uname -m)

# check if homebrew is installed
if ! command -v brew &> /dev/null
then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    echo "Checking for Intel or Apple Silicon..."
    if [ "$ARCH" = "arm64" ]; then
        # apple silicon
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.profile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ "$ARCH" = "x86_64" ]; then
        # intel
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.profile
        eval "$(/usr/local/bin/brew shellenv)"
    else
        echo "Unsupported architecture: ${ARCH}"
        exit 1
    fi
fi

# check if libreoffice is installed (either in /Applications or ~/Applications)
if [ ! -d "/Applications/LibreOffice.app" ] && [ ! -d "$HOME/Applications/LibreOffice.app" ]; then
    echo "LibreOffice not found. Installing LibreOffice..."
    brew install --cask libreoffice
else
    echo "LibreOffice is already installed."
fi

# check if unoconv is installed
if ! brew list unoconv &>/dev/null
then
    echo "Installing unoconv..."
    brew install unoconv
else
    echo "unoconv is already installed and up-to-date."
fi

# check if python packages are installed
check_python_packages() {
    while IFS= read -r package
    do
        # check if each package is installed
        if ! pip show "$package" &> /dev/null
        then
            echo "Missing Python package: $package"
            return 1
        fi
    done < <(grep -v '^#' requirements.txt | awk '{print $1}')
}
if check_python_packages; then
    echo "All Python packages from requirements.txt are already installed."
else
    echo "Installing missing Python packages..."
    pip install -r requirements.txt
fi

# final dependency check
if command -v brew &> /dev/null && ([ -d "/Applications/LibreOffice.app" ] || [ -d "$HOME/Applications/LibreOffice.app" ]) && brew list unoconv &>/dev/null && check_python_packages
then
    echo "All dependencies are already fulfilled."
else
    echo "All dependencies have been installed."
fi

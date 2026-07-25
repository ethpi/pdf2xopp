#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
progName="pdf2xopp"
executablePath="/usr/local/bin"
iconPath="/usr/share/pixmaps"
desktopFilePath="/usr/share/applications"

echo "Installing $progName..."

# --- Install Executable ---
echo "Copying executable to $executablePath/$progName..."
sudo cp -iv "$progName.py" "$executablePath/$progName"
sudo chown root:root "$executablePath/$progName"
sudo chmod 755 "$executablePath/$progName"

# --- Install Desktop File ---
echo "Copying desktop file to $desktopFilePath..."
sudo cp -iv "$progName.desktop" "$desktopFilePath/$progName.desktop"
sudo chown root:root "$desktopFilePath/$progName.desktop"
sudo chmod 644 "$desktopFilePath/$progName.desktop"

# --- Install Icon ---
echo "Copying icon to $iconPath..."
sudo cp -iv "$progName.png" "$iconPath/$progName.png"
sudo chown root:root "$iconPath/$progName.png"
sudo chmod 644 "$iconPath/$progName.png"

# --- Update System Databases ---
echo "Updating desktop database..."
sudo update-desktop-database "$desktopFilePath"

echo "Installation of $progName complete!"
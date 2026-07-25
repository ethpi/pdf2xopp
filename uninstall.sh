#!/bin/sh
set -e

progName="pdf2xopp"
executablePath="/usr/local/bin"
iconPath="/usr/share/pixmaps"
desktopFilePath="/usr/share/applications"

echo "Uninstalling $progName..."

sudo rm -vf "$executablePath/$progName"
sudo rm -vf "$desktopFilePath/$progName.desktop"
sudo rm -vf "$iconPath/$progName.png"

echo "Updating desktop database..."
sudo update-desktop-database "$desktopFilePath"

echo "Uninstallation of $progName complete."
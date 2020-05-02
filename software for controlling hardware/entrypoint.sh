#!/bin/bash

set -e 
cd sw

# Dependencies installation 
# OpenCV and h5py installed on the system via apt to satisfy library reqs
sudo pip3 install -r requirements.txt
mkdir -p logs
mkdir -p img

# Application startup
sudo pkill python3 || true
sudo python3 main.py --platform linuxfb

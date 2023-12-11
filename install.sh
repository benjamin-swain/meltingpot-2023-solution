#!/bin/bash

ASSETS_URL="http://storage.googleapis.com/dm-meltingpot/meltingpot-assets-2.1.0.tar.gz"
MELTINGPOT_DIR="./meltingpot"
mkdir -p "$MELTINGPOT_DIR"
echo "Downloading Melting Pot assets..."
curl -L "$ASSETS_URL" | tar -xz -C "$MELTINGPOT_DIR"

pip install -r requirements.txt
pip install ./feature_detector_territory
pip install ./feature_detector_clean_up

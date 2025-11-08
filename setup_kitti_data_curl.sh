#!/bin/bash
# Download and organize KITTI data for sensor fusion project using curl

echo "🚀 Setting up KITTI data for sensor fusion project..."

# Create download directory
mkdir -p data/raw/kitti/downloads
cd data/raw/kitti/downloads

echo "📥 Downloading key KITTI sequences using curl..."

# Download calibration data first
echo "📐 Downloading calibration data..."
curl -L -C - -o 2011_09_26_calib.zip "http://www.cvlibs.net/download.php?file=2011_09_26_calib.zip"

# Download a few key driving sequences for testing
echo "🚗 Downloading driving sequences..."

# City driving sequence
curl -L -C - -o 2011_09_26_drive_0001.zip "http://www.cvlibs.net/download.php?file=2011_09_26_drive_0001"

# Highway driving sequence  
curl -L -C - -o 2011_09_26_drive_0002.zip "http://www.cvlibs.net/download.php?file=2011_09_26_drive_0002"

# Residential driving sequence
curl -L -C - -o 2011_09_26_drive_0005.zip "http://www.cvlibs.net/download.php?file=2011_09_26_drive_0005"

echo "📦 Extracting downloaded files..."

# Extract calibration
if [ -f "2011_09_26_calib.zip" ]; then
    unzip -q 2011_09_26_calib.zip
    echo "✅ Extracted calibration data"
else
    echo "⚠️  Calibration file not found"
fi

# Extract driving sequences
if [ -f "2011_09_26_drive_0001.zip" ]; then
    unzip -q 2011_09_26_drive_0001.zip
    echo "✅ Extracted drive 0001"
else
    echo "⚠️  Drive 0001 file not found"
fi

if [ -f "2011_09_26_drive_0002.zip" ]; then
    unzip -q 2011_09_26_drive_0002.zip
    echo "✅ Extracted drive 0002"
else
    echo "⚠️  Drive 0002 file not found"
fi

if [ -f "2011_09_26_drive_0005.zip" ]; then
    unzip -q 2011_09_26_drive_0005.zip
    echo "✅ Extracted drive 0005"
else
    echo "⚠️  Drive 0005 file not found"
fi

echo "🔄 Organizing data into correct structure..."

# Go back to project root
cd ../../../..

# Run the organization script
python3 src/utils/organize_kitti.py --action organize --extracted_path data/raw/kitti/downloads

echo "✅ KITTI data setup complete!"
echo ""
echo "📊 Data summary:"
echo "  - Calibration files: data/raw/kitti/training/calib/"
echo "  - Camera images: data/raw/kitti/training/image_2/"
echo "  - LiDAR data: data/raw/kitti/training/velodyne/"
echo ""
echo "🚀 Ready to run sensor fusion pipeline!"

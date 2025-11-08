#!/bin/bash
# Download and organize KITTI data for sensor fusion project

echo "🚀 Setting up KITTI data for sensor fusion project..."

# Create download directory
mkdir -p data/raw/kitti/downloads
cd data/raw/kitti/downloads

echo "📥 Downloading key KITTI sequences..."

# Download calibration data first
echo "📐 Downloading calibration data..."
wget -c "http://www.cvlibs.net/download.php?file=2011_09_26_calib.zip" -O 2011_09_26_calib.zip

# Download a few key driving sequences for testing
echo "🚗 Downloading driving sequences..."

# City driving sequence
wget -c "http://www.cvlibs.net/download.php?file=2011_09_26_drive_0001" -O 2011_09_26_drive_0001.zip

# Highway driving sequence  
wget -c "http://www.cvlibs.net/download.php?file=2011_09_26_drive_0002" -O 2011_09_26_drive_0002.zip

# Residential driving sequence
wget -c "http://www.cvlibs.net/download.php?file=2011_09_26_drive_0005" -O 2011_09_26_drive_0005.zip

echo "📦 Extracting downloaded files..."

# Extract calibration
unzip -q 2011_09_26_calib.zip

# Extract driving sequences
unzip -q 2011_09_26_drive_0001.zip
unzip -q 2011_09_26_drive_0002.zip  
unzip -q 2011_09_26_drive_0005.zip

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

#!/bin/bash
# Download and organize ALL KITTI city clips for sensor fusion dataset

echo "🏙️ Setting up COMPLETE KITTI city dataset for sensor fusion..."

# Create download directory
mkdir -p data/raw/kitti/downloads
cd data/raw/kitti/downloads

echo "📥 Downloading ALL city sequences..."

# Download calibration data first
echo "📐 Downloading calibration data..."
curl -L -C - -o 2011_09_26_calib.zip "http://www.cvlibs.net/download.php?file=2011_09_26_calib.zip"

# Download ALL city driving sequences (2011_09_26)
echo "🚗 Downloading ALL city sequences from 2011_09_26..."

city_sequences=(
    "2011_09_26_drive_0001"  # City driving
    "2011_09_26_drive_0002"  # Highway driving  
    "2011_09_26_drive_0005"  # Residential driving
    "2011_09_26_drive_0009"  # City driving
    "2011_09_26_drive_0011"  # City driving
    "2011_09_26_drive_0013"  # City driving
    "2011_09_26_drive_0014"  # City driving
    "2011_09_26_drive_0015"  # City driving
    "2011_09_26_drive_0017"  # City driving
    "2011_09_26_drive_0018"  # City driving
    "2011_09_26_drive_0019"  # City driving
    "2011_09_26_drive_0020"  # City driving
    "2011_09_26_drive_0022"  # City driving
    "2011_09_26_drive_0023"  # City driving
    "2011_09_26_drive_0027"  # City driving
    "2011_09_26_drive_0028"  # City driving
    "2011_09_26_drive_0029"  # City driving
    "2011_09_26_drive_0032"  # City driving
    "2011_09_26_drive_0035"  # City driving
    "2011_09_26_drive_0036"  # City driving
    "2011_09_26_drive_0039"  # City driving
    "2011_09_26_drive_0046"  # City driving
    "2011_09_26_drive_0048"  # City driving
    "2011_09_26_drive_0051"  # City driving
    "2011_09_26_drive_0052"  # City driving
    "2011_09_26_drive_0056"  # City driving
    "2011_09_26_drive_0057"  # City driving
    "2011_09_26_drive_0059"  # City driving
    "2011_09_26_drive_0060"  # City driving
    "2011_09_26_drive_0061"  # City driving
    "2011_09_26_drive_0064"  # City driving
    "2011_09_26_drive_0070"  # City driving
    "2011_09_26_drive_0079"  # City driving
    "2011_09_26_drive_0084"  # City driving
    "2011_09_26_drive_0086"  # City driving
    "2011_09_26_drive_0087"  # City driving
    "2011_09_26_drive_0091"  # City driving
    "2011_09_26_drive_0093"  # City driving
    "2011_09_26_drive_0095"  # City driving
    "2011_09_26_drive_0096"  # City driving
    "2011_09_26_drive_0101"  # City driving
    "2011_09_26_drive_0104"  # City driving
    "2011_09_26_drive_0106"  # City driving
    "2011_09_26_drive_0113"  # City driving
    "2011_09_26_drive_0117"  # City driving
    "2011_09_26_drive_0119"  # City driving
)

# Download each sequence
for seq in "${city_sequences[@]}"; do
    echo "📥 Downloading $seq..."
    curl -L -C - -o "${seq}.zip" "http://www.cvlibs.net/download.php?file=${seq}"
    
    # Check if download was successful
    if [ -f "${seq}.zip" ] && [ -s "${seq}.zip" ]; then
        echo "✅ $seq downloaded successfully"
    else
        echo "⚠️  $seq download failed or empty"
    fi
done

echo "📦 Extracting downloaded files..."

# Extract calibration
if [ -f "2011_09_26_calib.zip" ] && [ -s "2011_09_26_calib.zip" ]; then
    unzip -q 2011_09_26_calib.zip
    echo "✅ Extracted calibration data"
else
    echo "⚠️  Calibration file not found or empty"
fi

# Extract all sequences
for seq in "${city_sequences[@]}"; do
    if [ -f "${seq}.zip" ] && [ -s "${seq}.zip" ]; then
        echo "📦 Extracting $seq..."
        unzip -q "${seq}.zip"
        echo "✅ Extracted $seq"
    else
        echo "⚠️  Skipping $seq (file not found or empty)"
    fi
done

echo "🔄 Organizing data into correct structure..."

# Go back to project root
cd ../../../..

# Run the organization script
python3 src/utils/organize_kitti.py --action organize --extracted_path data/raw/kitti/downloads

echo "✅ Complete KITTI city dataset setup finished!"
echo ""
echo "📊 Dataset summary:"
echo "  - Calibration files: data/raw/kitti/training/calib/"
echo "  - Camera images: data/raw/kitti/training/image_2/"
echo "  - LiDAR data: data/raw/kitti/training/velodyne/"
echo ""
echo "🎯 Total sequences: ${#city_sequences[@]} city driving sequences"
echo "🚀 Ready for large-scale sensor fusion training!"

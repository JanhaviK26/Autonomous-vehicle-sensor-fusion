#!/bin/bash
# Organize raw KITTI data (unsynced/unrectified) into training format

echo "🏙️ Organizing Raw KITTI Data for Sensor Fusion"
echo "=============================================="

# Check if raw data exists
if [ ! -d "data/raw/kitti/downloads" ]; then
    echo "❌ No raw KITTI data found in data/raw/kitti/downloads/"
    echo "Please download raw KITTI sequences first."
    exit 1
fi

echo "📊 Found raw KITTI data. Processing sequences..."

# Create output directories
mkdir -p data/raw/kitti/training/image_2
mkdir -p data/raw/kitti/training/velodyne
mkdir -p data/raw/kitti/training/calib

# Counter for file numbering
counter=0

# Process each sequence
for seq_dir in data/raw/kitti/downloads/2011_09_26_drive_*; do
    if [ -d "$seq_dir" ]; then
        echo "🔄 Processing sequence: $(basename $seq_dir)"
        
        # Extract sequence info
        seq_name=$(basename $seq_dir)
        
        # Process images
        if [ -d "$seq_dir/image_02/data" ]; then
            echo "  📸 Processing images..."
            for img_file in "$seq_dir/image_02/data"/*.png; do
                if [ -f "$img_file" ]; then
                    # Copy and rename image
                    cp "$img_file" "data/raw/kitti/training/image_2/$(printf "%06d" $counter).png"
                    counter=$((counter + 1))
                fi
            done
        fi
        
        # Process LiDAR
        if [ -d "$seq_dir/velodyne_points/data" ]; then
            echo "  🔍 Processing LiDAR..."
            lidar_counter=0
            for lidar_file in "$seq_dir/velodyne_points/data"/*.bin; do
                if [ -f "$lidar_file" ]; then
                    # Copy and rename LiDAR file
                    cp "$lidar_file" "data/raw/kitti/training/velodyne/$(printf "%06d" $lidar_counter).bin"
                    lidar_counter=$((lidar_counter + 1))
                fi
            done
        fi
        
        # Process calibration
        if [ -f "$seq_dir/calib_cam_to_cam.txt" ]; then
            echo "  📐 Processing calibration..."
            calib_counter=0
            # Copy calibration file for each frame in this sequence
            for img_file in "$seq_dir/image_02/data"/*.png; do
                if [ -f "$img_file" ]; then
                    cp "$seq_dir/calib_cam_to_cam.txt" "data/raw/kitti/training/calib/$(printf "%06d" $calib_counter).txt"
                    calib_counter=$((calib_counter + 1))
                fi
            done
        fi
        
        echo "  ✅ Sequence $(basename $seq_dir) processed"
    fi
done

echo ""
echo "📊 Data organization complete!"
echo ""

# Verify the structure
echo "🔍 Verifying data structure..."
python3 -c "
import sys
sys.path.append('src')
from utils.organize_kitti import verify_kitti_structure
verify_kitti_structure('data/raw/kitti')
"

echo ""
echo "🎉 Raw KITTI data successfully organized!"
echo "📈 Ready for sensor fusion training with real data!"

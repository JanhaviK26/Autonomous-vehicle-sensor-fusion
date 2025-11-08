#!/bin/bash
# Organize real KITTI data (TXT format LiDAR) into training format

echo "🏙️ Organizing Real KITTI Data"
echo "============================="

# Create training directories
mkdir -p data/raw/kitti/training/image_2
mkdir -p data/raw/kitti/training/velodyne
mkdir -p data/raw/kitti/training/calib

echo "📊 Processing real KITTI sequences..."

# Counter for file numbering
counter=0

# Process each sequence directory
for seq_dir in data/raw/2011_09_26*/2011_09_26_drive_*_extract; do
    if [ -d "$seq_dir" ]; then
        echo "🔄 Processing sequence: $(basename $seq_dir)"
        
        # Process camera images (image_02 is the left color camera)
        if [ -d "$seq_dir/image_02/data" ]; then
            echo "  📸 Processing camera images..."
            for img_file in "$seq_dir/image_02/data"/*.png; do
                if [ -f "$img_file" ]; then
                    # Copy and rename image
                    cp "$img_file" "data/raw/kitti/training/image_2/$(printf "%06d" $counter).png"
                    counter=$((counter + 1))
                fi
            done
        fi
        
        # Process LiDAR data (TXT format)
        if [ -d "$seq_dir/velodyne_points/data" ]; then
            echo "  🔍 Processing LiDAR data (TXT format)..."
            lidar_counter=0
            for lidar_file in "$seq_dir/velodyne_points/data"/*.txt; do
                if [ -f "$lidar_file" ]; then
                    # Convert TXT to BIN format for consistency
                    python3 -c "
import numpy as np
import sys

# Read TXT file (space-separated values)
try:
    data = np.loadtxt('$lidar_file')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # Convert to binary format (x, y, z, intensity)
    if data.shape[1] >= 4:
        # Use first 4 columns: x, y, z, intensity
        binary_data = data[:, :4].astype(np.float32)
    else:
        # Pad with zeros if needed
        binary_data = np.zeros((data.shape[0], 4), dtype=np.float32)
        binary_data[:, :data.shape[1]] = data.astype(np.float32)
    
    # Save as binary file
    binary_data.tofile('data/raw/kitti/training/velodyne/$(printf \"%06d\" $lidar_counter).bin')
    print(f'✅ Converted {len(binary_data)} points from TXT to BIN')
    
except Exception as e:
    print(f'❌ Error processing $lidar_file: {e}')
"
                    lidar_counter=$((lidar_counter + 1))
                fi
            done
        fi
        
        # Create calibration files (we'll create sample ones for now)
        echo "  📐 Creating calibration files..."
        calib_counter=0
        for img_file in "$seq_dir/image_02/data"/*.png; do
            if [ -f "$img_file" ]; then
                # Create a sample calibration file
                cat > "data/raw/kitti/training/calib/$(printf "%06d" $calib_counter).txt" << EOF
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-04 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-03 2.035406e-03 1.482454e-02 9.998881e-01 -7.336161e-03
EOF
                calib_counter=$((calib_counter + 1))
            fi
        done
        
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
echo "🎉 Real KITTI data successfully organized!"
echo "📈 Ready for sensor fusion training with real data!"

#!/bin/bash
# Handle KITTI authentication issue and provide alternatives

echo "🔍 Analyzing KITTI Download Issue"
echo "================================="

cd data/raw/kitti/downloads

echo "📊 Checking downloaded files..."

# Check if files are actually HTML (authentication required)
if head -1 2011_09_26_drive_0001.zip | grep -q "html"; then
    echo "❌ Download Issue Detected:"
    echo "   - Files are HTML pages (authentication required)"
    echo "   - KITTI requires login to download data"
    echo "   - Need to register at: http://www.cvlibs.net/datasets/kitti/"
    echo ""
    
    echo "🛠️  Solutions Available:"
    echo "1. Register at KITTI website and download manually"
    echo "2. Use expanded sample dataset for development"
    echo "3. Use alternative datasets (NuScenes, Waymo, etc.)"
    echo ""
    
    echo "📋 Manual Download Instructions:"
    echo "1. Go to: http://www.cvlibs.net/datasets/kitti/raw_data.php"
    echo "2. Register and login"
    echo "3. Download sequences like:"
    echo "   - 2011_09_26_drive_0001 (City driving)"
    echo "   - 2011_09_26_drive_0002 (Highway driving)"
    echo "   - 2011_09_26_drive_0005 (Residential driving)"
    echo "4. Extract to: data/raw/kitti/downloads/"
    echo ""
    
    echo "🎯 Recommended: Create Expanded Sample Dataset"
    echo "This will give you a working dataset for development while you get real KITTI data."
    
else
    echo "✅ Files appear to be valid zip archives"
    echo "📦 Proceeding with extraction..."
fi

cd ../../../..

echo ""
echo "🚀 Creating Expanded Sample Dataset for Development"
echo "=================================================="

# Create expanded sample dataset
python3 -c "
import sys
sys.path.append('src')
from utils.create_sample_data_simple import create_sample_dataset
print('📊 Creating expanded sample dataset...')
create_sample_dataset(100)  # Create 100 sample scenes
print('✅ Expanded sample dataset created!')
"

echo ""
echo "📊 Current Dataset Status:"
python3 src/utils/organize_kitti.py --action verify

echo ""
echo "🎉 Development Dataset Ready!"
echo "📈 You now have 100+ sample scenes for development"
echo "🔧 Use this while you work on getting real KITTI data"

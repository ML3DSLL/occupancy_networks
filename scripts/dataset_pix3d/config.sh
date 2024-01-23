ROOT=~/Documents/TUM/3rd_semester/ML43D/occupancy_networks

export MESHFUSION_PATH=$ROOT/external/mesh-fusion
export HDF5_USE_FILE_LOCKING=FALSE # Workaround for NFS mounts

INPUT_PATH=$ROOT/data/external/pix3d/model
IMG_PATH=$ROOT/data/external/pix3d/img
MASK_PATH=$ROOT/data/external/pix3d/mask
#CHOY2016_PATH=$ROOT/data/external/Choy2016
BUILD_PATH=$ROOT/data/pix3d.build
OUTPUT_PATH=$ROOT/data/pix3d

NPROC=12
TIMEOUT=180
N_VAL=100
N_TEST=100
N_AUG=50

declare -a CLASSES=(
bed 
bookcase
chair
desk
misc
sofa
table
tool
wardrobe
)

# Utility functions
lsfilter() {
 folder=$1
 other_folder=$2
 ext=$3

 for f in $folder/*; do
   filename=$(basename $f)
   if [ ! -f $other_folder/$filename$ext ] && [ ! -d $other_folder/$filename$ext ]; then
    echo $filename
   fi
 done
}

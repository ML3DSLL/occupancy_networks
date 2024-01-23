source dataset_pix3d/config.sh

# Function for processing a single model
reorganize_pix3d() {
  modelname=$(basename -- $3)
  output_path="$2/$modelname"
  build_path=$1
  # choy_vox_path=$2
  # choy_img_path=$1

  points_file="$build_path/4_points/$modelname.npz"
  points_out_file="$output_path/points.npz"

  pointcloud_file="$build_path/4_pointcloud/$modelname.npz"
  pointcloud_out_file="$output_path/pointcloud.npz"

  # vox_file="$choy_vox_path/$modelname/model.binvox"
  # vox_out_file="$output_path/model.binvox"
  vox_file="$build_path/5_voxels/$modelname.binvox"
  vox_out_file="$output_path/model.binvox"

  mask_dir="$build_path/6_masks/$modelname.png"
  mask_out_dir="$output_path/mask.png"

  # TODO:i choose jpg format of images
  img_dir="$build_path/6_images/$modelname.jpg"
  img_out_dir="$output_path/image.jpg"

  # metadata_file="$choy_img_path/$modelname/rendering/rendering_metadata.txt"
  # camera_out_file="$output_path/img_choy2016/cameras.npz"

  echo "Copying model $output_path"
  mkdir -p $output_path

  cp $points_file $points_out_file
  cp $pointcloud_file $pointcloud_out_file
  cp $vox_file $vox_out_file
  cp $mask_dir $mask_out_dir
  cp $img_dir $img_out_dir

  # python dataset_shapenet/get_r2n2_cameras.py $metadata_file $camera_out_file
  # counter=0
  # for f in $img_dir/*.png; do
  #   outname="$(printf '%03d.jpg' $counter)"
  #   echo $f
  #   echo "$img_out_dir/$outname"
  #   convert "$f" -background white -alpha remove "$img_out_dir/$outname"
  #   counter=$(($counter+1))
  # done
}

export -f reorganize_pix3d

# Make output directories
mkdir -p $OUTPUT_PATH

# Run build
for c in ${CLASSES[@]}; do
  echo "Parsing class $c"
  BUILD_PATH_C=$BUILD_PATH/$c
  OUTPUT_PATH_C=$OUTPUT_PATH/$c

  MODEL_PATH_C=$INPUT_PATH/$c
  #CHOY2016_IMG_PATH_C="$CHOY2016_PATH/ShapeNetRendering/$c"
  #CHOY2016_VOX_PATH_C="$CHOY2016_PATH/ShapeNetVox32/$c"
  mkdir -p $OUTPUT_PATH_C

  ls $MODEL_PATH_C | parallel -P $NPROC --timeout $TIMEOUT \
    reorganize_pix3d  $BUILD_PATH_C $OUTPUT_PATH_C {}

  echo "Creating split"
  python create_split.py $OUTPUT_PATH_C --r_val 0.1 --r_test 0.2
done

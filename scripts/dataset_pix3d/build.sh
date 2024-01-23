source dataset_pix3d/config.sh
# Make output directories
mkdir -p $BUILD_PATH

# Run build
for c in ${CLASSES[@]}; do
  echo "Processing class $c"
  input_path_c=$INPUT_PATH/$c
  build_path_c=$BUILD_PATH/$c

  mkdir -p $build_path_c/0_in \
           $build_path_c/1_scaled \
           $build_path_c/1_transform \
           $build_path_c/2_depth \
           $build_path_c/2_watertight \
           $build_path_c/4_points \
           $build_path_c/4_pointcloud \
           $build_path_c/4_watertight_scaled \
           $build_path_c/5_voxels \
           $build_path_c/6_images \
           $build_path_c/6_masks \

  echo "Converting meshes to OFF"
  lsfilter $input_path_c $build_path_c/0_in .off | parallel -P $NPROC --timeout $TIMEOUT \
     meshlabserver -i $input_path_c/{}/model.obj -o $build_path_c/0_in/{}.off;
  
  echo "Scaling meshes"
  python $MESHFUSION_PATH/1_scale.py \
    --n_proc $NPROC \
    --in_dir $build_path_c/0_in \
    --out_dir $build_path_c/1_scaled \
    --t_dir $build_path_c/1_transform
  
  echo "Create depths maps"
  python $MESHFUSION_PATH/2_fusion.py \
    --mode=render --n_proc $NPROC \
    --in_dir $build_path_c/1_scaled \
    --out_dir $build_path_c/2_depth
  
  echo "Produce watertight meshes"
  python $MESHFUSION_PATH/2_fusion.py \
    --mode=fuse --n_proc $NPROC \
    --in_dir $build_path_c/2_depth \
    --out_dir $build_path_c/2_watertight \
    --t_dir $build_path_c/1_transform

  echo "Process watertight meshes"
  python sample_mesh.py $build_path_c/2_watertight \
      --n_proc $NPROC --resize \
      --bbox_in_folder $build_path_c/0_in \
      --pointcloud_folder $build_path_c/4_pointcloud \
      --points_folder $build_path_c/4_points \
      --mesh_folder $build_path_c/4_watertight_scaled \
      --packbits --float16
  
  echo "Process voxels"
  for off_file in $build_path_c/0_in/*.off; do
    # Get the base name of the .off file
    modelname=$(basename $off_file .off)

    # Define the destination path
    destination_path="$build_path_c/5_voxels/"

    # Convert the .off file to .binvox format, resolution: 32
    binvox -d 32 -cb $off_file

    # Move the .binvox file to the destination path
    mv "$build_path_c/0_in/${modelname}.binvox" $destination_path
  done

done

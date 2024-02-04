# Generate the ground truth height map from sampled data
- `gt_generator.py`: includes the gt height map generation and tactile images generation
- `visualizeHeightmaps.py`: Visualize the pixel heightmaps of all the reconstruction methods 
- `basics/sensorParams.py`: the parameters of tactile sensor
- `unit_test/dataReader.py`: generate the gt height maps. It reads in the sample points from `data/*/*_sampled.mat` and the dense point cloud from `processed_data/*/*.ply`; randomly creates the pose defined on the touch center; generates the heightmap and interpolate the missing pixels; save the generated data in `processed_data`
- `unit_test/read_gt_heightMap.py`: test the generated data. It visualizes the generated height maps and prints the pose in orientation and position in the original object's coordinates.

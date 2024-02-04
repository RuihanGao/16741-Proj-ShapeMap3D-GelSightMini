# Shape Mapping Tactile
Shape mapping data collection from simulated/real GelSight. 
## Data folders
- Train data: `textured_120sampled`
- Test data:  `textured_60sampled`

Link: https://drive.google.com/drive/folders/142DfhuJ9LH54jZ-OFN1NasgQ19Aj47i0?usp=sharing

## Sim Data Pipeline
- From root: `source env/bin/activate` and `startup.m`
- `cd scripts/batch`
- `generateDataFromObj.m`  [MATLAB] # 'scripts/matlab'
  -  converts `.obj` file to `.mat` data, and generates random sample points 
- `./sim_ycb.sh`        [python] # (didn't find)
  - forward simulates tactile images and ground truth heightmaps for sample points
- `./sim_ycb_depth.sh`        [python] # (didn't find)
  - renders depth camera and saves noisy point cloud
- `./batchReconstruct.sh`  [python]
  - Uses learned model (MLP/FCRN/Lookup table) to estimate heightmap from images 
- `./batch_npy2mat.sh`    [python]
  - Converts all npy data to `.mat` files for MATLAB scripts
- `visualizeYCBHeightMaps.m`  [MATLAB]
  - Visualize the data collected and save as `.png`

## Real Data Pipeline
- From root: `source env/bin/activate` and `startup.m`
- `cd scripts/batch`
- `./batchReconstructReal.sh`  [python]
  - Uses learned model (MLP/FCRN/Lookup table) to estimate heightmap from images 
- `./batch_real2mat.sh`    [python]
  - Converts the raw data to `.mat` files for MATLAB scripts 
- `visualizeYCBReal.m`  [MATLAB]
  - Visualize the data collected and save as `.png`

## Miscellaneous scripts 
- `visualizeHeightmaps.py` in `heightmap_gen` can compare the different heightmap generation methods 
  - Example: `python3.8 visualizeHeightmaps.py 011_banana textured_60sampled`

## Evaluation scripts 
- `heightMapEval.py` for evaluating accuracy of network heightmaps and contact masks for GelSight 
- `ReconstructionEval.py` for evaluating accuracy of reconstructed model wrt # touches 

## Folder structure 
```
.
├── calib                                   (*GelSight calib/learned parameters*)
│   ├── dataPack.npz                            (*GelSight bg file*)
│   ├── gelmap2.npy                             (*Static gelmap*)
│   ├── mlp_net.pt                              (*Learned model: MLP*)
│   └── polycalib.npz                           (*RGB calib data*)
├── gelsight_data                           (*Groundtruth and rendered gelsight images*)
│   ├── <Dataset 1>
│   ├── <Dataset 2>
│   │   ├── <Obj 1>
│   │   ├── <Obj 2>
│   │   │   ├── gt_contact_mask
│   │   │   │   └── 0 - N.npy
│   │   │   ├── gt_height_map
│   │   │   │   └── 0 - N.npy
│   │   │   ├── tactile_imgs
│   │   │   │   └── 0 - N.jpg
│   │   │   ├── pose.txt
│   │   │   ├── gelsight_data.mat
│   │   │   └── gelsight_gt.png
│   │   └── <Obj N>
│   └── <Dataset M>
├── generated_data                           (*Estimated heightmaps from learned model*)
│   ├── <Dataset 1>
│   ├── <Dataset 2>
│   │   ├── mlp
│   │   │   ├── <Obj 1>
│   │   │   │   ├── 0 - N.npy
│   │   │   │   ├── gen_data.mat
│   │   │   │   └── gelsight_gen.png
│   │   │   ├── <Obj N>
│   │   │   └── error.txt
│   │   ├── fcrn
│   │   │   ├── <Obj 1>
│   │   │   │   ├── 0 - N.npy
│   │   │   │   ├── gen_data.mat
│   │   │   │   └── gelsight_gen.png
│   │   │   ├── <Obj N>
│   │   │   └── error.txt
│   │   └── lookup
│   │       ├── <Obj 1>
│   │       │   ├── 0 - N.npy
│   │       │   ├── gen_data.mat
│   │       │   └── gelsight_gen.png
│   │       ├── <Obj N>
│   │       └── error.txt
│   └── <Dataset 2>
├── models                                    (*Model files and sampled points*)
│   ├── <Obj 1>
│   ├── <Obj 2>
│   │   └── ...
│   └── <Obj N>
├── scripts                                   (*Evaluation scripts*)
│   ├── batch                                 (*Calls other scripts for all object models*)
│   │   └── ...
│   ├── matlab                                (*MATLAB scripts*)
│   │   └── ...
│   ├── python                                (*Python scripts*)
│   │   ├── heightmap_gen                     (*Simulate GelSight readings on object models*)
│   │   │   └── ...
│   │   └── tactile2height                    (*Images -> local shape via learned model*)
│   │       └── ...
│   └── ur5_experiments                       (*Scripts to operate real-robot*)
├── README.md
├── .gitignore
└── startup.m
```

## Todo bug fixes: 
- [x] Verify orientation of gelSight is correct (Z pointing in normal direction)
- [ ] Heightmap --> cloud conversation and look at scale ambiguity. 
- [ ] Fix the artifacts (with contact mask?)
- [ ] Visualize the local clouds with the model and run shape mapping 


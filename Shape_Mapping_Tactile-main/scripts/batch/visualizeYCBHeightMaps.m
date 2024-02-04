clc; clear; close all;

objectList = ["002_master_chef_can","003_cracker_box","004_sugar_box", "005_tomato_soup_can",...
              "006_mustard_bottle","007_tuna_fish_can","008_pudding_box","009_gelatin_box",...
              "010_potted_meat_can","011_banana","012_strawberry","013_apple","014_lemon",...
              "017_orange","019_pitcher_base","021_bleach_cleanser","024_bowl","025_mug",...
              "029_plate","035_power_drill","036_wood_block","037_scissors","042_adjustable_wrench",...
              "043_phillips_screwdriver",	"048_hammer","055_baseball","056_tennis_ball",...
              "072-a_toy_airplane","072-b_toy_airplane","077_rubiks_cube"];

% objectList = ["002_master_chef_can","004_sugar_box", "005_tomato_soup_can",...
%               "010_potted_meat_can", "021_bleach_cleanser", "036_wood_block"];
          
shapeMappingRootPath = getenv('SHAPE_MAPPING_ROOT');
addpath([shapeMappingRootPath,'/scripts', '/matlab']) 

% datasetName = "textured_120sampled"; 
datasetName = "textured_60sampled"; 

for obj = objectList
    vizHeightMaps(obj, datasetName, false, 'depth_map'); % estimtaed heightmaps
    vizHeightMaps(obj, datasetName, false, 'fcrn'); % estimtaed heightmaps
    vizHeightMaps(obj, datasetName, false, 'mlp'); % estimtaed heightmaps
    vizHeightMaps(obj, datasetName, false, 'lookup'); % estimtaed heightmaps
    vizHeightMaps(obj, datasetName, false, 'gt'); % ground truth heightmaps
    fprintf('\n');
end

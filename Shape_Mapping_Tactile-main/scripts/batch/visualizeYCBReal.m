clc; clear; close all;

objectList = ["002_master_chef_can","004_sugar_box", "005_tomato_soup_can",...
              "010_potted_meat_can", "021_bleach_cleanser", "036_wood_block"];

objectList = ["004_sugar_box"];
dir = "/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/";

shapeMappingRootPath = getenv('SHAPE_MAPPING_ROOT');
addpath([shapeMappingRootPath,'/scripts', '/matlab']) 

for obj = objectList
    vizHeightmapsReal(dir, obj, true, 'fcrn'); % estimtaed heightmaps
%     vizHeightmapsReal(dir, obj, false, 'mlp'); % estimtaed heightmaps
%     vizHeightmapsReal(dir, obj, false, 'lookup'); % estimtaed heightmaps
    fprintf('\n');
end

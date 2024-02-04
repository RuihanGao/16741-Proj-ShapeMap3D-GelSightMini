clc; clear; close all;

objectList.model =        ["002_master_chef_can","003_cracker_box","004_sugar_box", "005_tomato_soup_can",...
                          "006_mustard_bottle","007_tuna_fish_can","008_pudding_box","009_gelatin_box",...
                          "010_potted_meat_can","011_banana","012_strawberry","013_apple","014_lemon",...
                          "017_orange","019_pitcher_base","021_bleach_cleanser","024_bowl","025_mug",...
                          "029_plate","035_power_drill","036_wood_block","037_scissors","042_adjustable_wrench",...
                          "043_phillips_screwdriver",	"048_hammer","055_baseball","056_tennis_ball",...
                          "072-a_toy_airplane","072-b_toy_airplane","077_rubiks_cube"];


% objectList.model =        ["002_master_chef_can","004_sugar_box", "005_tomato_soup_can",...
%               "010_potted_meat_can", "021_bleach_cleanser", "036_wood_block"];

shapeMappingRootPath = getenv('SHAPE_MAPPING_ROOT');
numSamples = 60;
model2mat = false; 

for i = 1:size(objectList.model, 2)
    dataPath = fullfile(shapeMappingRootPath, 'models', objectList.model(i), 'google_512k', 'textured.obj');
    fprintf('\n');
    disp(dataPath);
    if model2mat
        Obj2Mat(dataPath, false); % only needed if the model .mat files don't exist
    end
    sampleModelPoints(objectList.model(i), numSamples, true);
%   sampleModelPoints(objectList.model(i), numSamples, false); % debug
end


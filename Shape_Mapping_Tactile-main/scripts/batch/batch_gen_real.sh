#!/bin/bash

python3.8 -W ignore ../python/tactile2height/all_test.py real

# Declare an array of string with type
declare -a objModels=("002_master_chef_can" "004_sugar_box" "005_tomato_soup_can" "010_potted_meat_can" "021_bleach_cleanser" "036_wood_block")

# Iterate the string array using for loop
for val in ${objModels[@]}; do
   python3.8 ../python/heightmap_gen/real2mat.py $val /media/suddhu/Backup\ Plus/suddhu/rpl/datasets/tactile_mapping/
done

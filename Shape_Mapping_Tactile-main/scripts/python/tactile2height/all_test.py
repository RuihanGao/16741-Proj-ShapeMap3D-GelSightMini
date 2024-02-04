from fcrn import Simulation as fcrn_Simulation
from mlp import Simulation as mlp_Simulation
from lookup import Simulation as lookup_Simulation
from contact_mask import Simulation as contact_mask_Simulation
import os
from os import path as osp
import sys


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(0)

    # change path to cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    source = str(sys.argv[1])

    test_fcrn = True
    test_mlp = True
    test_lookup = True
    test_contact_mask = True

    print('test_fcrn: ', test_fcrn)
    print('test_mlp: ', test_mlp)
    print('test_lookup: ', test_lookup)
    print('test_contact_mask: ', test_contact_mask)

    if source == "sim":
        from config import lookup_table_config,mlp_net_config,fcrn_net_config,contact_mask_config
        obj = None
    elif source == "real":
        from real_config import lookup_table_config,mlp_net_config,fcrn_net_config,contact_mask_config
        obj = ["002_master_chef_can","004_sugar_box", "005_tomato_soup_can", "010_potted_meat_can", "021_bleach_cleanser", "036_wood_block"]

    print('source: ', source)
    if test_fcrn:
        print('\n\nSimulating FCRN')
        sim = fcrn_Simulation(**fcrn_net_config)
        sim.simulate(obj)

    if test_mlp:
        print('\n\nSimulating MLP')
        sim = mlp_Simulation(**mlp_net_config)
        sim.simulate(obj)

    if test_lookup:
        print('\n\nSimulating Lookup')
        sim = lookup_Simulation(**lookup_table_config)
        sim.simulate(obj)

    if test_contact_mask:
        print('\n\nSimulating contact mask')
        sim = contact_mask_Simulation(**contact_mask_config)
        sim.simulate(obj)

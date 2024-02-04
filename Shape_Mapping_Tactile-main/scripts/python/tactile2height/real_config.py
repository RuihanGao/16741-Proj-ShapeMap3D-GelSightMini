

lookup_table_config = {
    'path2model' : '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/calib_real/reconstruct_calib.npz',
    'path2data': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/ycbSight/',
    'path2background': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/calib_real/real_bg.npy',
    'path2gel_model': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/calib_real/gelmap2.npy',
    'indent_depth': 1.0,
    'num_data': 40,
    'path2save': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/generated_data/lookup/',

}

mlp_net_config = {
    'path2model' : '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/calib_real/MLP_model.pt',
    'path2data': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/ycbSight/',
    'path2background': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/calib_real/real_bg.npy',
    'path2gel_model': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/calib_real/gelmap2.npy',
    'indent_depth': 1.0,
    'num_data': 40,
    'path2save': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/generated_data/mlp/',
}

fcrn_net_config = {
    'path2model' : '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/calib_real/0826_checkpoint.pth.tar',
    'path2data': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/ycbSight/',
    'num_data': 40,
    'path2save': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/generated_data/fcrn/'
}


contact_mask_config = {
    'path2data': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/ycbSight/',
    'path2sim_height_map': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/generated_data/fcrn/',
    'path2background': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/calib_real/real_bg.npy',
    'path2gel_model': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/calib_real/real_gelmap2.npy',
    'num_data': 40,
    'path2save': '/media/suddhu/Backup Plus/suddhu/rpl/datasets/tactile_mapping/generated_data/contact_masks/',
    'data_source': 'real'
}

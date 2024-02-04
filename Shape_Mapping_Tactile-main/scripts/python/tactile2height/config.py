

lookup_table_config = {
    'path2model' : '../../../calib/reconstruct_calib.npz',
    'path2data': '../../../gelsight_data/textured_60sampled/',
    'path2background': '../../../calib/bg.jpg',
    'path2gel_model': '../../../calib/gelmap2.npy',
    'indent_depth': 1.0,
    'num_data': 60,

    'path2save': '../../../generated_data/textured_60sampled/lookup/',

}

mlp_net_config = {
    'path2model' : '../../../calib/mlp_net.pt',
    'path2data': '../../../gelsight_data/textured_60sampled/',
    'path2background': '../../../calib/bg.jpg',
    'path2gel_model': '../../../calib/gelmap2.npy',
    'indent_depth': 1.0,
    'num_data': 60,
    'path2save': '../../../generated_data/textured_60sampled/mlp/',
}

fcrn_net_config = {
    'path2model' : '../../../calib/checkpoint.pth.tar',
    'path2data': '../../../gelsight_data/textured_60sampled/',
    'num_data': 60,
    'path2save': '../../../generated_data/textured_60sampled/fcrn/'
}

real_fcrn_net_config = {
    'path2model' : '../../../calib/0902_checkpoint.pth.tar',
    'path2data': '../../../gelsight_data/real/',
    'num_data': 80,

    'path2save': '../../../generated_data/real/fcrn/'
}


contact_mask_fcrn_config = {
    'path2model' : '../../../calib/0902_checkpoint.pth.tar',
    'path2data': '../../../gelsight_data/textured_50sampled/',
    'num_data': 50,

    'path2save': '../../../generated_data/textured_50sampled/contact_mask_fcrn/'

}

real_contact_mask_fcrn_config = {
    'path2model' : '../../../calib/0902_contact_mask.pth.tar',
    'path2data': '../../../gelsight_data/real/',
    'num_data': 80,

    'path2save': '../../../generated_data/real/contact_mask_fcrn/'

}



contact_mask_config = {
    'path2data': '../../../gelsight_data/textured_60sampled/',
    'path2sim_height_map': '../../../generated_data/textured_60sampled/fcrn/',
    'path2background': '../../../calib/bg.jpg',
    'path2gel_model': '../../../calib/gelmap2.npy',
    'num_data': 60,
    'path2save': '../../../generated_data/textured_60sampled/contact_masks/',
    'data_source': 'sim'
}

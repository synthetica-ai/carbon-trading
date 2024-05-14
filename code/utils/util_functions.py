from os import getcwd
from random import randint

import tensorflow as tf

def create_meta_parameters():
    """Creates all needed parameters to instantiate estimator."""

    result = dict()
    result['initial_directory'] = getcwd()
    result['logs_directory'] = getcwd() + '/logs'
    result['saved_model_directory'] = getcwd()+'/saved_models'
    result['data_directory'] = getcwd()+'/data'
    result['path_to_checkpoint'] = None
    result['epoch'] = 0
    result['warmup_exp_beta'] = 0.8

    result['batch_verbose']  = 1000
    result['seeds'] = {'train': None,
                       'validation': randint(1000,9999),
                       'baseline': None}

    result['dataset_sizes'] = {'train': 512,
                               'validation': 128,
                               'baseline': 128}
    result['batch_sizes'] = {'train': 64,
                               'validation': 36,
                               'baseline': 36}                           
                                   
    result['number_of_wp_epochs']  = 1
    result['from_checkpoint']  = False
    result['grad_norm_clipping']  = 1.0
    result['tanh_clipping'] = 10
    result['graph_size'] = 10


    # result['max_capacity'] = max(result['truck_capacities'])    
    # with open(getcwd() + '/data/locations_simos_220.pkl', 'rb') as f:
    #    result['locations_list'] = pickle.load(f)
    
    # with open(getcwd() + '/data/new_time_windows.pkl', 'rb') as f:
    #    new_time_windows = pickle.load(f)


    # result['locations_list'][2] = tf.expand_dims(new_time_windows[0],0)
    # result['locations_list'][3] = tf.expand_dims(new_time_windows[1],0)

    # all_locs = tf.concat([result['locations_list'][0],tf.squeeze(result['locations_list'][1])], axis = 0)
    # coord_mins = tf.reduce_min(all_locs,axis = 0)
    # coord_maxes = tf.reduce_max(all_locs,axis = 0)
    # coord_ranges = tf.subtract(coord_maxes,coord_mins)
    # all_locs = tf.divide(tf.subtract(all_locs,coord_mins),coord_ranges)
    # result['depot_location'] = tf.slice(all_locs, [0,0],[1,2])
    # result['rest_locations'] = tf.expand_dims(tf.slice(all_locs, [1,0],[220,2]), axis = 0)
    # result['start_windows'] = [[x / ((12.-4.)*3600.) for l in result['locations_list'][2] for x in l]]
    # result['end_windows'] = [[x / ((12.-4.)*3600.) for l in result['locations_list'][3] for x in l]]


    # result['graph_size']  = 220
    # result['filename']  = 'VRP_{}_{}'.format(result['graph_size'], strftime("%Y-%m-%d", gmtime()))
    # result['node_cap'] = list(result['locations_list'][-1].numpy()[0])


    # with open(getcwd() + '/data/distance_matrix_simos_220.pkl', 'rb') as f:
    #         dists = pickle.load(f)
    # unpacked_dist = list(chain.from_iterable(dists))
    # min_dist, range_dist = float(min(unpacked_dist)), float(max(unpacked_dist) - min(unpacked_dist))
    # result['distance_min'] = min_dist
    # result['distance_range'] = range_dist
    # result['distance_tensor'] = [[ float((x-min_dist) / range_dist) for x in l] for l in dists]        
    # result['distance_tensor_depot'] = [result['distance_tensor'][0]]#[dists[0]]
    # result['distance_tensor_clients'] = [result['distance_tensor'][1:]]#[dists[1:]]

    # min_durs, range_durs = float(min(unpacked_durs)), float(max(unpacked_durs) - min(unpacked_durs))
    # result['duration_min'] = min_durs
    # result['duration_range'] = range_durs        
    # result['duration_tensor'] = [[number / (8. * 3600.) for number in group] for group in durs]



    # LR things
    result['warmup_lr'] = 1e-3
    result['initial_lr'] = 1e-1
    result['warmup_epochs'] = 5
    result['warmup_step'] = (result['initial_lr'] - result['warmup_lr']) / result['warmup_epochs']
    result['reduce_lr_delta'] = 1e-12
    result['reduce_lr_factor'] = .97
    result['patience'] = 15
    result['min_lr'] = 1e-12
    result['max_lr'] = 1e-3
    result['cycle'] = 16
    result['step_size'] = 8
    result['lr_scheduler'] = 'cyclical'
    # result['cyclical_lr_values'] = linspace(result['min_lr'],result['max_lr'],result['step_size']).tolist()
    # result['cyclical_lr_values'] = result['cyclical_lr_values'] + result['cyclical_lr_values'][::-1][1:-1]

    result['embedding_dim']  = 512#256#128
    result['n_encode_layers'] = 3
    result['n_heads'] = 8#4
    result['feed_forward_hidden'] = 512#128
    result['learning_rate']  = 1e-4#1e-1
    result['learning_rate'] = result['min_lr']

    return result


def create_meta_parameters(operation):
    """Creates all needed parameters to instantiate estimator."""

    result = dict()
    result['initial_directory'] = getcwd()
    result['logs_directory'] = getcwd() + '/logs'
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    result['logs_current_directory'] = f"{result['logs_directory']}/{current_time}"
    result['train_log_dir'] = f"{result['logs_directory']}/{current_time}/train"
    result['val_log_dir'] = f"{result['logs_directory']}/{current_time}/val" 
    result['test_log_dir'] = f"{result['logs_directory']}/{current_time}/test"  
    result['hparam_log_dir'] = f"{result['logs_directory']}/{current_time}/hparam"  
    result['saved_model_directory'] = getcwd()+'/saved_models'
    # saved_models = list(filter(lambda x:  path.isdir(f"{getcwd()}/saved_models/{x}"), listdir(f"{getcwd()}/saved_models"))) 
    # version_number = 1 if len(saved_models) == 0 else max(list(map(int,saved_models)))+1
    # result['saved_model_filepath'] = f"{result['saved_model_directory']}/{version_number}"
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

    result['record_filenames'] = {'train': 'train.tfrecords',
                                  'validation': 'validation.tfrecords',
                                  'baseline': 'baseline.tfrecords'}                                           
    result['number_of_wp_epochs']  = 1
    result['from_checkpoint']  = False
    result['grad_norm_clipping']  = 1.0
    result['tanh_clipping'] = 10
    result['capacities'] = {
                            50: 15,
                            220: 15.
                        }
    # result['start_time_windows'] = ['5:00', '6:45', '6:15', '5:20', '8:00', '7:15', '6:10', '6:50',
    #                                  '5:40', '6:20', '6:00', '6:40', '8:30', '5:45', '7:30', '6:30', 
    #                                  '5:50', '4:00', '7:00', '5:30']                    
    # result['start_time_windows'] = list(map(lambda x: float(x.replace(":",".")) - 4.0,result['start_time_windows']))
    # result['start_time_windows'].sort(reverse = False)
    # result['start_time_windows_probs'] = [[1/len(result['start_time_windows'])]*len(result['start_time_windows'])]
    # result['delay_time_windows'] = arange(0,6,step=0.1).tolist()
    # result['delay_time_windows_probs'] = [[1/len(result['delay_time_windows'])]*len(result['delay_time_windows'])] 
    

    result['truck_capacities'] = [15, 15, 15, 15, 15, 15, 15, 12, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                10, 10,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
                                    8,  8,  8,  8,  8,  8]
    result['truck_capacities'] = [15, 15, 15, 15, 15, 15, 15, 12, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                    8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
                                    8,  8,  8]

    result['max_capacity'] = max(result['truck_capacities'])    
    with open(getcwd() + '/data/locations_simos_220.pkl', 'rb') as f:
       result['locations_list'] = pickle.load(f)
    
    with open(getcwd() + '/data/new_time_windows.pkl', 'rb') as f:
       new_time_windows = pickle.load(f)


    result['locations_list'][2] = tf.expand_dims(new_time_windows[0],0)
    result['locations_list'][3] = tf.expand_dims(new_time_windows[1],0)

    all_locs = tf.concat([result['locations_list'][0],tf.squeeze(result['locations_list'][1])], axis = 0)
    coord_mins = tf.reduce_min(all_locs,axis = 0)
    coord_maxes = tf.reduce_max(all_locs,axis = 0)
    coord_ranges = tf.subtract(coord_maxes,coord_mins)
    all_locs = tf.divide(tf.subtract(all_locs,coord_mins),coord_ranges)
    result['depot_location'] = tf.slice(all_locs, [0,0],[1,2])
    result['rest_locations'] = tf.expand_dims(tf.slice(all_locs, [1,0],[220,2]), axis = 0)
    result['start_windows'] = [[x / ((12.-4.)*3600.) for l in result['locations_list'][2] for x in l]]
    result['end_windows'] = [[x / ((12.-4.)*3600.) for l in result['locations_list'][3] for x in l]]
    result['postal_codes'] = [14568, 17455, 15123, 14231, 13122, 11523, 18233, 15344, 10447, 18535, 
                            16451, 14565, 10440, 19004, 10671, 11472, 10564, 11256, 12242, 15233, 15562, 12134, 11633, 16673, 
                            17562, 15125, 17564, 16674, 12351, 17234, 17674, 11476, 15341, 19013, 11632, 19016, 10680, 17237, 
                            16777, 11525, 17672, 10443, 11144, 13451, 16231, 14452, 19200, 10443, 18451, 11254, 17343, 19005, 17234, 
                            18541, 16675, 19002, 13122, 15342, 12136, 13123, 10433, 18534, 17124, 16672, 15234, 12135, 15238, 18546, 
                            19003, 19009, 13451, 19003, 19400, 15122, 17455, 12133, 15344, 14122, 19100, 13674, 19300, 19010, 13343, 
                            18233, 15351, 15344, 10443, 11631, 18536, 11745, 18648, 11633, 11361, 17671, 11147, 15451, 18535, 11524, 
                            12135, 18541, 15561, 11363, 18539, 17562, 11253, 11473, 10445, 14569, 15234, 13122, 15127, 14123, 18758, 
                            13231, 12241, 14343, 11853, 16345, 16674, 17563, 14235, 17675, 17455, 15238, 11146, 18544, 12135, 11142, 
                            11526, 16121, 16672, 13231, 11632, 18538, 13678, 16452, 18532, 18450, 19009, 13451, 12462, 17778, 12131, 
                            10444, 13123, 15344, 17343, 17343, 17342, 17343, 17124, 11745, 16671, 14564, 14564, 14671, 19300, 11146, 
                            19200, 15773, 19014, 11743, 18452, 11635, 12133, 13231, 11525, 10443, 16562, 17563, 16777, 17676, 17675, 
                            11741, 17455, 17123, 18345, 18547, 16674, 18542, 17455, 17342, 10440, 12351, 11523, 11364, 16672, 15771, 
                            11527, 17121, 19400, 14235, 13231, 16231, 17121, 19100, 17676, 17124, 15341, 18863, 10444, 17778, 15772, 
                            11634, 11364, 10446, 18546, 18541, 18122, 15124, 19500, 17456, 18120, 12461, 12242, 11852, 13121, 13671, 
                            16341, 17342, 12131]
    result['postal_codes'] = list(map(lambda x:( x - min(result['postal_codes'])) / (max(result['postal_codes'])-min(result['postal_codes'])),result['postal_codes']))
    result['postal_codes'] = [result['postal_codes'][1:]] #Skipping Simos postal
    result['graph_mode'] = 'distance_matrix' #real_lat_long'
    result['time_windows_mode'] = True
       
 
    result['graph_size']  = 220
    result['filename']  = 'VRP_{}_{}'.format(result['graph_size'], strftime("%Y-%m-%d", gmtime()))
    result['node_cap'] = list(result['locations_list'][-1].numpy()[0])


    with open(getcwd() + '/data/distance_matrix_simos_220.pkl', 'rb') as f:
            dists = pickle.load(f)
    unpacked_dist = list(chain.from_iterable(dists))
    min_dist, range_dist = float(min(unpacked_dist)), float(max(unpacked_dist) - min(unpacked_dist))
    result['distance_min'] = min_dist
    result['distance_range'] = range_dist
    result['distance_tensor'] = [[ float((x-min_dist) / range_dist) for x in l] for l in dists]        
    result['distance_tensor_depot'] = [result['distance_tensor'][0]]#[dists[0]]
    result['distance_tensor_clients'] = [result['distance_tensor'][1:]]#[dists[1:]]

    # print(colored(f"Comparing depot : {result['distance_tensor_depot']} \n\n ",'red'))
    # print(colored(f"Comparing nodes : {result['rest_locations']} \n\n ",'red')) #{}result['distance_tensor_clients']

    with open(getcwd() + '/data/duration_matrix_simos_220.pkl', 'rb') as f:
            durs = pickle.load(f)
    unpacked_durs = list(chain.from_iterable(durs))
    min_durs, range_durs = float(min(unpacked_durs)), float(max(unpacked_durs) - min(unpacked_durs))
    result['duration_min'] = min_durs
    result['duration_range'] = range_durs        
    result['duration_tensor'] = [[number / (8. * 3600.) for number in group] for group in durs]


    with open(getcwd() + '/data/0_demand_locations_simos_220.pkl', 'rb') as f:
        d0_demand = pickle.load(f)
    demand_wo_nan = tf.where(tf.math.is_nan(d0_demand[2]),tf.zeros_like(d0_demand[2]),d0_demand[2])
    tour0_start_windows = [[x / ((12.-4.)*3600.) for l in d0_demand[3] for x in l]]
    tour0_end_windows = [[x / ((12.-4.)*3600.) for l in d0_demand[4] for x in l]]

    # print(f"My 0 {d0_demand[0]}")
    # print(f"My others ")
    # print(f"Depo {tf.shape(result['distance_tensor_depot'])}")
    # print(f"Graphs {tf.shape(result['distance_tensor_clients'])}")
    # print(f"Postal {tf.shape(result['postal_codes'])}")
    # print(f"My 1 {d0_demand[1]}")
    # print(f"My 2 {d0_demand[2]}")
    # # print(f"My 3 {demand_wo_nan}")
    # # print(f"My 4 {d0_demand[4]}")
    result['tour_0_demand'] = [tf.cast(result['distance_tensor_depot'],tf.float32),
                                tf.cast(result['distance_tensor_clients'],tf.float32),
                                tf.cast(result['postal_codes'],dtype=tf.float32),
                                tf.divide(demand_wo_nan,15.),tour0_start_windows,tour0_end_windows]

    with open(getcwd() + '/data/dict_with_demands.pkl', 'rb') as f:
        result['dic'] = pickle.load(f)
    result['customers_list'] = d0_demand[6][0]



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
    result['lr_scheduler'] = 'cyclical'#'reduce_on_plateau'
    result['cyclical_lr_values'] = linspace(result['min_lr'],result['max_lr'],result['step_size']).tolist()
    result['cyclical_lr_values'] = result['cyclical_lr_values'] + result['cyclical_lr_values'][::-1][1:-1]

    # Hyperparams
    result['EMBEDDING_DIM_CHOICES'] = [64,128,256,512,1024]
    result['N_ENCODE_LAYERS_CHOICES'] = [1,2,3,4,5,6]
    result['N_HEADS_CHOICES'] = [2,4,8]
    result['FEED_FORWARD_HIDDEN_CHOICES'] = [64,128,256,512,1024]
    result['LEARNING_RATE_CHOICES']  = [1e-5,1e-4,1e-3,1e-2,1e-1] 

    result['embedding_dim']  = 512#256#128
    result['n_encode_layers'] = 3
    result['n_heads'] = 8#4
    result['feed_forward_hidden'] = 512#128
    result['learning_rate']  = 1e-4#1e-1
    result['learning_rate'] = result['min_lr']

    return result


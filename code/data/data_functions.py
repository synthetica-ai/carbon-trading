import tensorflow as tf

def train_input_fn(params): #
    """Generate temp dataset in memory
    """

    seed = params['seeds']['train']
    size = params['dataset_sizes']['train']
    batch_size = params['batch_sizes']['train']
    graph_mode = params['graph_mode']
    time_windows_mode = params['time_windows_mode']
    tf.random.set_seed(seed)

    depo = tf.random.uniform(minval=0, maxval=1, shape=(size, 2))
    graphs = tf.random.uniform(minval=0, maxval=1, shape=(size, params['graph_size'], 2))

                                         
    #demands
    # probs = tf.repeat(tf.math.log([[1 / 221,2 / 221,1 / 221,2 / 221,4 / 221,6 / 221,15 / 221,27 / 221,87 / 221,76 / 221]]),repeats = size ,axis = 0)
    # demands_probs = tf.random.categorical(probs, params['graph_size'],seed=seed)
    # demands_0 = tf.gather(tf.constant([12,11,8,7,6,5,4,3,2,1]),demands_probs)
    # demand = tf.cast(demands_0, tf.float32) / tf.cast(params['capacities'][params['graph_size']], tf.float32)




    # demand = tf.concat([demand,demand2],axis = 0)
    # start_window_indices = tf.random.categorical(tf.math.log(params['start_time_windows_probs']), size*params['graph_size'])
    # delay_window_indices = tf.random.categorical(tf.math.log(params['delay_time_windows_probs']), size*params['graph_size'])
    # start_window = tf.squeeze(tf.gather (tf.constant(params['start_time_windows']), start_window_indices))
    # delay_window = tf.squeeze(tf.gather (tf.constant(params['delay_time_windows']), delay_window_indices))
    # end_window = tf.add(start_window, delay_window)
    # start_window = tf.reshape(start_window, shape =(size,params['graph_size'],))
    # end_window = tf.reshape(end_window, shape =(size,params['graph_size'],))
    # start_window = tf.reshape(start_window, shape =(size,params['graph_size'],))
    # end_window = tf.reshape(end_window, shape =(size,params['graph_size'],))



    dataset = tf.data.Dataset.from_tensor_slices((list(depo), list(graphs)))
    dataset = dataset.batch(batch_size, drop_remainder = True).prefetch(tf.data.experimental.AUTOTUNE)  
    return dataset


    
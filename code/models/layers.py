from __future__ import print_function
import tensorflow as tf
import numpy as np


class ContractEncoder(tf.keras.layers.Layer):  # TODO bug is here
    def __init__(self, emb_dim):
        #                  num_heads,
        #                  tanh_clipping=10,
        #                  decode_type=None):

        super().__init__()
        #         self.contract_input = layers.Input(shape=36, name="contracts")
        self.emb_dim = emb_dim
        self.dense1 = tf.keras.layers.Dense(256, activation="relu", name="dense1")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu", name="dense2")
        self.embedding = tf.keras.layers.Dense(self.emb_dim, name="emb")

    def call(self, x):
        #         contract_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="contracts")
        #         fleet_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="fleet")

        # Encoding part
        #         initializer = tf.keras.initializers.GlorotUniform()
        x = self.dense1(x)
        x = self.dense2(x)
        embedding = self.embedding(x)
        return embedding


class ShipDecoder(tf.keras.layers.Layer):  # TODO bug is here
    """
    A class that concatenates the contracts embeddings with the fleet tensor
    to get the context
    """

    def __init__(self, output_size):
        #                  num_heads,
        #                  tanh_clipping=10,
        #                  decode_type=None):
        super().__init__()
        #         self.contract_input = layers.Input(shape=36, name="contracts")
        self.output_size = output_size
        self.dense_in = tf.keras.layers.Dense(256, activation="relu", name="relu_layer")
        self.dense_out = tf.keras.layers.Dense(
            self.output_size, activation="linear", name="linear_layer"
        )

    def call(self, emb, fleet_tensor):
        #         contract_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="contracts")
        #         fleet_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="fleet")
        #         for i in range(env.fleet):
        x = tf.concat([emb, fleet_tensor], 1)
        x = self.dense_in(x)
        logits = self.dense_out(x)

        return logits


class MultiHeadAttention(tf.keras.layers.Layer):
    """Attention Layer - multi-head scaled dot product attention (for encoder and decoder)
    Args:
        num_heads: number of attention heads which will be computed in parallel
        d_model: embedding size of output features
    Call arguments:
        q: query, shape (..., seq_len_q, depth_q)
        k: key, shape == (..., seq_len_k, depth_k)
        v: value, shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k) or None.
        Since we use scaled-product attention, we assume seq_len_k = seq_len_v
    Returns:
          attention outputs of shape (batch_size, seq_len_q, d_model)
    """

    def __init__(self, n_heads, d_model, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_depth = self.d_model // self.n_heads

        if self.d_model % self.n_heads != 0:
            raise ValueError("number of heads must divide d_model")

        # define weight matrices
        self.wq = tf.keras.layers.Dense(self.d_model, use_bias=False)  # (d_q, d_model)
        self.wk = tf.keras.layers.Dense(self.d_model, use_bias=False)  # (d_k, d_model)
        self.wv = tf.keras.layers.Dense(self.d_model, use_bias=False)  # (d_v, d_model)

        self.w_out = tf.keras.layers.Dense(
            self.d_model, use_bias=False
        )  # (d_model, d_model)

    def split_heads(self, tensor, batch_size):
        """Function for computing attention on several heads simultaneously
        Splits last dimension of a tensor into (num_heads, head_depth).
        Then we transpose it as (batch_size, num_heads, ..., head_depth) so that we can use broadcast
        """
        tensor = tf.reshape(tensor, (batch_size, -1, self.n_heads, self.head_depth))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    # treats first parameter q as input, and  k, v as parameters, so input_shape=q.shape
    def call(self, q, k, v, mask=None):
        # shape of q: (batch_size, seq_len_q, d_q)
        batch_size = tf.shape(q)[0]

        # compute Q = q * w_q, ...
        Q = self.wq(
            q
        )  # (batch_size, seq_len_q, d_q) x (d_q, d_model) --> (batch_size, seq_len_q, d_model)
        K = self.wk(k)  # ... --> (batch_size, seq_len_k, d_model)
        V = self.wv(v)  # ... --> (batch_size, seq_len_v, d_model)

        # split heads: d_model = num_heads * head_depth + reshape
        Q = self.split_heads(
            Q, batch_size
        )  # (batch_size, num_heads, seq_len_q, head_depth)
        K = self.split_heads(
            K, batch_size
        )  # (batch_size, num_heads, seq_len_k, head_depth)
        V = self.split_heads(
            V, batch_size
        )  # (batch_size, num_heads, seq_len_v, head_depth)

        # similarity between context vector Q and key K // self-similarity in case of self-attention
        compatibility = tf.matmul(
            Q, K, transpose_b=True
        )  # (batch_size, num_heads, seq_len_q, seq_len_k)
        # seq_len_q = n_nodes for encoder self-attention
        # seq_len_q = 1 for decoder context-vector attention
        # seq_len_k = n_nodes for both encoder & decoder
        # rescaling
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        compatibility = compatibility / tf.math.sqrt(dk)

        if mask is not None:
            # we need to reshape mask:
            # (batch_size, seq_len_q, seq_len_k) --> (batch_size, 1, seq_len_q, seq_len_k)
            # so that we will be able to do a broadcast:
            # (batch_size, num_heads, seq_len_q, seq_len_k) + (batch_size, 1, seq_len_q, seq_len_k)
            mask = mask[:, tf.newaxis, :, :]

            # we use tf.where since 0*-np.inf returns nan, but not -np.inf
            compatibility = tf.where(
                mask, tf.ones_like(compatibility) * (-np.inf), compatibility
            )

        compatibility = tf.nn.softmax(
            compatibility, axis=-1
        )  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # seq_len_k = seq_len_v
        attention = tf.matmul(
            compatibility, V
        )  # (batch_size, num_heads, seq_len_q, head_depth)

        # transpose back to (batch_size, seq_len_q, num_heads, head_depth)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        # concatenate heads (last 2 dimensions)
        attention = tf.reshape(
            attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        # project output to the same dimension
        # this is equiv. to sum in the article (project heads with W_o and sum), beacuse of block-matrix multiplication
        # e.g. https://math.stackexchange.com/questions/2961550/matrix-block-multiplication-definition-properties-and-applications

        output = self.w_out(attention)  # (batch_size, seq_len_q, d_model)

        return output


class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    """Feed-Forward Sublayer: fully-connected Feed-Forward network,
    built based on MHA vectors from MultiHeadAttention layer with skip-connections
        Args:
            num_heads: number of attention heads in MHA layers.
            input_dim: embedding size that will be used as d_model in MHA layers.
            feed_forward_hidden: number of neuron units in each FF layer.
        Call arguments:
            x: batch of shape (batch_size, n_nodes, node_embedding_size).
            mask: mask for MHA layer
        Returns:
               outputs of shape (batch_size, n_nodes, input_dim)
    """

    def __init__(self, input_dim, num_heads, feed_forward_hidden=512, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(n_heads=num_heads, d_model=input_dim, name="MHA")
        self.bn1 = tf.keras.layers.BatchNormalization(name="bn1", trainable=True)
        self.bn2 = tf.keras.layers.BatchNormalization(name="bn2", trainable=True)
        self.ff1 = tf.keras.layers.Dense(feed_forward_hidden, name="ff1")
        self.ff2 = tf.keras.layers.Dense(input_dim, name="ff2")

    def call(self, x, mask=None):
        mha_out = self.mha(x, x, x, mask)
        sc1_out = tf.keras.layers.Add()([x, mha_out])
        bn1_out = self.bn1(sc1_out, training=True)

        ff1_out = self.ff1(bn1_out)
        relu1_out = tf.keras.activations.relu(ff1_out)
        ff2_out = self.ff2(relu1_out)
        sc2_out = tf.keras.layers.Add()([bn1_out, ff2_out])
        bn2_out = self.bn2(sc2_out, training=True)

        return bn2_out


class GraphAttentionEncoder(tf.keras.layers.Layer):
    """Graph Encoder, which uses MultiHeadAttentionLayer sublayer.
    Args:
        input_dim: embedding size that will be used as d_model in MHA layers.
        num_heads: number of attention heads in MHA layers.
        num_layers: number of attention layers that will be used in encoder.
        feed_forward_hidden: number of neuron units in each FF layer.
    Call arguments:
        x: tuples of 3 tensors:  (batch_size, 2), (batch_size, n_nodes-1, 2), (batch_size, n_nodes-1)
        First tensor contains coordinates for depot, second one is for coordinates of other nodes,
        Last tensor is for normalized demands for nodes except depot
        mask: mask for MHA layer
    Returns:
           Embedding for all nodes + mean embedding for graph.
           Tuples ((batch_size, n_nodes, input_dim), (batch_size, input_dim))
    """

    def __init__(self, input_dim, num_heads, num_layers, feed_forward_hidden=512):
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feed_forward_hidden = feed_forward_hidden

        # initial embeddings (batch_size, n_nodes-1, 2) --> (batch-size, input_dim), separate for depot and other nodes
        self.init_embed_depot = tf.keras.layers.Dense(
            self.input_dim, name="init_embed_depot"
        )  # nn.Linear(2, embedding_dim)
        self.init_embed = tf.keras.layers.Dense(self.input_dim, name="init_embed")

        self.mha_layers = [
            MultiHeadAttentionLayer(
                self.input_dim, self.num_heads, self.feed_forward_hidden
            )
            for _ in range(self.num_layers)
        ]

    def call(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        x = tf.concat(
            (
                self.init_embed_depot(x[0])[
                    :, None, :
                ],  # (batch_size, 2) --> (batch_size, 1, 2)
                self.init_embed(
                    tf.concat(
                        (x[1], x[2][:, :, None], x[3][:, :, None], x[4][:, :, None]),
                        axis=-1,
                    )
                ),  # (batch_size, n_nodes-1, 2) + (batch_size, n_nodes-1)
            ),
            axis=1,
        )  # (batch_size, n_nodes, input_dim)

        # stack attention layers
        for i in range(self.num_layers):
            x = self.mha_layers[i](x)

        output = (x, tf.reduce_mean(x, axis=1))
        return output  # (embeds of nodes, avg graph embed)=((batch_size, n_nodes, input), (batch_size, input_dim))


class GraphAttentionDecoder(tf.keras.layers.Layer):  # TODO bug is here
    def __init__(self, output_dim, num_heads, tanh_clipping=10, decode_type=None):

        super().__init__()

        self.output_dim = output_dim
        self.num_heads = num_heads

        self.head_depth = self.output_dim // self.num_heads
        self.dk_mha_decoder = tf.cast(
            self.head_depth, tf.float32
        )  # for decoding in mha_decoder
        self.dk_get_loc_p = tf.cast(
            self.output_dim, tf.float32
        )  # for decoding in mha_decoder

        if self.output_dim % self.num_heads != 0:
            raise ValueError("number of heads must divide d_model=output_dim")

        self.tanh_clipping = tanh_clipping
        self.decode_type = decode_type

        # we split projection matrix Wq into 2 matrices: Wq*[h_c, h_N, D] = Wq_context*h_c + Wq_step_context[h_N, D]
        self.wq_context = tf.keras.layers.Dense(
            self.output_dim, use_bias=False, name="wq_context"
        )  # (d_q_context, output_dim)
        self.wq_step_context = tf.keras.layers.Dense(
            self.output_dim, use_bias=False, name="wq_step_context"
        )  # (d_q_step_context, output_dim)

        # we need two Wk projections since there is MHA followed by 1-head attention - they have different keys K
        self.wk = tf.keras.layers.Dense(
            self.output_dim, use_bias=False, name="wk"
        )  # (d_k, output_dim)
        self.wk_tanh = tf.keras.layers.Dense(
            self.output_dim, use_bias=False, name="wk_tanh"
        )  # (d_k_tanh, output_dim)

        # we dont need Wv projection for 1-head attention: only need attention weights as outputs
        self.wv = tf.keras.layers.Dense(
            self.output_dim, use_bias=False, name="wv"
        )  # (d_v, output_dim)

        # we dont need wq for 1-head tanh attention, since we can absorb it into w_out
        self.w_out = tf.keras.layers.Dense(
            self.output_dim, use_bias=False, name="w_out"
        )  # (d_model, d_model)

        self.problem = SimosFoodGroup

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def split_heads(self, tensor, batch_size):
        """Function for computing attention on several heads simultaneously
        Splits last dimension of a tensor into (num_heads, head_depth).
        Then we transpose it as (batch_size, num_heads, ..., head_depth) so that we can use broadcast
        """
        tensor = tf.reshape(tensor, (batch_size, -1, self.num_heads, self.head_depth))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def _select_node(self, logits):
        """Select next node based on decoding type."""

        # assert tf.reduce_all(logits == logits), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            # probs = tf.exp(logits)
            # selected = tf.math.argmax(probs, axis=-1) # (batch_size, 1)
            selected = tf.math.argmax(logits, axis=-1)  # (batch_size, 1)

        elif self.decode_type == "sampling":
            # logits has a shape of (batch_size, 1, n_nodes), we have to squeeze it
            # to (batch_size, n_nodes) since tf.random.categorical requires matrix
            selected = tf.random.categorical(logits[:, 0, :], 1)  # (bach_size,1)
        else:
            assert False, "Unknown decode type"

        return tf.squeeze(selected, axis=-1)  # (bach_size,)

    def get_step_context(self, state, embeddings, vehicle_id):
        """Takes a state and graph embeddings,
        Returns a part [h_N, D] of context vector [h_c, h_N, D],
        that is related to RL Agent last step.
        """
        # index of previous node
        # prev_node = state.prev_a  # (batch_size, 1)
        # prev_node= tf.cast(tf.gather_nd(state.prev_a_list,vehicle_id,batch_dims = 1),tf.int64)
        # used_capacity =  tf.gather_nd(state.used_capacity_list,vehicle_id,batch_dims = 1)
        # vehicle_cap =   tf.gather(params = state.vehicle_list, indices = vehicle_id)

        # from embeddings=(batch_size, n_nodes, input_dim) select embeddings of previous nodes
        # cur_embedded_node = tf.gather(embeddings, tf.cast(prev_node, tf.int32), batch_dims=1)  # (batch_size, 1, input_dim)
        # step_context = tf.concat([cur_embedded_node, tf.expand_dims(vehicle_cap,axis = 1) - used_capacity[:, :, None]], axis=-1)

        # same with time
        # vehicle_time = tf.expand_dims(tf.gather(state.truck_cur_times,state.vehicle_id,batch_dims=1),axis = 1)
        # step_context = tf.concat([cur_embedded_node, tf.expand_dims(state.vehicle_cap,axis = 1) - state.used_capacity[:, :, None],vehicle_time], axis=-1)

        # all 43 embeding try

        # locations
        all_cur_embedded_node = tf.gather(
            embeddings, tf.cast(state.prev_a_list, tf.int32), batch_dims=1
        )
        # times
        times = state.truck_cur_times[:, None]
        times_list = [y[:, None] for y in tf.unstack(times, axis=1)]
        # demands
        dems = state.vehicle_list - state.used_capacity_list
        dems_list = [y[:, None] for y in tf.unstack(dems, axis=1)]

        # concat features
        all_cur_embedded_node_list = [
            x for x in tf.unstack(all_cur_embedded_node, axis=1)
        ]
        re = [
            tf.concat(x, -1)
            for x in zip(all_cur_embedded_node_list, dems_list, times_list)
        ]
        step_context = tf.concat(re, axis=-1)

        return step_context  # (batch_size, 1, input_dim + 1)

    def decoder_mha(self, Q, K, V, mask=None):
        """Computes Multi-Head Attention part of decoder
        Basically, its a part of MHA sublayer, but we cant construct a layer since Q changes in a decoding loop.
        Args:
            mask: a mask for visited nodes,
                has shape (batch_size, seq_len_q, seq_len_k), seq_len_q = 1 for context vector attention in decoder
            Q: query (context vector for decoder)
                    has shape (..., seq_len_q, head_depth) with seq_len_q = 1 for context_vector attention in decoder
            K, V: key, value (projections of nodes embeddings)
                have shape (..., seq_len_k, head_depth), (..., seq_len_v, head_depth),
                                                                with seq_len_k = seq_len_v = n_nodes for decoder
        """

        # batch_size = tf.shape(Q)[0]

        compatibility = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(
            self.dk_mha_decoder
        )  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # dk = tf.cast(tf.shape(K)[-1], tf.float32)
        # compatibility = compatibility / tf.math.sqrt(dk)
        # compatibility = compatibility / tf.math.sqrt(self.dk_mha_decoder)

        if mask is not None:

            # we need to reshape mask:
            # (batch_size, seq_len_q, seq_len_k) --> (batch_size, 1, seq_len_q, seq_len_k)
            # so that we will be able to do a broadcast:
            # (batch_size, num_heads, seq_len_q, seq_len_k) + (batch_size, 1, seq_len_q, seq_len_k)
            mask = mask[:, tf.newaxis, :, :]

            # we use tf.where since 0*-np.inf returns nan, but not -np.inf
            compatibility = tf.where(
                mask, tf.ones_like(compatibility) * (-np.inf), compatibility
            )

        compatibility = tf.nn.softmax(
            compatibility, axis=-1
        )  # (batch_size, num_heads, seq_len_q, seq_len_k)
        attention = tf.matmul(
            compatibility, V
        )  # (batch_size, num_heads, seq_len_q, head_depth)

        # transpose back to (batch_size, seq_len_q, num_heads, depth)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        # concatenate heads (last 2 dimensions)
        attention = tf.reshape(
            attention, (self.batch_size, -1, self.output_dim)
        )  # (batch_size, seq_len_q, output_dim)

        output = self.w_out(
            attention
        )  # (batch_size, seq_len_q, output_dim), seq_len_q = 1 for context att in decoder

        return output

    def get_log_p(self, Q, K, mask=None):
        """Single-Head attention sublayer in decoder,
        computes log-probabilities for node selection.
        Args:
            mask: mask for nodes
            Q: query (output of mha layer)
                    has shape (batch_size, seq_len_q, output_dim), seq_len_q = 1 for context attention in decoder
            K: key (projection of node embeddings)
                    has shape  (batch_size, seq_len_k, output_dim), seq_len_k = n_nodes for decoder
        """

        compatibility = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(
            self.dk_get_loc_p
        )

        # dk = tf.cast(tf.shape(K)[-1], tf.float32)
        # compatibility = compatibility / tf.math.sqrt(dk)
        # compatibility = compatibility / tf.math.sqrt(self.dk_get_loc_p)

        compatibility = tf.math.tanh(compatibility) * self.tanh_clipping

        if mask is not None:

            # we dont need to reshape mask like we did in multi-head version:
            # (batch_size, seq_len_q, seq_len_k) --> (batch_size, num_heads, seq_len_q, seq_len_k)
            # since we dont have multiple heads

            compatibility = tf.where(
                mask, tf.ones_like(compatibility) * (-np.inf), compatibility
            )

        log_p = tf.nn.log_softmax(
            compatibility, axis=-1
        )  # (batch_size, seq_len_q, seq_len_k)

        return log_p

    def call(
        self,
        inputs,
        embeddings,
        context_vectors,
        node_cap,
        vehicle_cap,
        duration_tensor,
        distance_tensor,
    ):
        # embeddings shape = (batch_size, n_nodes, input_dim)
        # context vectors shape = (batch_size, input_dim)
        self.embeddings = embeddings
        self.batch_size = tf.shape(self.embeddings)[0]

        outputs = []
        sequences = []

        state = self.problem(
            inputs, node_cap, vehicle_cap, duration_tensor, distance_tensor
        )

        # we compute some projections (common for each policy step) before decoding loop for efficiency
        K = self.wk(self.embeddings)  # (batch_size, n_nodes, output_dim)
        K_tanh = self.wk_tanh(self.embeddings)  # (batch_size, n_nodes, output_dim)
        V = self.wv(self.embeddings)  # (batch_size, n_nodes, output_dim)
        Q_context = self.wq_context(
            context_vectors[:, tf.newaxis, :]
        )  # (batch_size, 1, output_dim)

        # we dont need to split K_tanh since there is only 1 head; Q will be split in decoding loop
        K = self.split_heads(
            K, self.batch_size
        )  # (batch_size, num_heads, n_nodes, head_depth)
        V = self.split_heads(
            V, self.batch_size
        )  # (batch_size, num_heads, n_nodes, head_depth)

        # Perform decoding steps
        i = 0

        while not state.all_finished():

            for vehicle_id in range(len(state.vehicle_list)):
                v_id = vehicle_id
                vehicle_id = tf.repeat(
                    tf.expand_dims([vehicle_id], 0), self.batch_size, 0
                )
                # print(vehicle_id,state.spare[v_id])
                if state.spare[v_id]:
                    continue
                step_context = self.get_step_context(
                    state, self.embeddings, vehicle_id
                )  # (batch_size, 1, input_dim + 1)
                Q_step_context = self.wq_step_context(
                    step_context
                )  # (batch_size, 1, output_dim)
                Q = Q_context + Q_step_context

                # split heads for Q
                Q = self.split_heads(
                    Q, self.batch_size
                )  # (batch_size, num_heads, 1, head_depth)

                # get current mask
                mask = state.get_mask(
                    vehicle_id
                )  # (batch_size, 1, n_nodes) with True/False indicating where agent can go

                # compute MHA decoder vectors for current mask
                mha = self.decoder_mha(Q, K, V, mask)  # (batch_size, 1, output_dim)

                # compute probabilities
                log_p = self.get_log_p(mha, K_tanh, mask)  # (batch_size, 1, n_nodes)

                # next step is to select node
                selected = self._select_node(log_p)

                state.step(selected, vehicle_id)
                # print("move to ",selected.numpy())
                outputs.append(log_p[:, 0, :])
                sequences.append(selected)

                i += 1
        vehicles_used = tf.transpose(state.vehicles_used.concat())[0]
        cost = state.cost
        distance_cost = state.distance_cost
        earliness_cost = state.earliness_cost
        tardiness_cost = state.tardiness_cost
        # print(vehicles_used)

        # Collected lists, return Tensor
        return (
            tf.stack(outputs, 1),
            tf.cast(tf.stack(sequences, 1), tf.float32),
            vehicles_used,
            cost,
            distance_cost,
            earliness_cost,
            tardiness_cost,
        )

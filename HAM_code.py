def ham_block(input_feature, name):
    with tf.variable_scope(name):
        attention_feature, channel = channel_attention(input_feature, 'ch_at', gamma=2, b=1)
        attention_feature = spatial_attention(attention_feature, channel, 'sp_at')
    return attention_feature

# channel attention submodule
def channel_attention(input_feature, name, gamma=2, b=1):
    with tf.variable_scope(name):
        alpha = tf.get_variable('alpha', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5), trainable=True)
        beta = tf.get_variable('beta', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5), trainable=True)
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
        N, H, W, C = np.shape(input_feature)
        C = np.int(C)
        assert avg_pool.get_shape()[1:] == (1, 1, C)
        assert max_pool.get_shape()[1:] == (1, 1, C)

        # Adaptive Mechanism Block
        add = (avg_pool + max_pool) / 2 + alpha * avg_pool + beta * max_pool
        assert add.get_shape()[1:] == (1, 1, C)

        # fast one-dimensional convolutional layer
        t = np.int(np.abs(np.log2(C) + b) / gamma)
        k = t if t % 2 else t + 1
        add = tf.squeeze(add, 2)
        add_conv1d = tf.layers.conv1d(inputs=add,
                                      filters=C,
                                      kernel_size=k,
                                      strides=1,
                                      padding='same',
                                      use_bias=False,
                                      name='conv1d',
                                      reuse=None)
        add_conv1d = tf.expand_dims(add_conv1d, 2)
        assert add_conv1d.get_shape()[1:] == (1, 1, C)
        channel = tf.sigmoid(add_conv1d, 'sigmoid')
    return input_feature * channel, channel


# spatial attention submodule

def spatial_attention(input_feature, channel, name):
    kernel_size = 7

    # keep_rate: separation rate
    keep_rate = 0.6
    C = channel.get_shape()[-1]
    C = np.int(C)
    h = np.int(C * keep_rate)
    if h % 2 == 1:
        k = h + 1
    else:
        k = h
    t = C - k

    # channel separation technique
    channel_top, channel_top_index = tf.nn.top_k(channel, k)
    kth = tf.reduce_min(channel_top, axis=3, keepdims=True)
    topk = tf.greater_equal(channel, kth)
    lessk = tf.less(channel, kth)
    im_mask = tf.cast(topk, dtype=tf.float32)
    sub_mask = tf.cast(lessk, dtype=tf.float32)
    input_feature_immask = tf.multiply(input_feature, im_mask)
    input_feature_submask = tf.multiply(input_feature, sub_mask)
    with tf.variable_scope(name):
        # important group
        avg_pool_im = tf.reduce_sum(input_feature_immask, axis=[3], keepdims=True)
        avg_pool_im = tf.divide(avg_pool_im, k)
        assert avg_pool_im.get_shape()[-1] == 1
        max_pool_im = tf.reduce_max(input_feature_immask, axis=[3], keepdims=True)
        assert max_pool_im.get_shape()[-1] == 1
        concat_im = tf.concat([avg_pool_im, max_pool_im], 3)
        assert concat_im.get_shape()[-1] == 2
        concat_im = tf.layers.conv2d(inputs=concat_im,
                                     filters=1,
                                     kernel_size=(kernel_size, kernel_size),
                                     strides=(1, 1),
                                     padding='same',
                                     activation=None,
                                     kernel_initializer='he_normal',
                                     use_bias=False,
                                     name='conv',
                                     reuse=None)
        assert concat_im.get_shape()[-1] == 1
        concat_im = tf.layers.batch_normalization(concat_im, training=True)
        concat_im = tf.nn.relu(concat_im, 'relu')
        concat_im = tf.sigmoid(concat_im, 'sigmoid')
        spatial_im = input_feature_immask * concat_im
       
        # sub-important group
        avg_pool_sub = tf.reduce_sum(input_feature_submask, axis=[3], keepdims=True)
        avg_pool_sub = tf.divide(avg_pool_sub, t)
        assert avg_pool_sub.get_shape()[-1] == 1
        max_pool_sub = tf.reduce_max(input_feature_submask, axis=[3], keepdims=True)
        assert max_pool_sub.get_shape()[-1] == 1
        concat_sub = tf.concat([avg_pool_sub, max_pool_sub], 3)
        assert concat_sub.get_shape()[-1] == 2
        concat_sub = tf.layers.conv2d(inputs=concat_sub,
                                      filters=1,
                                      kernel_size=(kernel_size, kernel_size),
                                      strides=(1, 1),
                                      padding='same',
                                      activation=None,
                                      use_bias=False,
                                      name='conv',
                                      reuse=True)
        assert concat_sub.get_shape()[-1] == 1
        concat_sub = tf.layers.batch_normalization(concat_sub, training=True)
        concat_sub = tf.nn.relu(concat_sub, 'relu')
        concat_sub = tf.sigmoid(concat_sub, 'sigmoid')
        spatial_sub = input_feature_submask * concat_sub

        final_spatial = spatial_im + spatial_sub
        assert final_spatial.get_shape()[3] == C
    return final_spatial
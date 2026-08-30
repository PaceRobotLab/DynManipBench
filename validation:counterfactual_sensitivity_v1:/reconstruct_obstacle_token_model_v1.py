import tensorflow as tf
from tensorflow.keras import layers, Model


def dense_prelu(x, units, name):
    x = layers.Dense(units, name=f"{name}_dense")(x)
    x = layers.PReLU(name=f"{name}_prelu")(x)
    return x


def self_attention(x, heads, key_dim, name, normalize=False):
    width = int(x.shape[-1])
    y = layers.MultiHeadAttention(
        num_heads=heads,
        key_dim=key_dim,
        output_shape=width,
        name=f"{name}_mha",
    )(x, x)
    if normalize:
        y = layers.LayerNormalization(name=f"{name}_norm")(y)
    return y


def build_model(heads=3, key_dim=16):
    """
    DynManipBench obstacle-token successor v1.

    Key experimental change from the Liang Figure-30 reconstruction:
      - obsInput remains externally compatible as shape (1,80)
      - internally it is reshaped into 10 obstacle tokens x 8 features:
            [previous x,y,z,size, current x,y,z,size]
      - MultiHeadAttention operates across the 10 obstacle tokens
        rather than across a sequence of length 1.

    This is NOT intended to reproduce Liang's 505,585-parameter model.
    It is a successor architecture for controlled comparison.
    """

    rb_input = layers.Input(shape=(1, 21), name="rbInput")
    const_input = layers.Input(shape=(1, 112), name="constInput")
    obs_input = layers.Input(shape=(1, 80), name="obsInput")
    num_obs_input = layers.Input(shape=(1, 1), name="numOfObsInput")

    # Robot branch: preserve Figure-30 widths.
    r = dense_prelu(rb_input, 42, "rb0")
    r = dense_prelu(r, 24, "rb1")
    r = dense_prelu(r, 16, "rb2")
    r = self_attention(r, heads, key_dim, "rb_attention", normalize=True)

    # Constant branch: preserve Figure-30 widths.
    c = dense_prelu(const_input, 44, "const0")
    c = dense_prelu(c, 28, "const1")
    c = dense_prelu(c, 18, "const2")
    c = self_attention(c, heads, key_dim, "const_attention", normalize=True)

    # ------------------------------------------------------------
    # Obstacle branch
    #
    # External obsInput layout:
    #   first  40 = previous snapshot, 10 x [x,y,z,size]
    #   second 40 = current  snapshot, 10 x [x,y,z,size]
    #
    # Convert to:
    #   (batch,10,8), one token per obstacle.
    # ------------------------------------------------------------

    def make_tokens(z):
        z = tf.reshape(z, (-1, 80))
        prev = tf.reshape(z[:, :40], (-1, 10, 4))
        curr = tf.reshape(z[:, 40:], (-1, 10, 4))
        return tf.concat([prev, curr], axis=-1)

    o = layers.Lambda(
        make_tokens,
        output_shape=(10, 8),
        name="obs_make_10x8_tokens",
    )(obs_input)

    # Lift each obstacle token before attention.
    o = dense_prelu(o, 24, "obs_token_embed")

    # Create a mask from numOfObsInput.  The current reconstructed
    # corpus uses 10 obstacles, but this makes the input functional
    # and supports counts <= 10.
    def make_mask(n):
        n = tf.cast(tf.reshape(n, (-1,)), tf.int32)
        valid = tf.sequence_mask(n, maxlen=10)        # (B,10)
        return tf.logical_and(
            valid[:, :, tf.newaxis],                 # (B,10,1)
            valid[:, tf.newaxis, :],                 # (B,1,10)
        )                                             # (B,10,10)

    mask = layers.Lambda(
        make_mask,
        output_shape=(10, 10),
        name="obs_attention_mask",
    )(num_obs_input)

    o = layers.MultiHeadAttention(
        num_heads=heads,
        key_dim=key_dim,
        output_shape=24,
        name="obs_attention_mha",
    )(o, o, attention_mask=mask)

    o = layers.LayerNormalization(name="obs_attention_norm")(o)

    # Preserve the Figure-30 obstacle branch's final representation width 24.
    o = dense_prelu(o, 48, "obs0")
    o = dense_prelu(o, 36, "obs1")
    o = dense_prelu(o, 24, "obs2")

    # Mask invalid obstacle tokens before pooling.
    def valid_mask(n):
        n = tf.cast(tf.reshape(n, (-1,)), tf.int32)
        return tf.cast(tf.sequence_mask(n, maxlen=10), tf.float32)[:, :, None]

    vm = layers.Lambda(
        valid_mask,
        output_shape=(10, 1),
        name="obs_valid_mask",
    )(num_obs_input)

    o = layers.Multiply(name="obs_masked_features")([o, vm])

    def masked_mean(args):
        features, m, n = args
        summed = tf.reduce_sum(features, axis=1, keepdims=True)
        denom = tf.maximum(tf.cast(n, tf.float32), 1.0)
        return summed / denom

    o = layers.Lambda(
        masked_mean,
        output_shape=(1, 24),
        name="obs_masked_mean_pool",
    )([o, vm, num_obs_input])

    o = layers.LayerNormalization(name="obs_norm")(o)

    # Fusion: same widths as Figure-30 reconstruction.
    x = layers.Concatenate(axis=-1, name="fusion")([r, c, o])

    if int(x.shape[-1]) != 58:
        raise RuntimeError(f"Expected fusion width 58, got {x.shape[-1]}")

    x = self_attention(x, heads, key_dim, "fusion_attention", normalize=False)

    x = dense_prelu(x, 189, "post0")
    x = layers.Dropout(0.2, name="dropout0")(x)

    # Preserve the exact-candidate choice used in the baseline.
    x = layers.Dense(126, name="post1_dense")(x)
    x = layers.Dropout(0.2, name="dropout1")(x)

    x = dense_prelu(x, 104, "post2")
    x = dense_prelu(x, 72, "post3")
    x = dense_prelu(x, 64, "post4")
    x = dense_prelu(x, 42, "post5")
    x = dense_prelu(x, 28, "post6")

    output = layers.Dense(7, name="next_configuration")(x)

    return Model(
        inputs=[rb_input, const_input, obs_input, num_obs_input],
        outputs=output,
        name="DynManipBench_obstacle_token_v1",
    )


def main():
    print("DynManipBench obstacle-token successor v1")
    print("=" * 72)

    for heads, key_dim in [(3, 16), (4, 16), (6, 16), (3, 32)]:
        tf.keras.backend.clear_session()
        model = build_model(heads=heads, key_dim=key_dim)
        print(
            f"heads={heads:2d} key_dim={key_dim:3d} "
            f"params={model.count_params():,}"
        )

    print()
    print("Default model summary")
    print("-" * 72)

    tf.keras.backend.clear_session()
    model = build_model()
    model.summary()


if __name__ == "__main__":
    main()

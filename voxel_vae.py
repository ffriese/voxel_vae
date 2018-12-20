import numpy as np
import tensorflow as tf


class VAE(object):
    def __init__(self, input_shape, latent_dimensions, beta=1):
        self.beta = beta
        self.input_shape = input_shape
        self.enc_in = tf.placeholder(tf.float32, (100,) + input_shape)
        kernel = 4
        stride = (2, 2, 2)
        num_filter = 32

        # batch size + spatial shape + channel size
        layer_1 = tf.contrib.layers.conv3d(inputs=self.enc_in, num_outputs=num_filter, stride=stride,
                                           kernel_size=kernel)
        layer_2 = tf.contrib.layers.conv3d(inputs=layer_1, num_outputs=num_filter, stride=stride,
                                           kernel_size=kernel)
        layer_3 = tf.contrib.layers.conv3d(inputs=layer_2, num_outputs=num_filter*2, stride=stride,
                                           kernel_size=kernel)
        layer_4 = tf.contrib.layers.conv3d(inputs=layer_3, num_outputs=num_filter*2, stride=stride,
                                           kernel_size=kernel)

        layer_flatten = tf.contrib.layers.flatten(inputs=layer_4)

        layer_idk = tf.contrib.layers.fully_connected(inputs=layer_flatten, num_outputs=256)

        self.mus = tf.contrib.layers.fully_connected(inputs=layer_idk, num_outputs=latent_dimensions,
                                                     activation_fn=None)

        self.log_var = tf.contrib.layers.fully_connected(inputs=layer_idk, num_outputs=latent_dimensions,
                                                         activation_fn=None)
        epsilons = tf.random_normal(self.mus.shape)
        self.z = self.mus + tf.exp(0.5 * self.log_var) * epsilons

        layer_dense_upsample = tf.contrib.layers.fully_connected(inputs=self.z, num_outputs=int(np.prod(layer_4.shape[1:])))
        layer_unflatten = tf.reshape(layer_dense_upsample, layer_4.shape)

        layer_de_1 = tf.contrib.layers.conv3d_transpose(inputs=layer_unflatten, num_outputs=num_filter*2, stride=stride,
                                                        kernel_size=kernel)
        layer_de_2 = tf.contrib.layers.conv3d_transpose(inputs=layer_de_1, num_outputs=num_filter*2, stride=stride,
                                                        kernel_size=kernel)
        layer_de_3 = tf.contrib.layers.conv3d_transpose(inputs=layer_de_2, num_outputs=num_filter, stride=stride,
                                                        kernel_size=kernel)
        layer_de_4 = tf.contrib.layers.conv3d_transpose(inputs=layer_de_3, num_outputs=num_filter, stride=stride,
                                                        kernel_size=kernel)
        self.dec_out = tf.contrib.layers.conv3d_transpose(inputs=layer_de_4, num_outputs=1, stride=1,
                                                          kernel_size=1)

        print(input_shape, layer_de_4.shape[1:], self.dec_out.shape[1:])
        assert input_shape == self.dec_out.shape[1:]
        self.optimizer = tf.train.AdamOptimizer()

        kl_loss = 1 + self.log_var - tf.pow(self.mus, 2) - tf.exp(self.log_var)
        kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=-1)
        reconstruction_err = tf.losses.mean_squared_error(labels=self.enc_in, predictions=self.dec_out)
        reconstruction_err * np.prod(self.input_shape[:-1])
        self.loss = tf.reduce_mean(self.beta * kl_loss + reconstruction_err)
        self.train_op = self.optimizer.minimize(self.loss)

    def train(self, batch_size, num_episodes, data):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for i in range(num_episodes):
                print('episode %r' % i)
                np.random.shuffle(data)
                for j in range(0, len(data), batch_size):
                    print(j)
                    if len(data) > j+batch_size:
                        batch = np.expand_dims(data[j:j+batch_size], axis=-1)
                        _, loss = session.run([self.train_op, self.loss], feed_dict={self.enc_in: batch})
                        print('loss:%r' % loss)


if __name__ == '__main__':
    vae = VAE((16, 16, 16, 1), 100)
    train_data = np.load("/home/ffriese/prj-robotic-arms/voxel_vae/transformed voxel_data.npy")

    for sample in train_data[:1]:
        print(sample.shape)

    vae.train(100, 100, train_data)



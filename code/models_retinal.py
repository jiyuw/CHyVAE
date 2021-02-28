from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
import os
import datasets
import utils
import pickle
from tqdm import tqdm
tf.get_logger().setLevel('INFO')

ds = tfp.distributions


class CHyVAE:
    def __init__(self, train_set, test_set, n_clusters, z_dim, im_h, im_w, channels, batch_size, n_epochs, nu, prior_cov, run_no=None):
        self.trainset = train_set
        self.testset = test_set
        self.n_clusters = n_clusters
        self.z_dim = z_dim
        self.im_h = im_h
        self.im_w = im_w
        self.channels = channels
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.nu_np = nu
        self.n_fc_units = 256
        scale = max(self.nu_np - self.z_dim - 1, 1)
        self.Psi_np = scale * prior_cov
        self.results_path = '../results/retianl_z_{:d}_nu_{:d}_run{:d}/'.format(z_dim, nu, run_no)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        self._build_model()
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = '1'
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)

    def _encoder(self, x, reuse=False):
        with tf.compat.v1.variable_scope('encoder', reuse=reuse):
            z = tf.keras.layers.Conv2D(32, 4, 2, padding='same', activation=tf.nn.relu)(x)
            z = tf.keras.layers.Conv2D(32, 4, 2, padding='same', activation=tf.nn.relu)(z)
            z = tf.keras.layers.Conv2D(64, 4, 2, padding='same', activation=tf.nn.relu)(z)
            z = tf.keras.layers.Conv2D(64, 4, 2, padding='same', activation=tf.nn.relu)(z)
            z = tf.keras.layers.Flatten()(z)
            z = tf.keras.layers.Dense(self.n_fc_units, activation=tf.nn.relu)(z)
            mu = tf.keras.layers.Dense(self.z_dim, activation=None)(z)
            A = tf.reshape(tf.keras.layers.Dense(self.z_dim * self.z_dim, activation=None)(z), [-1, self.z_dim, self.z_dim])
            L = tf.linalg.band_part(A, -1, 0)
            diag = tf.nn.softplus(tf.linalg.diag_part(A)) + 1e-4
            L = tf.linalg.set_diag(L, diag)
            L_LT = tf.matmul(L, L, transpose_b=True)
            Sigma = L_LT + 1e-4 * tf.eye(self.z_dim)
            return mu, Sigma

    def _decoder(self, z, reuse=False):
        with tf.compat.v1.variable_scope('decoder', reuse=reuse):
            z = tf.keras.layers.Dense(self.n_fc_units, activation=tf.nn.relu)(z)
            z = tf.keras.layers.Dense(31 * 48 * 64, activation=tf.nn.relu)(z)
            z = tf.reshape(z, [-1, 31, 48, 64])
            z = tf.keras.layers.Conv2DTranspose(64, 4, 2, padding='same', activation=tf.nn.relu)(z)
            z = tf.keras.layers.Conv2DTranspose(32, 4, 2, padding='same', activation=tf.nn.relu)(z)
            z = tf.keras.layers.Conv2DTranspose(32, 4, 2, padding='same', activation=tf.nn.relu)(z)
            x_logits = tf.keras.layers.Conv2DTranspose(self.channels, 4, 2, padding='same', activation=None)(z)
            return x_logits

    def _regularizer(self, z, mu, Sigma, Psi, nu, B):
        psi_zzT = Psi + tf.matmul(z, z, transpose_b=True)
        mu = tf.expand_dims(mu, -1)
        sigma_mumuT_psi = Sigma + tf.matmul(mu, mu, transpose_b=True) + Psi
        return -\
            0.5 * (nu + 1) * (tf.linalg.logdet(psi_zzT)) +\
            0.5 * tf.linalg.logdet(Sigma) -\
            0.5 * (nu + B) * tf.linalg.trace(tf.matmul(sigma_mumuT_psi, tf.matrix_inverse(psi_zzT)))

    def _build_model(self):
        # Model
        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.im_h, self.im_w, self.channels])
        self.Psi = tf.compat.v1.placeholder(tf.float32, [self.z_dim, self.z_dim])
        self.nu = tf.compat.v1.placeholder(tf.float32, ())
        self.mu, Sigma = self._encoder(self.x)
        mvn = ds.MultivariateNormalFullCovariance(loc=self.mu, covariance_matrix=Sigma)
        z = mvn.sample()
        z2 = tf.transpose(mvn.sample(1), perm=[1, 2, 0])
        x_hat_logits = self._decoder(z)
        self.loglikelihood = tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=x_hat_logits), [1, 2, 3]))
        self.regularizer = -tf.reduce_mean(self._regularizer(
            z2, self.mu, Sigma, self.Psi, self.nu, 1))
        self.loss = self.loglikelihood + self.regularizer
        self.optim_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        # Reconstruction
        self.x_test = tf.compat.v1.placeholder(tf.float32, [None, self.im_h, self.im_w, self.channels])
        z_test = self._encoder(self.x_test, reuse=True)[0]
        self.x_recons = tf.nn.sigmoid(self._decoder(z_test, reuse=True))

        # Generation
        self.noise = tf.compat.v1.placeholder(tf.float32, [None, self.z_dim])
        self.fake_images = tf.nn.sigmoid(self._decoder(self.noise, reuse=True))

    def train(self):
        train_path = os.path.join(self.results_path, 'train')
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        metrics_history = {'epoch': [], 'recons': [], 'disent': [], 'acc': []}
        n_batch = len(self.trainset)
        for epoch in range(1, self.n_epochs + 1):
            print(f'epoch {epoch}')
            loss = 0
            loglh = 0
            regr = 0
            start_time = time.time()
            for idx, (x_np, _) in enumerate(
                    tqdm(self.trainset, desc='training', total=len(self.trainset), ascii=True)):
                _, loss_np, rec_np, reg_np = self.sess.run([self.optim_op, self.loss, self.loglikelihood, self.regularizer], feed_dict={self.x: x_np, self.Psi: self.Psi_np, self.nu: self.nu_np})
                loss += loss_np
                loglh += rec_np
                regr += reg_np

            # epoch stats
            end_time = time.time()
            loss /= n_batch
            loglh /= n_batch
            regr /= n_batch
            print('Epoch: {:d} in {:.2f}s:: Loss: {:.3f} => Recons.: {:.3f}, Reg: {:.3f}'.format(
                epoch, end_time - start_time, loss, -loglh, -regr))

            # epoch eval
            for idx, (inputs, targets) in enumerate(
                    tqdm(self.testset, desc='evaluation', total=len(self.testset), ascii=True)):
                zi = self.sess.run(self.mu, feed_dict={self.x: inputs})
                if idx == 0:
                    mus = zi
                    y = np.array(targets)
                else:
                    mus = np.concatenate((mus, np.array(zi)), axis=0)
                    y = np.concatenate((y, np.array(targets)), axis=0)
            disent_metric = utils.compute_metric_retinal(mus, y, self.z_dim, n_clusters = self.n_clusters)
            metrics_history['epoch'].append(epoch)
            metrics_history['recons'].append(-loglh)
            metrics_history['disent'].append(disent_metric)
            print('Metric: {:.4f}'.format(disent_metric))

        with open(os.path.join(train_path, 'metrics.pkl'), 'wb') as pkl:
            pickle.dump(metrics_history, pkl)

    def generate(self):
        gen_path = os.path.join(self.results_path, 'gen')
        if not os.path.exists(gen_path):
            os.mkdir(gen_path)
        nu = self.z_dim + 1
        while True:
            z_np = utils.sample_noise(self.Psi_np, self.nu_np, 100)
            x_hat_np = self.sess.run(self.fake_images, feed_dict={self.noise: z_np})
            utils.render_images(x_hat_np, os.path.join(gen_path, 'nu_{:d}.png'.format(nu)))
            if nu >= 2 * self.nu_np:
                break
            nu = nu * 4

        x_test_np = next(self.test_batches)
        x_np = np.vstack([x_test_np, next(self.train_batches), next(self.test_batches)])
        means, = self.sess.run([self.mu], feed_dict={self.x: x_np})
        for num, base_point in enumerate(means):
            n_images_per_latent = 20
            z_np = []
            for i in range(self.z_dim):
                dim_i = np.repeat(base_point[None, :], n_images_per_latent, 0)
                dim_i[np.arange(n_images_per_latent), i] = np.linspace(-3, 3, n_images_per_latent)
                z_np.append(dim_i)
            z_np = np.vstack(z_np)
            x_hat_np = self.sess.run(self.fake_images, feed_dict={self.noise: z_np})
            img_path = os.path.join(gen_path, 'inter_{:d}_{:d}.png'.format(1, num))
            utils.render_images(x_hat_np, img_path, n_rows=self.z_dim, n_cols=n_images_per_latent)

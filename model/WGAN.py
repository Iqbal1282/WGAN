from tensorflow.keras.initializers import RandomNormal 
from tensorflow.keras.layers import Input, Activation,LeakyReLU, Flatten , Reshape, Conv2DTranspose, UpSampling2D, Conv2D, Dense,BatchNormalization
from tensorflow.keras.models import Model 
from tensorflow.keras import backend as k 
from tensorflow.keras.optimizers import RMSprop 
import numpy as np 
from tensorflow.keras.utils import plot_model 

import os 
import matplotlib.pyplot as plt 
import pickle 


class WGAN():
	def __init__(self):
		self.name = 'gan'
		self.weight_init = RandomNormal(mean = 0, stddev = 0.02)
		self.z_dim = 100 
		self.epoch = 0 
		self.d_losses = []
		self.g_losses = []

		self._build_critic()
		self._build_generator()
		self._build_adversarial()



	def wasserstein(self, y_true, y_pred):
		return -k.mean(y_true*y_pred)

	def get_activation(self, activation):
		if activation == 'leaky_relu':
			layer = LeakyReLU(alpha = 0.2)
		else:
			layer = Activation(activation)

		return layer 


	def _build_critic(self):
		critic_input = Input(shape = (32, 32, 3), name = 'critic_input')
		x = critic_input

		# layer 1 

		x = Conv2D(
			filters = 32,
			kernel_size = 5, 
			strides = 2,
			padding = 'same',
			name = 'critic_conv_0',
			kernel_initializer = self.weight_init 
			)(x)

		x = self.get_activation('leaky_relu')(x)


		#### layer 2 

		x = Conv2D(
			filters = 64,
			kernel_size = 5, 
			strides = 2,
			padding = 'same',
			name = 'critic_conv_1',
			kernel_initializer = self.weight_init 
			)(x)

		x = self.get_activation('leaky_relu')(x)

		### layer 3 

		x = Conv2D(
			filters = 128,
			kernel_size = 5, 
			strides = 2,
			padding = 'same',
			name = 'critic_conv_2',
			kernel_initializer = self.weight_init 
			)(x)

		x = self.get_activation('leaky_relu')(x)

		## layer 4 
		x = Conv2D(
			filters = 128,
			kernel_size = 5, 
			strides = 1,
			padding = 'same',
			name = 'critic_conv_3',
			kernel_initializer = self.weight_init 
			)(x)

		x = self.get_activation('leaky_relu')(x)

		x = Flatten()(x)

		critic_output = Dense(1, activation = None, kernel_initializer = self.weight_init)(x)

		self.critic = Model(critic_input, critic_output)


	def _build_generator(self):
		generator_input = Input(shape = (self.z_dim,), name ='generator_input')
		x = generator_input
		x = Dense(np.prod([4,4,128]), kernel_initializer = self.weight_init)(x)
		x = BatchNormalization(momentum = 0.8)(x)
		x = LeakyReLU(alpha = 0.2)(x)
		x = Reshape([4,4,128])(x)

		## layer 1 
		x = UpSampling2D()(x)
		x = Conv2D(
			filters = 128, 
			kernel_size = 5, 
			padding = 'same',
			strides = 1, 
			name = 'generator_conv_0'
			)(x)
		x = BatchNormalization(momentum = 0.8)(x)
		x = LeakyReLU(alpha = 0.2)(x)

		## layer 2 
		x = UpSampling2D()(x)
		x = Conv2D(
			filters = 64, 
			kernel_size = 5, 
			padding = 'same',
			strides = 1, 
			name = 'generator_conv_1'
			)(x)
		x = BatchNormalization(momentum = 0.8)(x)
		x = LeakyReLU(alpha = 0.2)(x)

		# layer 3 
		x = UpSampling2D()(x)

		x = Conv2D(
			filters = 32, 
			kernel_size = 5, 
			padding = 'same',
			strides = 1, 
			name = 'generator_conv_2'
			)(x)
		x = BatchNormalization(momentum = 0.8)(x)
		x = LeakyReLU(alpha = 0.2)(x)

		# layer 4 
		x = Conv2DTranspose(
			filters = 3, 
			kernel_size = 5, 
			padding = 'same',
			strides = 1, 
			name = 'generator_conv_3'
			)(x)
		
		x = Activation('tanh')(x)


		generator_output = x 

		self.generator = Model(generator_input, generator_output)
	def set_trainable(self, m, val):
		m.trainable = val 
		for l in m.layers: 
			l.trainable = val 


	def _build_adversarial(self):
		### 

		self.critic.compile(
			optimizer= RMSprop(lr= 0.00005), 
			loss = self.wasserstein
			)

		self.set_trainable(self.critic, False)

		model_input = Input(shape = (self.z_dim,), name = 'model_input')
		x = self.generator(model_input)
		model_output = self.critic(x)

		self.model = Model(model_input, model_output)

		self.model.compile(
			optimizer = RMSprop(0.00005),
			loss = self.wasserstein
			)

		self.set_trainable(self.critic, True)


	def train_critic(self, x_train, batch_size, clip_threshold, using_generator):
		valid = np.ones((batch_size, 1))
		fake = -1 * np.ones((batch_size, 1))

		if using_generator: 
			true_imgs = next(x_train)[0]
			if true_imgs.shape[0] != batch_size: 
				true_imgs = next(x_train)[0]

		else: 
			idx = np.random.randint(0, x_train.shape[0], batch_size)
			true_imgs = x_train[idx]

		noise = np.random.normal( 0, 1, (batch_size, self.z_dim))
		gen_imgs = self.generator.predict(noise)
		d_loss_real = self.critic.train_on_batch(true_imgs , valid)
		d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
		d_loss = 0.5 *(d_loss_fake + d_loss_real)


		for l in self.critic.layers: 
			weights = l.get_weights()
			weights = [np.clip(w, -clip_threshold, clip_threshold) for w in weights]
			l.set_weights(weights)

		return [d_loss, d_loss_real, d_loss_fake]


	def train_generator(self, batch_size):
		valid = np.ones((batch_size, 1))
		noise = np.random.normal(0,1, (batch_size, self.z_dim))
		return self.model.train_on_batch(noise, valid)


	def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches = 10,
			n_critic = 5,
			clip_threshold = 0.01,
			using_generator = False
		):
		for epoch in range(self.epoch, self.epoch+epochs):
			for _ in range(n_critic):
				d_loss = self.train_critic(x_train, batch_size, clip_threshold, using_generator)

			g_loss = self.train_generator(batch_size)

			print ("%d [D loss: (%.3f)(R %.3f, F %.3f)]  [G loss: %.3f] " % (epoch, d_loss[0], d_loss[1], d_loss[2], g_loss))

			self.d_losses.append(d_loss)
			self.g_losses.append(g_loss)


			# If at save interval => save generated image samples
			if epoch % print_every_n_batches == 0:
				self.sample_images(run_folder)
				self.model.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (epoch)))
				self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
				self.save_model(run_folder)
			
			self.epoch += 1 


	def sample_images(self, run_folder):
		r, c = 5, 5
		noise = np.random.normal(0, 1, (r * c, self.z_dim))
		gen_imgs = self.generator.predict(noise)

		#Rescale images 0 - 1

		gen_imgs = 0.5 * (gen_imgs + 1)
		gen_imgs = np.clip(gen_imgs, 0, 1)

		fig, axs = plt.subplots(r, c, figsize=(15,15))
		cnt = 0

		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(np.squeeze(gen_imgs[cnt, :,:,:]), cmap = 'gray_r')
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % self.epoch))
		plt.close()


	def plot_model(self, run_folder):
		plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
		plot_model(self.critic, to_file=os.path.join(run_folder ,'viz/critic.png'), show_shapes = True, show_layer_names = True)
		plot_model(self.generator, to_file=os.path.join(run_folder ,'viz/generator.png'), show_shapes = True, show_layer_names = True)




	def save(self, folder):
			self.plot_model(folder)

	def save_model(self, run_folder):
		self.model.save(os.path.join(run_folder, 'model.h5'))
		self.critic.save(os.path.join(run_folder, 'critic.h5'))
		self.generator.save(os.path.join(run_folder, 'generator.h5'))
		#pickle.dump(self, open( os.path.join(run_folder, "obj.pkl"), "wb" ))

	def load_weights(self, filepath):
		self.model.load_weights(filepath)


















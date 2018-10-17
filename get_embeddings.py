import tensorflow as tf
import os
from tensorflow.python.platform import gfile
import numpy as np

# 1. load images
# 2. load the trained model. here, I use the model from
# https://github.com/davidsandberg/facenet
# 3. run images through the trained model

IMAGES_DIR = './img'
MODEL_DIR = './20180402-114759'

def main(images_dir, model_dir):
	with tf.Session() as sess:
		load_model(model_dir)
		images = load_images(images_dir, [160, 160])
		images = images.batch(batch_size=1, drop_remainder=True)		
		iterator = images.make_one_shot_iterator()
		# next_element = iterator.get_next()
		# image_batch = tf.train.batch_join(next_element, batch_size=128,
  #                       enqueue_many=False, allow_smaller_final_batch=True)
		
		# print(next_element.shape)
		# print(next_element)
		
		init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init)

		embeddings_layer = tf.get_default_graph().get_tensor_by_name('embeddings:0')
		# print(embeddings_layer.shape)
		img_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
		phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
		# print(phase_train_placeholder.__class__)
		# res = img_placeholder(next_element)
		embeddings_array = None

		while True:
			try:
				next_element = iterator.get_next()
				next_element = sess.run(next_element)
				embeddings = sess.run(embeddings_layer, feed_dict={img_placeholder: next_element,
										phase_train_placeholder: False})
				# print(embeddings.shape)
				# print(embeddings)
				# print(embeddings)
				# print('*' * 40)
				# print(embeddings_array)
				# print('*' * 40)

				embeddings_array = np.concatenate((embeddings_array, embeddings)) if embeddings_array is not None else embeddings
				# print('Emb array')
				# print(embeddings_array)
				

			except tf.errors.OutOfRangeError:
				break

	return embeddings



def load_model(model_dir):
	"""
	load pretrained facenet
	"""
	for fname in os.listdir(model_dir):
		if os.path.splitext(fname)[1] == '.pb':
			model_file = os.path.join(MODEL_DIR, fname)
			break
	# print(model_file)

	assert model_file is not None

	with gfile.FastGFile(model_file, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')


def load_images(images_dir, size):
	"""
	:param images_dir: directory with images (string)
	"""
	try:
		filenames = tf.constant([IMAGES_DIR + '/' + x for x in os.listdir(images_dir)])
		dataset = tf.data.Dataset.from_tensor_slices(filenames)
		dataset = dataset.map(lambda img: load_and_process_image(img, size=size))

		return dataset

	except OSError as e:
		print(e)


def load_and_process_image(image, size):
	"""
	:param image: image file
	:returns img: decoded image array
	"""
	contents = tf.read_file(image)
	img = tf.image.decode_jpeg(contents, channels=3)
	processed = resize_and_standardize_image(img, size)

	return processed

def resize_and_standardize_image(image, size):
	"""
	:param image: decoded image
	:param size: list(height, width)
	:return img: resized image:
	"""
	cropped = tf.random_crop(image, size=size + [3])
	cropped.set_shape(size + [3])
	standardized = tf.image.per_image_standardization(cropped)

	return standardized


if __name__ == '__main__':
	main(IMAGES_DIR, MODEL_DIR)
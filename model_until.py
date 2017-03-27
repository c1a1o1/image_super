import os
import tensorflow as tf
import scipy.misc

# save model
def save(checkpoint_dir, saver, sess, step):
    #oldname = 'checkpoint_old.txt'
    #newname = 'checkpoint_new.txt'

    #oldname = os.path.join(checkpoint_dir, oldname)
    #newname = os.path.join(checkpoint_dir, newname)

    #try:
     #   tf.gfile.Remove(oldname)
     #   tf.gfile.Remove(oldname + '.meta')
    #except:
     #   pass

    #try:
     #  tf.gfile.Rename(newname, oldname)
     #  tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    #except:
     #   pass

    #saver.save(sess, newname)
    model_name = 'ImprovedGAN.model'
    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)
    print(" Checkpoint saved")


# In[ ]:

def load(checkpoint_dir, saver, sess):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        print(" [*] Failed to find a checkpoint")
        return False

def setup_inputs(sess, image_size, filenames, batch_size, capacity_factor = 3):
    """read each JPEG file"""
    reader = tf.WholeFileReader()
    filename_queue = tf.train.string_input_producer(filenames)
    key, value = reader.read(filename_queue)
    channels = 3

    #read one image
    image = tf.image.decode_jpeg(value, channels)
    image.set_shape([None, None, channels])

    #crop and other random augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.95, 1.05)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.95, 1.05)

    wiggle = 8
    off_x, off_y = 25-wiggle, 60-wiggle
    crop_size = 128
    crop_size_plus = crop_size + 2 * wiggle
    image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_plus, crop_size_plus)
    image = tf.random_crop(image, [crop_size, crop_size, 3])

    image = tf.reshape(image, [1, crop_size, crop_size, 3])
    image = tf.cast(image, tf.float32)/255.0

    if crop_size != image_size:
        image = tf.image.resize_area(image, [image_size, image_size])

    #the feature is simply a k downscaled version
    K = 4
    downsampled = tf.image.resize_area(image, [image_size // K, image_size // K])

    feature = tf.reshape(downsampled, [image_size // K, image_size // K, 3])
    label = tf.reshape(image, [image_size, image_size, 3])

    features, labels = tf.train.batch([feature, label], batch_size, num_threads = 4, capacity = capacity_factor * batch_size)

    tf.train.start_queue_runners(sess = sess)

    return features, labels

def summarize_progress(sess, train_dir, feature, label, gene_output, batch, suffix, max_samples = 8):
    size = [label.shape[1], label.shape[2]]

    nearest = tf.image.resize_nearest_neighbor(feature, size)
    nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

    bicubic = tf.image.resize_bicubic(feature, size)
    bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

    clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

    label = tf.maximum(tf.minimum(label, 1.0), 0.0)

    image = tf.concat(axis=2, values=[nearest, bicubic, clipped, label])

    image = image[0: max_samples, :, :, :]
    image = tf.concat(axis=0, values=[image[i, :, : ,:]for i in range(max_samples)])
    image = sess.run(image)

    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(train_dir, filename)
    scipy.misc.toimage(image, cmin=0, cmax=1.).save(filename)
    print("  Saved %s" % (filename,))





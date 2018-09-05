const tf = require('@tensorflow/tfjs');

require('@tensorflow/tfjs-node');

base_image = tf.fromPixels('../image/test.png');
style_image = tf.fromPixels('../image/style.png');

// Need to understand how to create a placeholder for an image size that we don't know.
combination_image = tf.fromPixels();

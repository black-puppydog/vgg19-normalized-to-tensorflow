# Convert VGG19 to tensorflow

This converts the normalized weights for VGG19 (provided by [Bethge et. al.](http://bethgelab.org/deepneuralart/)) to a tensorflow checkpoint.
The `.caffemodel` file can directly be downloaded [here](http://bethgelab.org/media/uploads/deeptextures/vgg_normalised.caffemodel)

Edit `extract_caffe_features.py` to the appropriate input and output paths and run it.

Next, edit `convert_to_tf_checkpoint.py` and run it.
This leaves you with a regular tensorflow checkpoint which you can load as usual:

    with slim.arg_scope(vgg.vgg_arg_scope()):
      _, end_points = vgg.vgg_19(image_rgb_centered,
                                 is_training=False,
                                 spatial_squeeze=False)
    vgg_variable_names = slim.get_model_variables('vgg_19')
    init_fn = slim.assign_from_checkpoint_fn(model_path, vgg_variable_names)
    
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      init_fn(sess)

      [...]
    

Yes, these two scripts coulp probably be done in one, but with my setup, having both caffe and tensorflow installed gave me a headache, so f**k it. :P


## Preprocessing

The resulting network expects **centered** data in the range **[0, 255]** and **RGB** channel order.
So this should be enough, really:
    
    image_rgb_centered = scipy.misc.imread('cat.jpg', mode='RGB') - [123.68, 116.78, 103.94]
    image_rgb_centered_batch = image_rgb_centered[np.newaxis]

Ther produced outputs are equivalent to the caffe outputs, i.e. the activations after the appropriate transpositions match quite well.
See the output of `convert_to_tf_checkpoint.py` for how well.

If you want to inspect the errors in more detail, set `SAVE_OUTPUTS = True` and inspect the resulting `.npz` files.

# Why? (aka "Why not use caffe-tensorflow or such?")

There are a few scripts going around for converting any Caffe model into a tensorflow graph, but I find that for a conversion task as simple as this one, a hand-written script is easy to understand, instructional and leaves little room for misinterpretations. 

# Credits

The original VGG19 model and weights were published by [Simonyan and Zisserman](http://www.robots.ox.ac.uk/%7Evgg/research/very_deep/) under the [CC-BY](https://creativecommons.org/licenses/by/4.0/) license.

Cat image CC0 by InstaWalli: https://www.pexels.com/photo/brown-tabby-cat-167773/
Penguin image CC0 by MemoryCatcher: https://pixabay.com/en/penguins-emperor-antarctic-life-429128/

## License

These scripts are released under the Apache License, Version 2.0.


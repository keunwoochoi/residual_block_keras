# Residual block in Keras

### NOTICE
The example.py doesn't seem to work properly, but it used to be working for other applications I tried. I'll update after investigating what's going on.

### The original articles
 * [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) (the 2015 ImageNet competition winner)
 * [Identity Mappings in Deep Residual Networks](http://arxiv.org/abs/1603.05027) (an update).

### Keras
This implementation is based on [Keras](https://github.com/fchollet/keras).

### MNIST example
You can easily run an example code in [`example.py`](https://github.com/keunwoochoi/residual_block_keras/blob/master/example.py). It load MNIST dataset and
 * zeropadding `(2,2)` to convert the size `(28,28)` to `(32,32)`
 * add residual blocks
 * add average pooling at the end (aka fully convolutional, though it is not 100% - the output of average pooling is `(n_feature, 1, 1)`, then the final output layer is also a fully connected layer. 

However there are some changes on the block, which you can easily detect. Such as...
 * I don't use strided convnet to subsample but added `MaxPooling2D` on the main path. I thought it may cause some trobles if `stride` > `kernel size` because then it ends up ignoring some pixels.
 * Same problem on the shortcut path. I used `AveragePooling2D` instead of strided `1x1` convolution as in the article because i) I overlooked the part of the article and ii) wasn't sure if it would work (Definitely it worked in their experiments..). I also tried to use `MaxPoolinng2D` but seemed like it's not trained well. 
 * I put `1x1` convolution, as one of the things tested on the [original article](http://arxiv.org/abs/1512.03385) to extend the `num_feature_map`. There are simpler solution in the paper - which is just pad zeros, and worked well, FYI.
 * The block is updated according to the authors' new paper, [Identity Mappings in Deep Residual Networks](http://arxiv.org/abs/1603.05027).

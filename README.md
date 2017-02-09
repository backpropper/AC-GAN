# AC-GAN (Auxiliary Classifier Generative Adversarial Network)

Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.

Run as:
=======
th main.lua

To train on gpu (using cuDNN library-recommended):
th main.lua --gpu=1
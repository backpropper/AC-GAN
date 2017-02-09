# AC-GAN

A basic implementation of Auxiliary Classifier Generative Adversarial Network (ACGAN) in Torch7 on the
MNIST dataset by [Odena et al.](https://arxiv.org/abs/1610.09585).

##Basic Requirements:
[Torch](http://torch.ch/docs/getting-started.html#_)
[CuDNN](https://developer.nvidia.com/cudnn) (not required but recommended)

## Run as:

`th main.lua`

To train on gpu:

`th main.lua --gpu=1`

Smaple images will be stored in `samples` directory. Both the generator and the discriminator models are 
saved in the `checkpoints` folder after each epoch.
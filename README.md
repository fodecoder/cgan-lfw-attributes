# Conditional GAN on LFWCrop Face Attributes

A conditional GAN in PyTorch that generates 32×32 face images conditioned on 36 continuous facial attributes from the LFW attribute set (e.g. *Male*, *Smiling*, *Eyeglasses*, *Blond Hair*). By changing a single attribute value while keeping the noise fixed, the generator produces faces that vary along that attribute.

University group project (2020).

## Architecture

- **Generator**: latent noise (100-d) → transposed convolutions; the attribute vector is concatenated with the intermediate feature map, then further transposed convolutions with batch normalization produce the image.
- **Discriminator**: convolutional feature extractor; the attribute vector is concatenated before the fully connected classifier.
- Loss: binary cross-entropy; SGD with momentum.

## Run

Designed for Google Colab with a GPU. The first cells download the LFWCrop dataset loader.

## Credits

- Developed together with [@Antonio210696](https://github.com/Antonio210696), who also prepared the [LFWCrop dataset loader](https://github.com/Antonio210696/LFWCrop_dataset_pytorch).
- The `ListModule` / maxout helper classes are adapted from [Duncanswilson/maxout-pytorch](https://github.com/Duncanswilson/maxout-pytorch).

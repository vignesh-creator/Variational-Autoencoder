This folder consists of Implementation of Conditional Variational Autoencoder.

In regular Variational Autoencoder,the decoder gives a valid output but it doesn't know what it's generating.Inorder to make a decoder that understands what it's printing, we add condition to it.

We basically add labels to the image and will pass it through the autoencoder.

One-hot encoding is done to the labels of the image and will be concatenated to the image and network is trained using these new images.

A brief insight about CVAE is given [here](https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/)

After training:

Loss curve 

![](https://i.imgur.com/CMfPKRA.png)

Original MNIST image:
![](https://i.imgur.com/IsywPmC.png)

Reconstructed image:
![](https://i.imgur.com/lETZLW3.png)


Randomly Generated image:

![](https://i.imgur.com/6j12dHj.png)



Conditional Generative Adversial Network Demo
================

### Generator

Network that build target objects (MNIST images) from 2 components: - Noise vector
- The labels defining the object condition (which digit to produce)

![](www/Generator.png)

### Discriminator

![](www/Discriminator.png)

### Training process

Starting from noise

![](www/CGAN_iter_1.png)

Slowly getting it

![](www/CGAN_iter_200.png)

Generate specified digit images on demand

![](www/CGAN_iter_2400.png)

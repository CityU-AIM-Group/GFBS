# GFBS: Exploring Gradient Flow Based Saliency for DNN Model Compression [ACM MM'21]

> [**Exploring Gradient Flow Based Saliency for DNN Model Compression**](https://dl.acm.org/doi/10.1145/3474085.3475474)<br>
> [Xinyu Liu](https://xinyuliu-jeffrey.github.io/), Baopu Li, [Zhen Chen](https://franciszchen.github.io/), [Yixuan Yuan](http://www.ee.cuhk.edu.hk/~yxyuan/)<br>The Chinese Univerisity of Hong Kong, Oracle Cloud Infrastructure (OCI), Centre for Artificial Intelligence and Robotics (CAIR)

We propose **GFBS**, which is a structured pruning method for deep convolutional neural networks. It analyzes the channel's influence based on Taylor expansion and integrates the effects of BN layer and ReLU activation function. The channel importance can be evaluated with a single batch forward and backpropagation.

## Get Started

### Install requirements

Run the following command to install the dependences:

```bash
pip install -r requirements.txt
```

### Data preparation

For CIFAR-10, the data will be downloaded automatically when pruning.

For ImageNet, We need to prepare the dataset from [`http://www.image-net.org/`](http://www.image-net.org/).

- ImageNet-1k

ImageNet-1k contains 1.28 M images for training and 50 K images for validation.
The images shall be stored as individual files:

```
ImageNet/
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
...
├── val
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
...
```

### Training Baseline Models

Before pruning, we need to prepare the baseline unpruned models.

For CIFAR-10:

Run the following command to train a VGG-16BN model for 160 epochs:
```bash
python train.py --net gatevgg16
```

For ImageNet:

We use the [torchvision ResNet-50 model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) instead of training it from scratch. Run the following command to download and convert it for pruning:
```bash
python prepare_torch_imagenet_models.py --data_dir <path-to-imagenet>
```

### Pruning with GFBS

For CIFAR-10:
```bash
python gfbs_cifar.py --net gatevgg16 --p <channel-pruning-ratio>
```
this will fintune the pruned model for 160 epochs. Users can also add --smooth to finetune for 30 epochs after pruning each layer. Users can define the desired channel pruning ratio.

For ImageNet:
```bash
python gfbs_imagenet.py --net resnet50 --data_dir <path-to-imagenet> --p <channel-pruning-ratio> --gpu 0,1,2,3,4,5,6,7
```
this will fintune the pruned model for 120 epochs. Users can define the desired channel pruning ratio and number of GPUs to use.


## Citation

If you find our project is helpful, please feel free to leave a star and cite our paper:
```BibTeX
@inproceedings{liu2021exploring,
  title={Exploring gradient flow based saliency for dnn model compression},
  author={Liu, Xinyu and Li, Baopu and Chen, Zhen and Yuan, Yixuan},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={3238--3246},
  year={2021}
}
```

## License

- [License](./LICENSE)

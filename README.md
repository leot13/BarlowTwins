# BarlowTwins
![Model Architecture](https://user-images.githubusercontent.com/14848164/120419539-b0fab900-c330-11eb-8536-126ce6ce7b85.png)

This project is a Pytorch implementation from scratch of the paper [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/pdf/2103.03230.pdf)

```
@article{zbontar2021barlow,
  title={Barlow Twins: Self-Supervised Learning via Redundancy Reduction},
  author={Zbontar, Jure and Jing, Li and Misra, Ishan and LeCun, Yann and Deny, St{\'e}phane},
  journal={arXiv preprint arXiv:2103.03230},
  year={2021}
}
```
## Training

This model was trained with the CIFAR training set during 170 epochs . It was then evaluated on the CIFAR validation set by a linear layer trained on top of a frozen Barlow Twins model.

* Training was done on Tesla P100 and takes around 1:40 min per epoch

Work is still in progress, here are the current results:

Epochs | Batch Size | Acc top1 |
--- | --- | --- |
170 | 256 | 66.32 | 

# Are Sparse Neural Networks Better Hard Sample Learners?


Code for [Are Sparse Neural Networks Better Hard Sample Learners?](https://www.arxiv.org/abs/2409.09196) [BMVC 2024].

## Abstract

While deep learning has demonstrated impressive progress, it remains a daunting challenge to learn from hard samples as these samples are usually noisy and intricate. These hard samples play a crucial role in the optimal performance of deep neural networks. Most research on Sparse Neural Networks (SNNs) has focused on standard training data, leaving gaps in understanding their effectiveness on complex and challenging data. This paper's extensive investigation across scenarios reveals that most SNNs trained on challenging samples can often match or surpass dense models in accuracy at certain sparsity levels, especially with limited data. We observe that layer-wise density ratios tend to play an important role in SNN performance, particularly for methods that train from scratch without pre-trained initialization.

## Requirements
- PyTorch 1.12.1
- torchvision 0.13.1
- advertorch 0.2.4
- numpy
- wanda

## Training and Evaluation


#### Adversarial Training: 

##### CIFAR10:
```
python -u main_ad_att.py \
    --gpu 0 \
    --norm linf \
    --update_frequency 4 \
    --p_data 0.5 \
    --sparse True \
    --fix False \
    --data cifar10 \
    --model vgg16 \
    --method SET \
    --density 0.2 \
    --seed 15

```

##### CIFAR100:
```
python -u main_ad_att.py \
    --gpu 1 \
    --norm linf \
    --update_frequency 4 \
    --p_data 0.5 \
    --sparse True \
    --fix False \
    --data cifar100 \
    --model ResNet18 \
    --method SET \
    --density 0.2 \
    --seed 15

```

## Acknowledgements
We appreciate the following github repos a lot for their valuable code.

- https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization
- https://github.com/VITA-Group/Sparsity-Win-Robust-Generalization

## Citation

```
@article{xiao2024sparse,
  title={Are Sparse Neural Networks Better Hard Sample Learners?},
  author={Xiao, Qiao and Wu, Boqian and Yin, Lu and Gadzinski, Christopher Neil and Huang, Tianjin and Pechenizkiy, Mykola and Mocanu, Decebal Constantin},
  journal={arXiv preprint arXiv:2409.09196},
  year={2024}
}
```






  
​        
​    

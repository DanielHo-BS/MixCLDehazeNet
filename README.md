# MixCLDehazeNet

## Introduction
   - Background
   - Objective: Multi-modal Image Dehazing

## Methodology
   - A. Weakly supervised zero-shot classification using CLIP
      1. Define two domains: indoor and outdoor
      2. Utilize CLIP's pre-trained text_encoder for classification

   - B. Generate text features
      1. Use pre-trained text_encoder from CLIP

   - C. Train dehazeNet_encoder
      1. Train a custom dehazeNet_encoder for generating image features

   - D. Multi-modal feature fusion
      1. Embed text features and image features
      2. Fuse as input features

   - E. Perform multi-modal image dehazing task
      1. Use integrated features for image dehazing

## Experiments and Results
   - A. Experimental design
   - B. Performance evaluation
   - C. Result analysis

## Conclusion
   - A. Summary
   - B. Discussion of method advantages
   - C. Future outlook


## Reference

### MixDehazeNet : Mix Structure Block For Image Dehazing Network

[[Paper]](https://doi.org/10.48550/arXiv.2305.17654)
[[GitHub]](https://github.com/AmeryXiong/MixDehazeNet)
[[Ranked]](https://paperswithcode.com/sota/image-dehazing-on-reside-6k?p=mixdehazenet-mix-structure-block-for-image)

>**Abstract:**
Image dehazing is a typical task in the low-level vision field. Previous studies verified the effectiveness of the large convolutional kernel and attention mechanism in dehazing. However, there are two drawbacks: the multi-scale properties of an image are readily ignored when a large convolutional kernel is introduced, and the standard series connection of an attention module does not sufficiently consider an uneven hazy distribution. In this paper, we propose a novel framework named Mix Structure Image Dehazing Network (MixDehazeNet), which solves two issues mentioned above. Specifically, it mainly consists of two parts: the multi-scale parallel large convolution kernel module and the enhanced parallel attention module. Compared with a single large kernel, parallel large kernels with multi-scale are more capable of taking partial texture into account during the dehazing phase. In addition, an enhanced parallel attention module is developed, in which parallel connections of attention perform better at dehazing uneven hazy distribution. Extensive experiments on three benchmarks demonstrate the effectiveness of our proposed methods. For example, compared with the previous state-of-the-art methods, MixDehazeNet achieves a significant improvement (42.62dB PSNR) on the SOTS indoor dataset.

>**Framework**:
![image](https://github.com/AmeryXiong/MixDehazeNet/assets/102467128/885f69da-ab72-4c9c-8223-1b7425e98d3a)


### CLIP: Contrastive Language-Image Pre-Training

[[Paper]](https://arxiv.org/abs/2103.00020)
[[Github]](https://github.com/openai/CLIP/tree/main)


>**Abstract:**
CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.

>**Framework**:
![image](https://github.com/openai/CLIP/blob/main/CLIP.png?raw=true)

## Citation
Useful paper for my research:

```bibtex
@article{Mixdehazenet,
  title={MixDehazeNet: Mix Structure Block For Image Dehazing Network},
  author={Lu, LiPing and Xiong, Qian and Chu, DuanFeng and Xu, BingRong},
  journal={arXiv preprint arXiv:2305.17654},
  year={2023}
}
```

```bibtex
@misc{radford2021learning,
      title={Learning Transferable Visual Models From Natural Language Supervision}, 
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

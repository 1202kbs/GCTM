# Generalized Consistency Trajectory Models

Official PyTorch implementation of [Generalized Consistency Trajectory Models for Image Manipulation](https://arxiv.org/abs/2403.12510) by [Beomsu Kim](https://scholar.google.co.kr/citations?user=TofIFUgAAAAJ&hl=en)\*, Jaemin Kim\*, [Jeongsol Kim](https://scholar.google.com/citations?user=ZaVNwcQAAAAJ&hl=en), and [Jong Chul Ye](https://scholar.google.com/citations?user=HNMjoNEAAAAJ&hl=en) (\*Equal contribution).

Diffusion models suffer from two limitations.
- They require large number of function evaluations (NFEs) to generate high-fidelity images.
- They only enable noise-to-image generation.

We propose the **Generalized Consistency Trajectory Model (GCTM)**, which learns the probability flow ODE (PFODE) between arbitrary distributions via Flow Matching theory. Thus, GCTMs are capable of
- Noise-to-image and image-to-image translation,
- Score or velocity evaluation with NFE = 1,
- Traversal between arbitrary points of the PFODE with NFE = 1.

<p align="center">
  <img src="https://github.com/1202kbs/GCTM/blob/main/assets/figure1.PNG"  width="70%" height="70%" />
</p>

Consequently, GCTMs are applicable to a wide variety of tasks, such as but not limited to
- Unconditional generation
- Image-to-image translation
- Zero-shot and supervised image restoration
- Image editing
- Latent manipulation

### Unconditional Generation

<p align="center">
  <img src="https://github.com/1202kbs/GCTM/blob/main/assets/figure2.PNG"  width="70%" height="70%" />
</p>

### Image-to-Image Translation

<p align="center">
  <img src="https://github.com/1202kbs/GCTM/blob/main/assets/figure3.PNG"  width="70%" height="70%" />
</p>

### Zero-shot and Supervised Image Restoration

<p align="center">
  <img src="https://github.com/1202kbs/GCTM/blob/main/assets/figure4.PNG"  width="70%" height="70%" />
</p>

<p align="center">
  <img src="https://github.com/1202kbs/GCTM/blob/main/assets/figure5.PNG"  width="70%" height="70%" />
</p>

### Image Editing

<p align="center">
  <img src="https://github.com/1202kbs/GCTM/blob/main/assets/edit.png"  width="90%" height="90%" />
</p>

### Latent Manipulation

<p align="center">
  <img src="https://github.com/1202kbs/GCTM/blob/main/assets/figure7.PNG"  width="90%" height="90%" />
</p>

## Environment

- CUDA version 12.0
- NVCC version 11.5.119
- Python version 3.11.5
- PyTorch version 2.0.1+cu118
- Torchvision version 0.15.2+cu118
- Torchaudio version 2.0.2+cu118

## Datasets

- CIFAR10 : https://www.cs.toronto.edu/~kriz/cifar.html
- FFHQ : https://github.com/NVlabs/ffhq-dataset
- Image-to-Image : https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/

## Training 

Use ```train_gctm.py``` to train unconditional and image-to-image models, and use ```train_gctm_inverse.py``` to train supervised image restoration models. To train unconditional or image-to-image models, one first needs to create a ```FID_stats``` directory and save the Inception activation statistics in the format ```(dataset name)_(resolution).npz```. Inception activation statistics can be computed using ```save_fid_stats``` function in ```./pytorch_fid/fid_score.py```.

Example training scripts are provided in the ```./configs``` directory. For instance, to train a CIFAR10 unconditional model with independent coupling, one may use the command

```
sh ./configs/unconditional/cifar10.sh
```

## References

If you find this paper useful for your research, please consider citing
```bib
@article{
  kim2024gctm,
  title={Generalized Consistency Trajectory Models for Image Manipulation},
  author={Beomsu Kim and Jaemin Kim and Jeongsol Kim and Jong Chul Ye},
  journal={arXiv preprint},
  year={2024}
}
```

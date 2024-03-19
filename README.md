# Generalized Consistency Trajectory Models

Official PyTorch implementation of Generalized Consistency Trajectory Models for Image Manipulation by [Beomsu Kim](https://scholar.google.co.kr/citations?user=TofIFUgAAAAJ&hl=en)\*, Jaemin Kim\*, [Jeongsol Kim](https://scholar.google.com/citations?user=ZaVNwcQAAAAJ&hl=en), and [Jong Chul Ye](https://scholar.google.com/citations?user=HNMjoNEAAAAJ&hl=en) (\*Equal contribution).

Diffusion models suffer from two limitations.
- They require large number of function evaluations (NFEs) to generate high-fidelity images.
- They only enable noise-to-image generation.

We propose the **Generalized Consistency Trajectory Model (GCTM)**, which learns the probability flow ODE (PFODE) between arbitrary distributions via Flow Matching theory. Thus, GCTMs are capable of
- Noise-to-image and image-to-image translation,
- Score or velocity evluation with NFE = 1,
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

TBA

## Dataset Download

TBA

## Training 

TBA

## Test & Evaluation

TBA

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

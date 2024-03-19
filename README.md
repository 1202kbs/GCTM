# Generalized Consistency Trajectory Models

Official PyTorch implementation of Generalized Consistency Trajectory Models for Image Manipulation by [Beomsu Kim](https://scholar.google.co.kr/citations?user=TofIFUgAAAAJ&hl=en)\*, Jaemin Kim\*, [Jeongsol Kim](https://scholar.google.com/citations?user=ZaVNwcQAAAAJ&hl=en), and [Jong Chul Ye](https://scholar.google.com/citations?user=HNMjoNEAAAAJ&hl=en) (\*Equal contribution).

<p align="center">
  <img src="https://github.com/cyclomon/UNSB/blob/main/assets/gif.gif" />
</p>

<p align="center">
  <img src="https://github.com/cyclomon/UNSB/blob/main/assets/main2.jpg" />
</p>

Diffusion models suffer from two limitations.
- They require large number of function evaluations (NFEs) to generate high-fidelity images.
- They only enable noise-to-image generation.

We propose the **Generalized Consistency Trajectory Model (GCTM)**, which learns the probability flow ODE (PFODE) between arbitrary distributions via Flow Matching theory. Thus, GCTMs are capable of
- Noise-to-image and image-to-image translation,
- Score or velocity evluation with NFE = 1,
- Traversal between arbitrary points of the PFODE with NFE = 1.

<p align="center">
  <img src="https://github.com/cyclomon/UNSB/blob/main/assets/main_result_2.jpg" />
</p>

Quantitatively, out method out-performed all one-step baseline methods based on GANs.

<p align="center">
  <img src="https://github.com/cyclomon/UNSB/blob/main/assets/table.png" width="80%" height="80%" />
</p>

The superior performance of UNSB can be attributed to the fact that UNSB generates images in multiple stages. Indeed, we observe in the graph below that sample quality improves with more NFEs.

<p align="center">
  <img src="https://github.com/cyclomon/UNSB/blob/main/assets/NFE_FID.png" width="40%" height="40%" />
</p>

However, occasionally, too much NFEs led to "over-translation", where the target domain style is excessively applied to the source image. A failure case is shown below. This may be the reason behind increasing FID for some datasets at NFEs 4 or 5.

<p align="center">
  <img src="https://github.com/cyclomon/UNSB/blob/main/assets/Main_failure.png" width="40%" height="40%" />
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

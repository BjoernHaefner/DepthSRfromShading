# DepthSRfromShading
This code implements the approach for the following [research paper](https://vision.in.tum.de/members/haefner):

> **Fight ill-posedness with ill-posedness: Single-shot variational depth super-resolution from shading**  
> *B. Haefner, Y. Quéau, T. Möllenhoff, D. Cremers; Computer Vision and Pattern Recognition (CVPR), 2018.*  
> *Spotlight Presentation*  
![alt tag](https://vision.in.tum.de/_media/spezial/bib/haefner2018cvpr.png)

We put forward a principled variational approach for up-sampling a single depth map to the resolution of the companion color image provided by an RGB-D sensor. We combine heterogeneous depth and color data in order to jointly solve the ill-posed depth super-resolution and shape-from-shading problems. The low-frequency geometric information necessary to disambiguate shape-from-shading is extracted from the low-resolution depth measurements and, symmetrically, the high-resolution photometric clues in the RGB image provide the high-frequency information required to disambiguate depth super-resolution.

## 1. Requirements

This code has four third party dependencies:

0) MATLAB (Code was tested and works under MATLAB R2017b)

1) [minFunc](http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html) (mandatory)

2) [MumfordShahCuda](https://github.com/BjoernHaefner/MumfordShahCuda) (mandatory)

3) [inpaint_nans](https://de.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/4551/versions/2/download/zip) (mandatory)

4) [CMG](http://www.cs.cmu.edu/%7Ejkoutis/cmg.html) (optional, but highly recommended)

Clone these four repositories into the specified folders in the `third_party` directory of this repo and build them.



## 2. Input

- One super-resolution RGB image `I`.
- One companion low-resolution depth image `z0`.
- A binary `mask` describing the object of interest in the RGB image.
- The 3x3 intrinsic parameter matrix `K` of the RGB image.

#### 2.1 Notes

- **Reflectance assumptions**  
This code assumes that the reflectance of the object of interest is piecewise-constant.
- **Depth assumptions**  
This code assumes that the depth has no large jumps.
- **RGB and depth assumptions**  
This code assumes, that the RGB image and the depth image are aligned, but the scaling between the RGB and depth image along the rows and columns can be arbitrary e.g., `size(I)=1280x1024x3` and `size(z0)=640x480` works perfectly fine as long as the intrinsic parameters of the depth camera are provided, s. below and compare with the provided datasets.
- **RGB and depth not aligned?**  
If the RGB image `I` and the depth image `z0` are registered (same pose), but not aligned (differen intrinsic matrices), we provide an alignment tool. The MATLAB function `warpDepth2RGB.m` warps the depth onto the RGB image and returns the warped depth, given the RGB image `I`, the depth image `z0` and the corresponding intrinsic camera matrices `K_rgb` and `K_depth`.




## 3 Parameters
```
params.gamma 
	Is the parameter to turn the SfS term on and off, thus it should be 1 (recommended) or 0.
	Default: 1
	
params.mu
	Is the parameter that penalizes the depth data weight i.e., mu*||Dz - z0||_2^2.
	High values of mu force the output to be close to the input z0.
	Recommended range: [0.01,1]
	Default: 0.1.
	
params.nu
	Is the parameter that penalizes the minimal surface prior i.e., nu*||d(grad(z))||_1.
	With larger nu, this term favors regularization of the output depth
	Recommended range: nu<10
  Default: 1e-4
	
params.lambda
	Is the parameter that penalizes the piecewise constant reflectance assumption i.e., lambda*||grad(rho)||_0.
	Large lambda induce less segmentation in the reflectance.
	Recommended range: [0.1,10]
	Default: 1.

params.tol
	Is the parameter describing the tolerance of the relative residual until convergence. The residual at iteration k is calculated as norm(z^{k+1}-z^{k})/norm(z^{0}).
	Default: 1e-5.

params.tol_EL
  Is the parameter describing the tolerance between the energy and the corresponding lagrangian function.
  Default: 5e-6;

params.max_iter
	Describes the maximum number of iterations.
	Default: 100.

options.do_display
	If positive integer, then at each iteration the result is shown in figure (options.do_display).
	Range: {0,1,2,3,...}
	Default: 0.

options.verbose
	Describes the level of detail printed during the algorithm.
	0: nothing is printed
	1: high-level information is printed, such as current iteration, estimated timings and the relative residual.
	2: low-level information is printed i.e., information from the algorithms used to update each variable. Do not forget to adjust the corresponding options.{PD,BFGS,PCG}
	Range: {0,1,2}
	Default: 1

options.harmo_order
	Is the order of the spherical harmonics.
	Range: {1,2}
	Default: 1.

options.ADMM.kappa
	Is the step size for the dual update. Compare with Boyd et al. Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers
	Default: 1e-4.

options.ADMM.eta
	Varying penalty parameter.
	Default: 10.

options.ADMM.tau
	Varying penalty parameter.
	Default: 2.

options.PD <struct>
	options for the algorithm concerning the albedo update using the pieceswise constant mumford shah approach. See https://github.com/BjoernHaefner/MumfordShahCuda.
options.BFGS <struct>
	options for the algorithm concerning the nonlinear theta update update. See http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html.
options.PCG <struct>
	options for the algorithm concerning the linear z update. See https://de.mathworks.com/help/matlab/ref/pcg.html.
	Details:
	options.PCG.precond determines the preconditioning
	options.PCG.tolFun corresponds to tol in https://de.mathworks.com/help/matlab/ref/pcg.html.
	options.PCG.maxIter corresponds to maxit in https://de.mathworks.com/help/matlab/ref/pcg.html.
```
## 4. Examples
Along with this comes multiple challenging synthetic and real-world examples ([Download](http://vision.in.tum.de/~haefner/depthsrfromshading_data.zip)):

```
% synthetic data
data_name = 'data/synthetic/constant_480x640to120x160.mat';
data_name = 'data/synthetic/constant_480x640to240x320.mat';
data_name = 'data/synthetic/rectcircle_480x640to120x160.mat';
data_name = 'data/synthetic/rectcircle_480x640to240x320.mat';
data_name = 'data/synthetic/voronoi_480x640to120x160.mat';
data_name = 'data/synthetic/voronoi_480x640to240x320.mat';

% real world data
data_name = 'data/real/android1280x960to640x480.mat';
data_name = 'data/real/android1280x960to320x240.mat';
data_name = 'data/real/augustus1296x968to640x480.mat';
data_name = 'data/real/blanket1280x960to640x480.mat';
data_name = 'data/real/blanket1280x960to320x240.mat';
data_name = 'data/real/clothes1280x960to640x480.mat';
data_name = 'data/real/clothes1280x960to320x240.mat';
data_name = 'data/real/dress1280x960to640x480.mat';
data_name = 'data/real/dress1280x960to320x240.mat';
data_name = 'data/real/gate1296x968to640x480.mat';
data_name = 'data/real/lucy640x480to640x480.mat';
data_name = 'data/real/failure1280x960to640x480.mat';
data_name = 'data/real/failure1280x960to320x240.mat';
data_name = 'data/real/monkey1280x960to640x480.mat';
data_name = 'data/real/monkey1280x960to320x240.mat';
data_name = 'data/real/relief1296x968to640x480.mat';
data_name = 'data/real/wool1280x960to640x480.mat';
data_name = 'data/real/wool1280x960to320x240.mat';
```
Just uncomment the corresponding dataset in the `main.m` file to  be able to reconstruct the results from our paper.

## 5. License

DepthSRfromShading is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, see [here](http://creativecommons.org/licenses/by-nc-sa/4.0/), with an additional request:

If you make use of the library in any form in a scientific publication, please refer to `https://github.com/BjoernHaefner/DepthSRfromShading` and cite the paper

```
@inproceedings{Haefner2018CVPR,
 title = {Fight ill-posedness with ill-posedness: Single-shot variational depth super-resolution from shading},
 author =  {Haefner, B. and Quéau, Y. and Möllenhoff, T. and Cremers, D.},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 year = {2018},
 titleurl = {},
}
```

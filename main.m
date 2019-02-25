%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The implementation of the paper:                                        %
% "Fight ill-posedness with ill-posedness:                                %
% Single-shot variational depth super-resolution from shading"            %
%                                                                         %
% Bjoern Haefner, Yvain Queau, Thomas Möllenhoff, Daniel Cremers          %
%                                                                         %
% CVPR 2018 Spotlight Presentation                                        %
%                                                                         %
% The code can only be used for research purposes.                        %
%                                                                         %
% CopyRight (C) 2017 Bjoern Haefner (bjoern.haefner@in.tum.de)            %
% Computer Vision Group, TUM                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%The functional being optimized is                                        %
%min_{z,rho,l}  gamma   * || I - rho * <l,H(n)> ||_2^2                    %
%             + mu      * ||     Dz - z0        ||_2^2                    %
%             + nu      * ||     d(grad(z))     ||_1                      %
%             + lambda  * ||     grad(rho)      ||_0                      %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                        %
% DEPENDENCIES: This code depends on four publicly available libraries    %
%               1) minFunc (mandatory)                                    %
%                   http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html  %
%               2) MumfordShahCuda (mandatory)                            %
%                   https://github.com/BjoernHaefner/MumfordShahCuda      %
%               3) inpaint_nans (mandatory)                               %
%                   https://de.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/4551/versions/2/download/zip
%               4) CMG (optional, but highly recommended)                 %
%                   http://www.cs.cmu.edu/%7Ejkoutis/cmg.html             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% setup

clear
% close all
% clc

addpath('src/');
addpath(genpath('third_party/'));

% turn of warning
warning('off', 'Images:initSize:adjustingMag');

%% load data
data_name = 'data/android1280x960to640x480.mat';

load(data_name);

%% set parameters
params.gamma = 1;        % SFS weight: either 1 or 0
params.mu = 0.1;         % Depth data weight:      mu*||Dz - z0||_2^2
params.nu = 1e-4;         % Depth smoothing weight: nu*||d(grad(z))||_1
params.lambda = 1;       % Potts weight:           lambda*||grad(rho)||_0
params.tol = 1e-5;       % Stoping criterion / tolerance
params.tol_EL = 5e-6;
params.max_iter = 100;    % Maximum number of iteration

%% already have a guess for the albedo ?
% params.albedo = albedo;

%% minor options
do_save = 1;            % Save workspace as .mat-file to results directory and store depth and albedo estimate as obj file

options.do_display = 0; % Plot the results at each iteration, 0 otherwise.
% Note do_display also refers to the figure number

options.verbose = 1;    % verbose=0 means no further information is printed
% verbose=1 means to print high-level information
% verbose=2 prints some more low-level information of the algorithms used to update each variable.
% if verbose==2, don't forget to adjust verbose levels in the low-level parameters section

%% RGB and depth not aligned?
%If two intrinsic (rgb and depth) cameras are given use the following
%function to align the depth with the rgb image. Remember to also warp
%mask_lr if it is aligned with z0 before warping
% [z0, K_lr] = warpDepth2RGB(I, z0, K_rgb, K_depth);

%% low-level parameters for optimization
options.harmo_order = 1; % Order of spherical harmonics

%parameters for the overall ADMM approach
options.ADMM.kappa            = 1e-4; %kappa in our paper corresponds to the step size
options.ADMM.eta              = 10;   %corresponds to mu  in eq (3.13) in the ADMM paper of Boyd "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers"
options.ADMM.tau              = 2;    %corresponds to tau in eq (3.13) in the ADMM paper of Boyd "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers"

%parameters concering primal-dual mumford-shah model for albedo rho
options.PD.alpha              = -1; %-1 for piecewise constant albedo; >0 for piecwise smooth albedo
options.PD.maxIter            = 1000;
options.PD.tol                = 1e-5;
options.PD.gamma              = 2;
options.PD.verbose            = false;
options.PD.tau                = 0.25;
options.PD.sigma              = 0.5;

%parameters concerning bfgs algorithm for auxiliary variable theta
options.BFGS.display          = 'iter';
options.BFGS.verbose          = false;
options.BFGS.verboseI         = false;
options.BFGS.numDiff          = 0; %0 for analytic derivative
options.BFGS.DerivativeCheck	= 'off';
options.BFGS.MaxIter          = 10;
options.BFGS.optTol           = 1.0000e-09;
options.BFGS.progTol          = 1.0000e-09;

%parameters concerning pcg algorithm for depth z
options.PCG.precond           = 'cmg'; %'none', 'cmg' (recommended, CMG lib needed) or 'ichol'
options.PCG.maxIter           = 10;
options.PCG.tolFun            = 1.0000e-09;

%% run algorithm
tic
[z_est, albedo_est, l_est] = depthSRfromShading(I_noise, z0_noise, mask_sr, K_sr, params, options);
toc
%% show results
figure(1);subplot(2,2,1); imShow('rgb', I_noise, []); title('input image');
if exist('mask_lr','var') && exist('K_lr','var')
  figure(1);subplot(2,2,2); imShow('depth3d', double(z0_noise), mask_lr, K_lr); title('input depth');
end
figure(1);subplot(2,2,3); imShow('rgb', albedo_est, []); title('estimated albedo');
figure(1);subplot(2,2,4); imShow('depth3d', z_est, mask_sr, K_sr); title('super-resolution depth');
drawnow;

%% save data (optional)

if do_save
  %save whole workspace to file in results folder
  [~,name,~] = fileparts(data_name);
  %save result as obj mesh
  depth2Obj(z_est,K_sr,mask_sr,albedo_est,['results/result_',name])
  % save result as mat-file
  save(['results/result_',name,'.mat'], ...
    'I_noise', 'K_lr', 'K_sr', 'mask_lr', 'mask_sr', 'z0_noise',...
    'params', 'options',...
    'z_est', 'albedo_est', 'l_est');
end

function [ rmse_light, rmse_albedo, rmse_depth_sr, rmse_depth_lr, mae_depth_sr, mae_depth_lr] = calcErrors( ...
  l, l_est, albedo, rho_est, z_gt, z_est, z0, z0_noise, mask_sr, mask_lr, K_sr, K_lr)
%calcErrors calculates the errors for a given set of variables
%INPUT:
%       l (optional) is the ground truth lighting vector
%       l_est (optional) is the estimated lighting vector
%       albedo (optional) is the ground truth albedo of size mxnxd
%       rho_est (optional) is the estimated albedo of size mxnxd
%       z_gt is the ground truth super-resolution depth of size mxnx1
%       z_est is the estimated super-resolution depth of size mxnx1
%       z0 is the ground truth low-resolution depth of size m_lrxn_lrx1
%       z0_noise is the input low-resolution depth of size m_lrxn_lrx1
%       mask_sr is the binary super-resolution mask of size mxnx1
%       mask_lr is the binary low-resolution mask of size m_srxn_srx1
%       K_sr is the super-resolution camera intrinsics matrix of size 3x3
%       K_lr is the low-resolution camera intrinsics matrix of size 3x3
%OUTPUT:
%       rmse_light is the root mean squared error of the lighting vector
%       rmse_albedo is the root mean squared error of the albedo
%       rmse_depth_sr is the root mean squared error of the super-resolution depth
%       rmse_depth_lr is the root mean squared error of the input depth
%       mae_depth_sr is the mean angular error of the super-resolution depth
%       mae_depth_lr is the mean angular error of the low-resolution depth
%
% Copyright by
% Authors: Bjoern Haefner
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

if isempty(l) || isempty(l_est)
  rmse_light = [];
else
  factor_lighting = sum(sum(l_est.*l)) / sum(sum(l_est.^2));
  rmse_light = calcRmse(l, l_est.*factor_lighting);
  fprintf('RMSE for light: %12.9f\n', rmse_light);
end

if isempty(albedo) || isempty(rho_est)
  rmse_albedo = [];
else
  factor_albedo = sum( rho_est(mask_sr).*albedo(mask_sr) ) / sum( rho_est(mask_sr).^2 );
  rmse_albedo = calcRmse(albedo, rho_est.*factor_albedo, mask_sr);
  fprintf('RMSE for albedo: %12.9f\n', rmse_albedo);
end

rmse_depth_sr = calcRmse(z_gt, z_est, mask_sr);
fprintf('RMSE for depth sr: %12.9f\n', rmse_depth_sr);

rmse_depth_lr = calcRmse(z0, z0_noise, mask_lr);
fprintf('RMSE for depth lr: %12.9f\n', rmse_depth_lr);

N_gt    = depth2Normals(z_gt, K_sr, mask_sr);
N_est   = depth2Normals(z_est, K_sr, mask_sr);
N_gt_lr = depth2Normals(z0, K_lr, mask_lr);
N_input = depth2Normals(z0_noise, K_lr, mask_lr);

mae_depth_sr = calcMae(N_gt, N_est);
fprintf('MAE for depth sr (in degrees): %12.9f\n', mae_depth_sr);

mae_depth_lr = calcMae(N_gt_lr, N_input);
fprintf('MAE for depth lr (in degrees): %12.9f\n', mae_depth_lr);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calcMae%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mae] = calcMae(N, Nn, mask)
% by haefner
% N : original normals
% In: noisy normals
% mask which values to take into account
%calculates mean angular error on normals

if (exist('mask','var'))
  mask3d = repmat(mask(:), 1, 3);
  N  = reshape(N(mask3d) ,[],3);
  Nn = reshape(Nn(mask3d),[],3);
end

%calculate mae
mae = mean(atan2(sqrt(sum(cross(N,Nn).^2,2)),dot(N,Nn,2)))*180/pi;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calcRmse%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rmse = calcRmse(I, In, mask)
% by haefner
% I : original signal
% In: noisy signal(ie. original signal + noise signal)
% mask which values to take into account
% rmse = sqrt(sum((I(mask) - In(mask)).^2)/numel(mask))


if (~exist('mask','var'))
  mask3d = true(size(I));
else
  mask3d = repmat(mask,1,1,size(I,3));
end

% calculate rmse
rmse = sqrt(sum((I(mask3d) - In(mask3d)).^2)/sum(mask3d(:)));

end


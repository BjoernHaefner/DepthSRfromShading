function [z0_warped, K_depth_new] = warpDepth2RGB(I, z0, K_rgb, K_depth)
%warpDepth2RGB warps the depth map z0 onto the RGB image I.
%INPUT:
%        I input RGB image
%        z0 input depth map
%        K_rgb intrinsic 3x3 camera matrix of rgb image
%        K_depth intrinsic 3x3 camera matrix of depth image
%OUTPUT:
%        z0_warped warped depth map (invalid depth values are not warped,
%                  due to interpolation more values might be invalid).
%        K_depth_new new intrinsic camera parameters of warped depth.
%
%Copyright
%Author: Bjoern Haefner
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% check for correct data types

if ~isa(I, 'double')
  I = im2double(I); %I in range [0,1] now.
end

if ~isa(z0, 'double')
  z0 = double(z0); %z0 still the same range, but has double accuracy
end
%% map depth to rgb

[xx_d, yy_d] = meshgrid(0:size(z0,2)-1, 0:size(z0,1)-1);

pts_d = [ xx_d(:).';
          yy_d(:).';
          ones(1,numel(xx_d))];

K_depth_new = diag( [size(z0,2)/size(I,2) size(z0,1)/size(I,1) 1] ) * K_rgb;
pts_d_warped = reshape( transpose(K_depth_new*(K_depth\pts_d)), size(z0,1), size(z0,2), 3);

z0(z0 == 0) = NaN;
z0_warped = interp2( pts_d_warped(:,:,1), pts_d_warped(:,:,2), z0, xx_d, yy_d);
z0_warped(isnan(z0_warped)) = 0;

end


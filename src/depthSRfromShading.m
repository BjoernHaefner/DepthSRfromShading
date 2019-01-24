function [z_out, albedo_out, light_out] = depthSRfromShading(I, z0,  mask, K, params, options)
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
%The functional being optimized is
%min_{z,rho,l}  gamma   * || I - rho * <l,H(n)> ||_2^2
%             + mu      * ||     Dz - z0        ||_2^2
%             + nu      * ||     d(grad(z))     ||_1
%             + lambda  * ||     grad(rho)      ||_0
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% INPUT:
%         I           is the rgb (or grayscale) input image of size m1xn1xd
%         z0          is the low-resolution depth map of size m2xn2x1
%         mask        is the mask describing the valid pixels in I of size
%                     m1xn1x1
%         K           is the 3x3 intrinsic camera matrix of I.
%                     Note that it is assumed that the camera matrix of z0
%                     is as follows: K_z0 = diag([size(z0,2)/size(I,2)
%                     size(z0,1)/size(I,1) 1]) * K; if this is not the case
%                     have a look at the function "warpDepth2RGB"
%         params      is a struct with the following fields:
%                     -gamma is either 1 or 0 (sfs weight)
%                     -mu should be within [1e-2, 1] and is the depth prior
%                      weight
%                     -nu should be <10 and is the weight for the minimal
%                      surface prior
%                     -lambda should be in [0.1, 10] and is the weight for
%                      the piecewise constant albedo prior
%                     -tol is the tolerance ||z_{k+1}-z_{k}||_2/||z_{0}||_2
%                     -max_iter is the maximum number of iterations
%         options     is a struct with fields consisting of all options of the code and different
%                     algorithms used in here.
%                     -do_display >=0, where 0 means that nothing is
%                      plotted and any positive integer will plot the
%                      results in the corresponding figure: 
%                      figure (do_display)
%                     -verbose 0 means no further information is printed
%                              1 means to print high-level information
%                              2 prints some more low-level information
%                                of the algorithms used to update each
%                                variable. Don't forget to adjust verbose
%                                levels in the low-level parameters section
%                     -harmo_order {1,2} order of spherical harmonics
%                     Low-level parameter structs:
%                     -ADMM a struct concerning the overall ADMM algorithm:
%                       -kappa corresponds to the step size in our paper
%                       -eta corresponds to mu in eq (3.13) in the ADMM
%                        paper of Boyd "Distributed Optimization and
%                        Statistical Learning via the Alternating Direction
%                        Method of Multipliers"
%                       -tau corresponds to tau in eq (3.13) in the ADMM
%                        paper of Boyd "Distributed Optimization and
%                        Statistical Learning via the Alternating Direction
%                        Method of Multipliers"
%                     -PD a struct concerning the primal-dual iterations of
%                      the piecewise constant albedo estimation
%                     -BFGS a struct concerning the BFGS algorithm for the
%                      auxiliary variable theta
%                     -PCG struct concerning PCG algorithm for the depth z
%
% OUTPUT:
%         z_out       resulting super-resolution depth estimate
%         albedo_out  resulting albedo estimate
%         light_out   resulting lighting estimate
%
% DEPENDENCIES: This code depends on four publicly available libraries
%               1) minFunc (mandatory)
%                   http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
%               2) MumfordShahCuda (mandatory)
%                   https://github.com/BjoernHaefner/MumfordShahCuda
%               3) inpaint_nans (mandatory)
%                   https://de.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/4551/versions/2/download/zip
%               4) CMG (optional, but highly recommended)
%                   http://www.cs.cmu.edu/%7Ejkoutis/cmg.html
%
% Copyright by
% Authors: Bjoern Haefner and Yvain Queau
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
if ~isa(I, 'double')
  I = im2double(I); %I in range [0,1] now.
end

if ~isa(z0, 'double')
  z0 = double(z0); %z0 still the same range, but has double accuracy
end

if ~isa(mask, 'logical')
  mask = logical(mask); %make sure mask is really a logical matrix
end

if size(mask,3) > 1
  mask = mask(:,:,1); %make sure mask is only grayscale and thus the same for each channel of the rgb image
end

img_size = size(I);

if options.do_display
  fig = figure(options.do_display);
  subplot(2,3,1)
  imShow('rgb',I,[]);
  title(sprintf('Input intensity [%d x %d]',size(mask,2),size(mask,1)));
end

if options.verbose
  fprintf('Calculate downsampling operator D...');
end
[ D ] = getDownsampleOperator( size(I), size(z0) );
if options.verbose
  fprintf('Done\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

%make sure all depth values have the same "invalid value"
z0( z0==0 | isnan(z0) ) = NaN;

%generate mask for low-resolution image
mask_lr = vec2Img( logical(D*mask(:)), size(z0) ) & ~isnan(z0); %reshape and find subset where downsampled mask and kinect depth is valid
% mask_lr = logical(imresize(mask,size(z0),'method','nearest')) & ~isnan(z0); %reshape and find subset where downsampled mask and kinect depth is valid

% new downsampling matrix, after taking masks into account; don't forget to reweight the rows
D = D(mask_lr, mask);
D = sparse(1:size(D,1),1:size(D,1),1./sum(D,2))*D;

if options.do_display
  
  K_lr = diag([size(z0,2)/size(I,2) size(z0,1)/size(I,1) 1]) * K;
  set(0,'CurrentFigure',fig)
  subplot(2,3,2)
  imShow('depth3d', z0, mask_lr, K_lr);
  title(sprintf('Input depth [%d x %d]',size(mask_lr,2),size(mask_lr,1)));
  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing
z = z0;

%Inpainting of NaN values
if sum(isnan(z(:)))>0 %needed if values in the center are missing
  if options.verbose
    fprintf('Inpaint NaN values...');
  end
  z = inpaint_nans(z);
  if options.verbose
    fprintf('Done\n');
  end
end

% Smoothing
if options.verbose
  fprintf('Apply smoothing for initial guess...');
end
z_max = max( z(mask_lr) );
%Guided filter to smooth the input depths (mainly to get better initial guess)
z = imguidedfilter( z / z_max ) .* z_max;
if options.verbose
  fprintf('Done\n');
end

% Bicubic upsampling
z = imresize(z, img_size(1:2), 'bicubic');
z(~mask) = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% parameter normalization

mu = params.mu/( sum(mask_lr(:)) *    (mean(z0(mask_lr))/8.8314)^2 ) * size(I,3) * sum(mask(:));
nu = params.nu/( sum(mask(:))  *   abs(mean(z0(mask_lr))/8.8045)^2 ) * size(I,3) * sum(mask(:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
z  = img2Vec(z,mask);
z0 = img2Vec(z0,mask_lr);
norm_z_init = norm(z);

% Initial normals
[N, ~, zx, zy, xx, yy, Gx, Gy] = depth2Normals(z, K, mask);
theta = [zx;zy;z];
% Initial dual variables for ADMM
u = zeros(length(theta),1);

[sph_harm, nb_harmo] = normals2SphericalHarmonics(N, options.harmo_order);

% Initial guess
lighting = zeros(size(I,3), nb_harmo, size(I,4));
lighting(:, 3, :) = -1; % lighting (frontal directional)
albedo = I; %albedo is intensity

% Vectorization
I = img2Vec(I, mask);% Vectorized intensities
albedo = img2Vec(albedo, mask); % Vectorized albedo

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initiate c++/cuda for albedo estimation
ms = mumfordShahMEX('initMumfordShah', params.lambda/params.gamma, options.PD.alpha, options.PD.maxIter, options.PD.tol, options.PD.gamma, options.PD.verbose, options.PD.tau, options.PD.sigma);
mumfordShahMEX('setDataMumfordShah', ms,	single(vec2Img(I, img_size, mask)), mask);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% start algorithm

if options.verbose
  disp('Starting algorithm');
end

z_old = z;

for iter = 1:params.max_iter
  %% albedo update
  if options.verbose
    fprintf('Update albedo');
  end
  t_start = tic;
  [albedo] = albedoUpdate(sph_harm*lighting', albedo, mask, img_size, ms);
  t_albedo_cur = toc(t_start);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% lighting update
  if options.verbose
    fprintf(' - Update lighting');
  end
  t_start = tic;
  [lighting] = lightUpdate(I, albedo, sph_harm);
  t_light_cur = toc(t_start);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% auxiliary variable theta update
  if options.verbose
    fprintf(' - Update theta');
  end
  t_start = tic;
  
  [theta, dual_residual] = auxiliaryUpdate(I, lighting, albedo, z, zx, zy, u, K, params.gamma, nu, options.ADMM.kappa, xx, yy, options.BFGS, options.verbose, theta);
  
  t_theta_cur = toc(t_start);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% depth update
  if options.verbose
    fprintf(' - Update depth');
  end
  t_start = tic;
  
  [z, zx, zy] = depthUpdate(theta, u, z0, mu, options.ADMM.kappa, Gx, Gy, D, options.PCG, options.verbose, z);
  
  t_depth_cur = toc(t_start);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% update lagrange multiplier and step size
  if options.verbose
    fprintf(' - Update lagrange multiplier and step size\n');
  end
  t_start = tic;
  
  [u, options.ADMM.kappa] = lagrangeMultiplierUpdate(options.ADMM.kappa, u, z, zx, zy, theta, options.ADMM.eta, options.ADMM.tau, norm([zx;zy;z]-theta), dual_residual);
  
  % Update Normal map and spherical harmonics
  N = depth2Normals(theta(2*size(I,1)+1:3*size(I,1)), K, [], xx, yy, theta(1:size(I,1)), theta(size(I,1)+1:2*size(I,1)));
  sph_harm = normals2SphericalHarmonics(N, options.harmo_order);
  
  t_mult_cur = toc(t_start);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% calculate relative residual
  rel_res = norm( z(:) - z_old(:) ) / norm_z_init;
  diff_EL = sum(([zx;zy;z] - theta) .* u) + 0.5*options.ADMM.kappa*sum(([zx;zy;z] - theta).^2);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% Plot and Prints
  
  if options.verbose
    fprintf('[%d] albedo : %.2f, light : %.2f, theta : %.2f, depth: %.2f, lagrange multiplier: %.2f\n',...
      iter, t_albedo_cur, t_light_cur, t_theta_cur, t_depth_cur, t_mult_cur);
    fprintf('[%d] ||z-z_old||_rel : %f\n', iter, rel_res);
    fprintf('[%d] E-L : %f\n\n', iter, abs(diff_EL));
  end
  
  if options.do_display
    
    % Plot estimated albedo
    set(0,'CurrentFigure',fig);
    subplot(2,3,4)
    imShow('rgb',vec2Img(albedo, img_size, mask),[]);
    title(sprintf('Estimated albedo [%d x %d]',img_size(2),img_size(1)));
    axis off;
    
    % visualize depth
    set(0,'CurrentFigure',fig);
    subplot(2,3,5);
    imShow('depth3d',vec2Img(z, img_size(1:2), mask), mask, K);
    title(sprintf('Estimated SR depth [%d x %d]',img_size(2),img_size(1)));
    
    % Plot Normal map
    set(0,'CurrentFigure',fig);
    subplot(2,3,6)
    imShow('normals', vec2Img(N, [img_size(1:2), 3], mask));
    title(sprintf('Estimated normals [%d x %d]',img_size(2),img_size(1)))
    
    % Plot lighting
    set(0,'CurrentFigure',fig)
    subplot(2,3,3)
    imShow('lighting', lighting, 256, []);
    title('Estimated lighting');
    drawnow
    
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% Test if converged
  if((rel_res < params.tol && abs(diff_EL) < params.tol_EL) || iter == params.max_iter )
    
    z_out = vec2Img(z, img_size(1:2), mask);
    z_out(z_out==0) = NaN;
    albedo_out = vec2Img(albedo, img_size, mask);
    light_out = lighting;
    
    %close GPU
    mumfordShahMEX('closeMumfordShah', ms);
    
    if options.verbose
      disp('Done! Enjoy the result.');
      
      if rel_res < params.tol
        fprintf('Converged: %d < %d\n', rel_res, params.tol);
      else %iter > params.max_iter
        fprintf('Reached max iterations: %d\n', iter);
      end
      
    end
    
    break;
    
  end
  
  z_old = z;
  
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% albedoUpdate%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [albedo] = albedoUpdate(shading, albedo, mask, img_size, ms)

shading = single(vec2Img(shading, img_size, mask));
albedo  = single(vec2Img(albedo , img_size, mask));

albedo = mumfordShahMEX('runMumfordShah', ms,  shading);
albedo = img2Vec(double(albedo),  mask);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% lightUpdate%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lighting] = lightUpdate(I, albedo, sph_harm)

nChannels = size(I,2);

L_mat = [];
for ci = 1:nChannels
  L_mat = [L_mat; bsxfun(@times,albedo(:,ci),sph_harm)];
end

lighting = repmat(transpose(L_mat\I(:)),nChannels,1);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% auxiliaryUpdate%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [theta, dual_residual] = auxiliaryUpdate(I, lighting, albedo, z, zx, zy, u, K, gamma, nu, kappa, xx, yy, options_minfunc, verbose, theta)

theta_before = theta;

% Nonlinear theta update
if verbose==2
  theta = minFunc(@(theta)thetaUpdate(theta, z, zx, zy, u, I, lighting, albedo, kappa, xx, yy, K, gamma, nu ), theta_before, options_minfunc);
else
  evalc('theta = minFunc(@(theta)thetaUpdate(theta, z, zx, zy, u, I, lighting, albedo, kappa, xx, yy, K, gamma, nu ), theta_before, options_minfunc)');
end

% Dual residual
dual_residual = norm(-kappa * ( theta - theta_before));

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% thetaUpdate%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cost, J] = thetaUpdate(theta, z, zx, zy, u, I, lighting, albedo, kappa, xx, yy, K, gamma, nu)

theta = reshape(theta,[],3);
u = reshape(u,[],3);

[normals, dz, ~, ~, ~, ~, ~, ~, J_n, dz_p, dz_q, dz_z] = depth2Normals(theta(:,3), K, [], xx, yy, theta(:,1), theta(:,2));
[sph_harm, ~, J_sh] = normals2SphericalHarmonics(normals, sqrt(size(lighting,2))-1 , J_n);

if(nargout>1)
  
  siggnn = sign(theta(:,3).*dz);
  J = - kappa * ( [zx;zy;z] - theta(:) + u(:) )...%jacobian of lagrangian
    + nu .* [ siggnn.*theta(:,3).*dz_p;...%jacobian of minimal surface prior
              siggnn.*theta(:,3).*dz_q;...
              siggnn.*(theta(:,3).*dz_z+dz) ]./(K(1,1)*K(2,2));
end

cost = 0.5 * kappa * sum( ( [zx;zy;z] - theta(:) + u(:) ) .^ 2 ) ...%lagrangian cost
  + nu .* sum( abs(theta(:,3).*dz) )./(K(1,1)*K(2,2)); %minimal surface cost

for ch = 1:size(I,2)
  
  cost_ch = albedo(:,ch) .* (sph_harm * lighting(ch,:).') - I(:,ch);
  cost = cost + 0.5 * gamma * sum( cost_ch .^ 2 );
  
  if(nargout>1)
    %calc jacobian of photometric term
    %     DFDP = cost_ch .* albedo(:,ch) .* J_sh(:,:,1)*lighting(ch,:).';
    %     DFDQ = cost_ch .* albedo(:,ch) .* J_sh(:,:,2)*lighting(ch,:).';
    %     DFDZ = cost_ch .* albedo(:,ch) .* J_sh(:,:,3)*lighting(ch,:).';
    DFDP = bsxfun(@times, cost_ch .* albedo(:,ch), J_sh(:,:,1))*lighting(ch,:).';
    DFDQ = bsxfun(@times, cost_ch .* albedo(:,ch), J_sh(:,:,2))*lighting(ch,:).';
    DFDZ = bsxfun(@times, cost_ch .* albedo(:,ch), J_sh(:,:,3))*lighting(ch,:).';
    
    J = J + gamma * [DFDP;DFDQ;DFDZ];
  end
end

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% depthUpdate%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z, zx, zy] = depthUpdate(theta, u, z0, mu, kappa, Gx, Gy, D, options_pcg, verbose, z)

% Linear z update
mat_z = [ sqrt( 0.5 * mu    ) * D; ...% Prior part
  sqrt( 0.5 * kappa ) * Gx; ...% ADMM part
  sqrt( 0.5 * kappa ) * Gy;...
  sqrt( 0.5 * kappa ) * speye(length(z)) ];% ADMM part

sec_z = [ sqrt( 0.5 * mu    ) * z0; ...% Prior part
  sqrt( 0.5 * kappa ) * ( theta - u ) ];% ADMM part

% Normal equations
sec_z = mat_z'*sec_z;
mat_z = mat_z'*mat_z;

% Preconditioning
if(strcmp(options_pcg.precond,'none'))
  M1 = [];
  M2 = [];
elseif(strcmp(options_pcg.precond,'ichol'))
  M1 = ichol(mat_z);
  M2 = transpose(M1);
elseif(strcmp(options_pcg.precond,'cmg'))
  try
    if verbose == 2
      M1 = cmg_sdd(mat_z);
    else
      evalc('M1 = cmg_sdd(mat_z)');
    end
    M2 = [];
  catch ME %use ichol if sdd does fail, due to numerical reasons
    warning(ME.message);
    M1 = ichol(mat_z);
    M2 = transpose(M1);
  end
end

% PCG
if verbose == 2
  z = pcg(mat_z,sec_z,options_pcg.tolFun,options_pcg.maxIter,M1,M2,z);
else
  evalc('z = pcg(mat_z,sec_z,options_pcg.tolFun,options_pcg.maxIter,M1,M2,z)');
end

%update derivatives
zx = Gx*z;
zy = Gy*z;

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% lagrangeMultiplierUpdate%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [u, kappa] = lagrangeMultiplierUpdate(kappa, u, z, zx, zy, theta, eta, tau, primal_residual, dual_residual)

% Dual update
u = u + [zx;zy;z] - theta;

% Update penalty
%if(primal_residual/dual_residual > eta)
  kappa = tau*kappa;
  u = u./tau;
%elseif(dual_residual/primal_residual > eta)
%   kappa = kappa/tau;
%   u = u.*tau;
% end
end
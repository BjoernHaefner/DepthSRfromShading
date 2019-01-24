function [normals, dz, zx, zy, xx, yy, Dx, Dy, J, dz_p, dz_q, dz_z] = depth2Normals(z, K, mask, xx, yy, Dx, Dy)
%depth2Normals is a function to calculate the normal map from a depth map
%based on perspective projection
%INPUT:
%       z     is the depth map of size mxn (if mask is given) or of size mnx1
%             (if no mask is given).
%       K     is the 3x3 matrix corresponding to the intrinsic parameters
%       mask  is an mxn binary mask (only necessary, if xx,yy,Dx,Dy) are not
%             known so far. If xx,yy,Dx,Dy are known, mask can be left empty
%             i.e., mask = [].
%       xx    are the pixels in x-direction wrt the principal point K(1,3)
%       yy    are the pixels in y-direction wrt the principal point K(2,3)
%       Dx    is either a sparse matrix representing the gradient in
%             x-direction or a non-sparse vector representing the gradient of z
%             in x-direction.
%       Dy    is either a sparse matrix representing the gradient in
%             y-direction or a non-sparse vector representing the gradient of z
%             in y-direction.
%OUTPUT:
%       normals are the normals of size mnx3
%       dz      is the unnormalized norm of the normals before
%               normalization. It is strongly related to the minimal surface
%               element.
%       zx      is the derivative of z in x-direction
%       zy      is the derivative of z in y-direction
%       xx      are the pixels in x-direction wrt the principal point K(1,3)
%       yy      are the pixels in y-direction wrt the principal point K(2,3)
%       Dx      is either a sparse matrix representing the gradient in
%               x-direction.
%       Dy      is either a sparse matrix representing the gradient in
%               y-direction.
%
%OPTIONAL OUTPUT:
%       J       is the Jacobian matrix J(zx,zy,z)
%       dz_p    represents K(1,1) * n_x - xx .* n_z
%       dz_q    represents K(2,2) * n_y - yy .* n_z
%       dz_z    represents -n_z
%
%EXAMPLE: You can call this function in three different ways. If you call this function
%for the first time you should call it as described in 1., after that you
%should call it as described in 2. or 3.
%         1. z is some mxn matrix representing depth values of type double
%           [N, dz, zx, zy, xx, yy, Gx, Gy] = depth2Normals(z, K, mask);
%
%         2. z is some mnx1 vector representing depth values of type double
%            xx and yy are vectors, same as the output from 1.
%            Dx, Dy are sparse matrices, same as the output from 1.
%            [N, dz, zx, zy] = depth2Normals(z, K, [], xx, yy, Dx, Dy);
%         3. z is some mnx1 vector representing depth values of type double
%            xx and yy are vectors, same as the output from 1.
%            zx, zy are vectors vectors describing the derivative of z wrt {x,y}
%            [N, dz] = depth2Normals(z, K, [], xx, yy, zx, zy);
% Copyright by
% Author: Bjoern Haefner
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

if ~isempty(mask)
  
  if (~exist('xx','var') || ~exist('yy','var')) || (isempty(xx) || isempty(yy))
    [xx, yy] = meshgrid( 0:size(mask,2) - 1 , 0:size(mask,1) - 1 );
    xx = xx(mask) - K(1,3);
    yy = yy(mask) - K(2,3);
  end
  
  if (~exist('Dx','var') || ~exist('Dy','var')) || (isempty(Dx) || isempty(Dy))
    Dy = getGradient(mask);
    Dx = Dy(1:2:end-1,:);
    Dy = Dy(2:2:end,:);
  end
  
  if size(z,2)>1 %if z is not yet a vector, make it a vector
    z = z(mask);
  end
  
end

if size(z,2)>1 %if z is not yet a vector, make it a vector
  error('Error in depth2Normals\nz is assumed to be a vector if no mask is given');
end


if issparse(Dx) && issparse(Dy)
  zx = Dx*z;
  zy = Dy*z;
elseif size(Dx,2) == 1 && size(Dy,2) == 1%assumes that Dx and Dy are already the derivatives
  zx = Dx;
  zy = Dy;
else
  error('Error in depth2Normals\nDx and Dy should either be sparse gradient matrices or represent the derivatives of z');
end

[normals, dz] = getNormalMap(z, zx, zy, K, xx, yy);

if nargout >= 9
  [J, dz_p, dz_q, dz_z] = calcJacobian_new(normals, z, zx, zy, dz, K, xx, yy);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% getNormalMap%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [N_normalized, dz, zx, zy] = getNormalMap(z, zx, zy, K, xx, yy)
%variable explanation:
% z the depth as a vector
% zx = Dx*z\in Nx1, i.e. a vector;
% zy = Dy*z\in Nx1, i.e. a vector;
% K the intrinsic matrix
% xx and yy is the meshgrid as vector, where principal point is already
% taken into account, i.e. xx = xx - K(1,3) & yy = y - K(2,3)

%%
%get number of pixel in vector
nPix = size(z,1);

% get unnormalized normals
N_normalized = zeros(nPix,3);
N_normalized(:,1) = K(1,1) * zx;
N_normalized(:,2) = K(2,2) * zy;
N_normalized(:,3) = ( -z -xx .* zx - yy .* zy );

% get normalizing constant
dz = max(eps,sqrt( sum( N_normalized .^ 2, 2)  ));

% normalize normals
N_normalized = bsxfun(@times, N_normalized, 1./dz);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calcJacobian%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%NEW VERSION (BUGGY; TODO: FIX)%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [J, dz_p, dz_q, dz_z] = calcJacobian_new(normals, z, zx, zy, dz, K, xx, yy)

grad_dz = [ K(1,1)*normals(:,1) - xx .* normals(:,3),...%derivative of dz wrt. theta_1 (theta_1=zx)
            K(2,2)*normals(:,2) - yy .* normals(:,3),...%derivative of dz wrt. theta_2 (theta_2=zy)
                           -normals(:,3)              ];%derivative of dz wrt. theta_3 (theta_3=z)

%some helper variables
f1 = K(1,1)*ones(size(xx));
f2 = K(2,2)*ones(size(xx));
c0 = zeros(size(xx));
c1 = ones(size(xx));

J = zeros([size(normals),3]);

%nabla_p(n) (p=theta_1)
%J(:,:,1) =  ([f1 c0 -xx] - normals.*grad_dz(:,1))./dz;

%nabla_q(n) (q=theta_2)
%J(:,:,2) =  ([c0 f2 -yy] - normals.*grad_dz(:,2))./dz;

%nabla_z(n) (z=theta_3)
%J(:,:,3) =  ([c0 c0 -c1] - normals.*grad_dz(:,3))./dz;

%nabla_p(n) (p=theta_1)
J(:,:,1) =  bsxfun(@rdivide, ([f1 c0 -xx] - bsxfun(@times, normals,grad_dz(:,1))), dz);

%nabla_q(n) (q=theta_2)
J(:,:,2) =  bsxfun(@rdivide, ([c0 f2 -yy] - bsxfun(@times, normals,grad_dz(:,2))), dz);

%nabla_z(n) (z=theta_3)
J(:,:,3) =  bsxfun(@rdivide, ([c0 c0 -c1] - bsxfun(@times, normals,grad_dz(:,3))), dz);


dz_p = grad_dz(:,1);
dz_q = grad_dz(:,2);
dz_z = grad_dz(:,3);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calcJacobian%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%OLD VERSION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [J, dz_p, dz_q, dz_z] = calcJacobian_old(normals, z, zx, zy, dz, K, xx, yy)

dz_p = K(1,1) * normals(:,1) - xx .* normals(:,3);
dz_q = K(2,2) * normals(:,2) - yy .* normals(:,3);
dz_z = -normals(:,3);

dz2 = dz.^2;

J = zeros([size(normals),3]);

J(:,1,1) =  ( K(1,1) * dz - K(1,1) *zx .* dz_p ) ./ dz2;
J(:,2,1) = -( K(2,2) * zy .* dz_p ) ./ dz2;
J(:,3,1) =  ( -xx .* dz - ( -z -xx .* zx - yy .* zy ) .* dz_p ) ./ dz2;

J(:,1,2) = -( K(1,1) * zx .* dz_q ) ./ dz2;
J(:,2,2) =  ( K(2,2) * dz - K(2,2) * zy .* dz_q ) ./ dz2;
J(:,3,2) =  ( -yy .* dz - ( -z -xx .* zx - yy .* zy ) .* dz_q ) ./ dz2;

J(:,1,3) = -( K(1,1) * zx .* dz_z ) ./ dz2;
J(:,2,3) = -( K(2,2) * zy .* dz_z ) ./ dz2;
J(:,3,3) =  ( -dz - ( -z -xx .* zx - yy .* zy ) .* dz_z ) ./ dz2;

end

function [] = imShow(kind,varargin)
%imShow is a function which shows data in figures, depending on their
%representation
%INPUT:
%       kind = 'normals'
%       kind = 'depth'
%       kind = 'depth3d'
%       kind = 'rgb'
%       kind = 'lighting'
%       varargin descirbes the corresponding data, followed by arguments
%       parsed to imshow
%
%EXAMPLE: 1.  plotting normals: imShow('normals',N); where N is a mxnx3
%             matrix
%         2.  plotting depth: imShow('depth',z,[]); where z is a mxnx1
%             matrix
%         3.  plotting depth in 3D: imShow('depth3d', z, mask, K); where z
%             is a mxnx1 matrix, mask is a mxnx1 binary mask and K are the
%             intrinsic camera parameters
%         4.  plotting rgb: imShow('rgb', im2uint8(I)); where I is an
%             mxnx{1,3} image.
%         5.  plotting lighting: imShow('lighting', l, img_size); l
%             describes the light vector and img_size is a scalar
%             describing the size of the resulting square image.
%
%Copyright
%Author: Bjoern Haefner
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
if strcmp('normals', kind)
  %showNormals(N, varargin)
  showNormals(varargin{:});
elseif strcmp('depth', kind)
  imshow(varargin{:});
elseif strcmp('depth3d', kind)
  %showDepth3D(z, mask, K)
  showDepth3D(varargin{:});
elseif strcmp('rgb', kind)
  imshow(varargin{:});
elseif strcmp('lighting', kind)
  %showLighting(lighting,img_size,varargin)
  showLighting(varargin{:});
else
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% showNormals%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function showNormals(N, varargin)

N(:,:,3) = -N(:,:,3);
imshow(0.5+0.5*N,varargin{:});

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% showDepth3D%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function showDepth3D(z, mask, K)
% visualize the object shape
%
% INPUT:
% z - depth map
% mask - binary mask
%
% Author: Yvain Queau

z(mask==0 | z==0) = NaN; % Do not display outside the mask or where z is 0

[xx,yy] = meshgrid(0:size(mask,2)-1,0:size(mask,1)-1);
xx = z.*(xx-K(1,3))./K(1,1);
yy = z.*(yy-K(2,3))./K(2,2);

surfl(xx,yy,-z,[0 90]); % Plot a shaded depth map, with frontal lighting
axis equal; % Sets axis coordinates to be equal in x,y,z
axis ij; % Uses Matlab matrix coordinates instead of standard xyz
axis off; % Removes the axes
shading flat; % Introduces shading
colormap gray; % Use graylevel shading
view(0,90) % Set camera to frontal
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% showLighting%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = showLighting(lighting, img_size, varargin)

[channels, nb_harmo] = size(lighting); 

[Nx,Ny] = meshgrid(1:img_size,1:img_size);
r = 0.5*img_size;

Nx = Nx-r;
Ny = Ny-r;
Nz = -sqrt(r^2-Nx.^2-Ny.^2);

norm = sqrt(Nx.^2+Ny.^2+Nz.^2);
mask = r^2-Nx.^2-Ny.^2 > 0;

Nx = Nx./norm;
Ny = Ny./norm;
Nz = Nz./norm;

N = [Nx(mask) Ny(mask) Nz(mask)];

sh = normals2SphericalHarmonics(N, sqrt(nb_harmo)-1);

% Make image
I = vec2Img(sh * lighting',[img_size,img_size,channels],mask);
I = I./max(I(:));

imshow(I,varargin{:});
axis image

end

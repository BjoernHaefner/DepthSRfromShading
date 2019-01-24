function [spherical_harmonics, nb_harmo, J_sh] = normals2SphericalHarmonics(normals, harmo_order, J_n)
%normals2SphericalHarmonics is a function which calculates the spherical
%harmonics based on the normals and the corresponding spherical harmonics
%order.
%INPUT:
%       normals is a nx3 matrix, each column represents [nx,ny,nz]
%       harmo_order = {0, 1, 2, ...} and describes the spherical harmonics
%       order
%OUTPUT:
%       spherical_harmonics is of size nxnb_harmo
%       nb_harmo = {1, 4, 9, ...} and describes the dimension of
%       approximation of the spherical harmonics
%
%OPTIONAL OUTPUT:
%       J       is the Jacobian matrix J(zx,zy,z)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 

nb_harmo = (harmo_order+1)^2;

spherical_harmonics = zeros(size(normals,1), nb_harmo);

if ( harmo_order == 0)
  spherical_harmonics(:) = 1;
  return;
end

if ( harmo_order == 1 || harmo_order == 2)
  spherical_harmonics(:,1:3) = normals;
  spherical_harmonics(:,4) = 1;
end

if (harmo_order == 2)
  spherical_harmonics(:,5) = normals(:,1)      .* normals(:,2);
  spherical_harmonics(:,6) = normals(:,1)      .* normals(:,3);
  spherical_harmonics(:,7) = normals(:,2)      .* normals(:,3);
  spherical_harmonics(:,8) = normals(:,1) .^ 2 -  normals(:,2) .^ 2;
  spherical_harmonics(:,9) = 3 * normals(:,3) .^ 2 - 1;
end

if ( harmo_order > 2)
  error('Error in normals2SphericalHarmonics(): Unknown order of spherical harmonics %d; Not yet implemented', nb_harmo);
end

if nargin == 3 && nargout == 3
  [J_sh] = calcJacobian(spherical_harmonics, harmo_order, J_n);
end


end

function [J_sh] = calcJacobian(spherical_harmonics, harmo_order, J_n)

J_sh = zeros([size(spherical_harmonics),3]);

if ( harmo_order == 1 || harmo_order == 2)
  J_sh(:,1:3,:) = J_n;
  %   J(:,4,:) = zeros(size(J(:,4,:)));
end

if (harmo_order == 2)
  %ii=1 corresponds to zx;
  %ii=2 corresponds to zy
  %ii=3 corresponds to z
  for ii = 1:3
    J_sh(:,5,ii) =     J_n(:,1,ii) .* spherical_harmonics(:,2)	+     J_n(:,2,ii) .* spherical_harmonics(:,1);
    J_sh(:,6,ii) =     J_n(:,1,ii) .* spherical_harmonics(:,3)	+     J_n(:,3,ii) .* spherical_harmonics(:,1);
    J_sh(:,7,ii) =     J_n(:,2,ii) .* spherical_harmonics(:,3)	+     J_n(:,3,ii) .* spherical_harmonics(:,2);
    J_sh(:,8,ii) = 2 * J_n(:,1,ii) .* spherical_harmonics(:,1)	- 2 * J_n(:,2,ii) .* spherical_harmonics(:,2);
    J_sh(:,9,ii) = 6 * J_n(:,3,ii) .* spherical_harmonics(:,3);
  end
end


end


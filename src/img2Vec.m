function [vec] = img2Vec(img,mask)
%img2Vec is a function which reshapes an input image img to a vector vec.
%INPUT: img is a mxnxd matrix
%       mask is a mxnx1 or mxnxd matrix
%OUTPUT:
%       vec is a vector of size mndx1 (if no mask is given or if mask and
%           img have the same size) or mnxd (if mask is given and img and
%           mask differ in size along 3rd dimension).
%
% Copyright by
% Author: Bjoern Haefner
% Date: March 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

if exist('mask','var')
  
  if ~isa(mask, 'logical')
    mask = logical(mask);
  end
  
  if isequal(size(img), size(mask))
    vec = img(mask);
  elseif isequal(size(img,1),size(mask,1)) && isequal(size(img,2),size(mask,2)) && ~isequal(size(img,3),size(mask,3))
    %only third dimension differs
    
    img = reshape(img,[size(img,1)*size(img,2), size(img,3)]);
    vec = zeros(sum(mask(:)),size(img,2));
    for ci = 1:size(img,2)
      vec(:, ci) = img(mask, ci);
    end
  else
    error('Error in img2Vec \n rows and columns of img and mask must be equal.');
  end
  
else
  
  vec = img(:);
  
end

end


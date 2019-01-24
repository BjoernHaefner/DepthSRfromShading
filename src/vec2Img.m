function [img] = vec2Img(vec,img_size,mask)
%vec2Img transforms a vector vec to an image.
%INPUT:
%         vec is a nxd vector where n=n1*n2
%         img_size resulting size of the image img
%         mask (optional) describes a mask where the first to dimensions
%         are equal to img_size(1:2). If the third dimension differs i.e.,
%         img_size(3) ~= size(mask,3), then mask will be repeatedly
%         attached, such that the third dimension is equal.
%OUTPUT:
%         img is the resulting image of size img_size
%
%Copyright
%Author: Bjoern Haefner
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 


if exist('mask','var') && ~isempty(mask)
  
  if ~isa(mask, 'logical')
    mask = logical(mask);
  end
  
  if length(img_size) == 3 && size(mask,3) == 1
    mask = repmat(mask,1,1,img_size(3));
  end
  
  if ~isequal(img_size,size(mask))
    error('Error in img2Vec \n size of img and mask must fit.');
  end
  
  img = zeros(img_size);
  img(mask) = vec;
  
else
  img = reshape(vec,img_size);
end


end


function [] = depth2Obj(z,K,mask,texture,filename)

if ~isa(z, 'double')
  z = double(z); %z0 still the same range, but has double accuracy
end
if ~isa(texture, 'uint8')
  texture = im2uint8(texture); %z0 still the same range, but has double accuracy
end
if ~isa(mask, 'logical')
  mask = logical(mask); %z0 still the same range, but has double accuracy
end
if ~exist('filename', 'var')
  filename = 'mesh'; %z0 still the same range, but has double accuracy
end

[XYZ] = Backproject3D(z, K);
[xx, yy, Dx, Dy] = normalsSetup(mask, K);
[N] = getNormalMap(z(mask), Dx*z(mask), Dy*z(mask), K, xx, yy);
N = reproduceImage(N, find(mask), size(XYZ,1), size(XYZ,2), 3);
exportObjQuads(XYZ,N,texture,mask,filename);


end

%%
function [XYZ] = Backproject3D( Z, K )

rows = size(Z,1);
cols = size(Z,2);

% note that we need to start indexing from 0, since matlab is 1 based we'll
% add the 1 again in the end, after projecting onto the image plane
[X,Y]  = meshgrid( 0:cols-1, 0:rows-1 );

if exist('K', 'var')
  fx_from = K(1,1);
  fy_from = K(2,2);
  cx_from = K(1,3);
  cy_from = K(2,3);
else
  fx_from = 1;
  fy_from = 1;
  cx_from = 0;
  cy_from = 0;
end

X = Z .* (X - cx_from) ./fx_from;
Y = Z .* (Y - cy_from) ./fy_from;


XYZ(:,:,1) = X;
XYZ(:,:,2) = Y;
XYZ(:,:,3) = Z;

end

%%
function [xx, yy, Gx, Gy] = normalsSetup(mask, K)

%generate grid,mask and substract principal point
[xx, yy] = meshgrid( 0:size(mask,2) - 1 , 0:size(mask,1) - 1 );

xx = xx(mask) - K(1,3);
yy = yy(mask) - K(2,3);

%calculate gradient matrix of valid pixels
G = make_gradient(mask);
Gx = G(1:2:end-1,:);
Gy = G(2:2:end,:);

end

%%
function [M,imask] = make_gradient(mask)
% Function for computing the gradient operator on non-rectangular domains
% 
% [M,imask] = make_gradient(mask)
% 
% INPUT:
% mask - given binary mask
% 
% OUTPUT:
% M - Derivative matrix in x and y direction
% imask - indexes within the mask
%
% Author: Yvain Queau

	% Compute forward (Dxp and Dyp) and backward (Dxm and Dym) operators
	[Dyp,Dym,Dxp,Dxm,Sup,Sum,Svp,Svm,Omega,index_matrix,imask] = gradient_operators(mask);
	[nrows,ncols] = size(mask);

	% When there is no bottom neighbor, replace by backward (or by 0 if no top)
	Dy = Dyp;
	no_bottom = find(~Omega(:,:,1));
	no_bottom = nonzeros(index_matrix(no_bottom));
	Dy(no_bottom,:) = Dym(no_bottom,:);

	% Same for the x direction (right / left)
	Dx = Dxp;
	no_right = find(~Omega(:,:,3));
	no_right = nonzeros(index_matrix(no_right));
	Dx(no_right,:) = Dxm(no_right,:);

	M = sparse([],[],[],2*size(Dx,1),size(Dx,2),2*length(imask));
	M(1:2:end-1,:) = Dx;
	M(2:2:end,:) = Dy;
end

%%
function [Dup,Dum,Dvp,Dvm,Sup,Sum,Svp,Svm,Omega,index_matrix,imask] = gradient_operators(mask)

	[nrows,ncols] = size(mask);
	Omega_padded = padarray(mask,[1 1],0);

	% Pixels who have bottom neighbor in mask
	Omega(:,:,1) = mask.*Omega_padded(3:end,2:end-1);
	% Pixels who have top neighbor in mask
	Omega(:,:,2) = mask.*Omega_padded(1:end-2,2:end-1);
	% Pixels who have right neighbor in mask
	Omega(:,:,3) = mask.*Omega_padded(2:end-1,3:end);
	% Pixels who have left neighbor in mask
	Omega(:,:,4) = mask.*Omega_padded(2:end-1,1:end-2);
	

	imask = find(mask>0);
	index_matrix = zeros(nrows,ncols);
	index_matrix(imask) = 1:length(imask);

	% Dv matrix
	% When there is a neighbor on the right : forward differences
	idx_c = find(Omega(:,:,3)>0);
	[xc,yc] = ind2sub(size(mask),idx_c);
	indices_centre = index_matrix(idx_c);	
	indices_right = index_matrix(sub2ind(size(mask),xc,yc+1));
	indices_right = indices_right(:);
	II = indices_centre;
	JJ = indices_right;
	KK = ones(length(indices_centre),1);
	II = [II;indices_centre];
	JJ = [JJ;indices_centre];
	KK = [KK;-ones(length(indices_centre),1)];
	
	Dvp = sparse(II,JJ,KK,length(imask),length(imask));
	Svp = speye(length(imask));
	Svp = Svp(index_matrix(idx_c),:);

	% When there is a neighbor on the left : backward differences
	idx_c = find(Omega(:,:,4)>0);
	[xc,yc] = ind2sub(size(mask),idx_c);
	indices_centre = index_matrix(idx_c);	
	indices_right = index_matrix(sub2ind(size(mask),xc,yc-1));
	indices_right = indices_right(:);
	II = [indices_centre];
	JJ = [indices_right];
	KK = [-ones(length(indices_centre),1)];
	II = [II;indices_centre];
	JJ = [JJ;indices_centre];
	KK = [KK;ones(length(indices_centre),1)];
	
	Dvm = sparse(II,JJ,KK,length(imask),length(imask));
	Svm = speye(length(imask));
	Svm = Svm(index_matrix(idx_c),:);



	% Du matrix
	% When there is a neighbor on the bottom : forward differences
	idx_c = find(Omega(:,:,1)>0);
	[xc,yc] = ind2sub(size(mask),idx_c);
	indices_centre = index_matrix(idx_c);	
	indices_right = index_matrix(sub2ind(size(mask),xc+1,yc));
	indices_right = indices_right(:);
	II = indices_centre;
	JJ = indices_right;
	KK = ones(length(indices_centre),1);
	II = [II;indices_centre];
	JJ = [JJ;indices_centre];
	KK = [KK;-ones(length(indices_centre),1)];
	
	Dup = sparse(II,JJ,KK,length(imask),length(imask));
	Sup = speye(length(imask));
	Sup = Sup(index_matrix(idx_c),:);

	% When there is a neighbor on the top : backward differences
	idx_c = find(Omega(:,:,2)>0);
	[xc,yc] = ind2sub(size(mask),idx_c);
	indices_centre = index_matrix(idx_c);	
	indices_right = index_matrix(sub2ind(size(mask),xc-1,yc));
	indices_right = indices_right(:);
	II = [indices_centre];
	JJ = [indices_right];
	KK = [-ones(length(indices_centre),1)];
	II = [II;indices_centre];
	JJ = [JJ;indices_centre];
	KK = [KK;ones(length(indices_centre),1)];
	
	Dum = sparse(II,JJ,KK,length(imask),length(imask));
	Sum = speye(length(imask));
	Sum = Sum(index_matrix(idx_c),:);	

end

%%
function [N_normalized] = getNormalMap(z, zx, zy, K, xx, yy)
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
normal_norm = sqrt( sum( N_normalized .^ 2, 2)  );
                
dz = max(eps,normal_norm);

% normalize normals
N_normalized = bsxfun(@times, N_normalized, 1./dz);

end

%%
function [img] = reproduceImage(vec_in, imask, nRows, nCols, nChannels)

img = zeros(nRows, nCols, nChannels);
for ci = 1:nChannels
  img_ch = zeros(nRows, nCols, 1);
  img_ch(imask) = vec_in(:,ci);
  img(:,:,ci) = img_ch;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%	Nom ............ : 	export_obj.m
%	Version ........ : 1
%
%	Description..... : 	Save the reconstruction OBJ format, in 
%						mesh.obj, mesh.mtl and mesh.png
%						INPUT : XYZ, N, RHO -- nrows x ncols x 3
%						
%	Auteur ......... : 	Yvain Queau pour Toulouse Tech Transfer 
%
%	Date de création : 	06/10/2014
%	Date de modif... :  06/01/2014 par Yvain (moins de points, aussi 
% 						precis)
%
%	Licence ........ : 	Propriétaire
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = exportObjQuads(XYZ,N,rho,mask,filename)
	
	if (~exist('mask','var')|isempty(mask)) mask=ones(size(XYZ(:,:,1))); end;
	if (~exist('rho','var')|isempty(rho))
      rho=im2uint8(ones(size(XYZ)));
      rho(repmat(mask,1,1,3)) = NaN;
  end
	if (~exist('filename','var')|isempty(filename)) filename='mesh'; end;
	
	[nrows,ncols] = size(mask);
	
	% Switch to usual axis
	%~ XYZ(:,:,2) = - XYZ(:,:,2);
	%~ XYZ(:,:,3) = - XYZ(:,:,3);
	%~ N(:,:,2) = - N(:,:,2);
	%~ N(:,:,3) = - N(:,:,3);
	%~ 
	% Make a material structure
	material(1).type='newmtl';
	material(1).data='skin';
	material(2).type='Ka';
	material(2).data=[0.5 0.5 0.5];
	material(3).type='Kd';
	material(3).data=[1 1 1];
	material(4).type='Ks';
	material(4).data=[0.3 0.3 0.3];
	material(5).type='illum';
	material(5).data=2;
	material(6).type='Ns';
	material(6).data=10;
	
	% Nuage de Points : 
	indices_mask = find(mask>0);
	[Imask,Jmask]=ind2sub(size(mask),indices_mask);
	indices = zeros(size(mask));
	indices(indices_mask) = 1:length(indices_mask);		
	mask=[mask;zeros(1,size(mask,2))];
	mask=[mask,zeros(size(mask,1),1)];
	X = XYZ(:,:,1);
	Y = XYZ(:,:,2);
	Z = XYZ(:,:,3);
	vertices = [X(indices_mask),Y(indices_mask),Z(indices_mask)];
	clear X Y Z
	NX = N(:,:,1);
	NY = N(:,:,2);
	NZ = N(:,:,3);
	normals = [NX(indices_mask),NY(indices_mask),NZ(indices_mask)];
	clear NX NY NZ
	[X,Y] = meshgrid(1:size(XYZ,2),size(XYZ,1):-1:1);
	X = X/max(X(:));
	Y = Y/max(Y(:));
	texture = [X(indices_mask),Y(indices_mask)];
	clear X Y
	
	disp('Meshing ...')
	indices_quads = find(mask(1:end-1,1:end-1)>0 & mask(2:end,2:end) & mask(2:end,1:end-1) & mask(1:end-1,2:end)); 
	[I_lt,J_lt] = ind2sub([nrows ncols],indices_quads);	
	indices_bas = sub2ind([nrows ncols],I_lt+1,J_lt);
	indices_bas_droite = sub2ind([nrows ncols],I_lt+1,J_lt+1);
	indices_droite = sub2ind([nrows ncols],I_lt,J_lt+1);
	face_vertices = [indices(indices_quads),indices(indices_bas),indices(indices_bas_droite),indices(indices_droite)];
	face_texture = face_vertices;
	face_normals = face_vertices;
		
	% Write texture
	%~ imwrite(max(0,uint8(255*rho(:,:,:)./max(rho(:)))),'mesh.png');
	imwrite(rho,sprintf('%s.png',filename));
	material(7).type='map_Kd';
  [~,name,~] = fileparts(filename);
	material(7).data=sprintf('%s.png',name);

	% Define OBJ structure
	clear OBJ
	OBJ.vertices = vertices;
	OBJ.vertices_normal = normals;
	OBJ.vertices_texture = texture;
    OBJ.material = material;
    OBJ.objects(2).type='g';
    OBJ.objects(2).data='skin';
    OBJ.objects(3).type='usemtl';
    OBJ.objects(3).data='skin';
    OBJ.objects(1).type='f';
    OBJ.objects(1).data.vertices=face_vertices;
    OBJ.objects(1).data.texture=face_texture;
    OBJ.objects(1).data.normal=face_normals;
    
    % Write
    disp('Writing OBJ ...')
    write_wobj_quads(OBJ,sprintf('%s.obj',filename));	
    
    %~ disp('Writing PLY ...')
    %~ write_ply(vertices,face_vertices,normals,'mesh.ply','binary_little_endian');	
	
	
	

end


function write_wobj_quads(OBJ,fullfilename)
% Write objects to a Wavefront OBJ file
%
% write_wobj(OBJ,filename);
%
% OBJ struct containing:
%
% OBJ.vertices : Vertices coordinates
% OBJ.vertices_texture: Texture coordinates 
% OBJ.vertices_normal : Normal vectors
% OBJ.vertices_point  : Vertice data used for points and lines   
% OBJ.material : Parameters from external .MTL file, will contain parameters like
%           newmtl, Ka, Kd, Ks, illum, Ns, map_Ka, map_Kd, map_Ks,
%           example of an entry from the material object:
%       OBJ.material(i).type = newmtl
%       OBJ.material(i).data = 'vase_tex'
% OBJ.objects  : Cell object with all objects in the OBJ file, 
%           example of a mesh object:
%       OBJ.objects(i).type='f'               
%       OBJ.objects(i).data.vertices: [n x 4 double]
%       OBJ.objects(i).data.texture:  [n x 4 double]
%       OBJ.objects(i).data.normal:   [n x 4 double]
%
% example reading/writing,
%
%   OBJ=read_wobj('examples\example10.obj');
%   write_wobj(OBJ,'test.obj');
%
% example isosurface to obj-file,
%
%   % Load MRI scan
%   load('mri','D'); D=smooth3(squeeze(D));
%   % Make iso-surface (Mesh) of skin
%   FV=isosurface(D,1);
%   % Calculate Iso-Normals of the surface
%   N=isonormals(D,FV.vertices);
%   L=sqrt(N(:,1).^2+N(:,2).^2+N(:,3).^2)+eps;
%   N(:,1)=N(:,1)./L; N(:,2)=N(:,2)./L; N(:,3)=N(:,3)./L;
%   % Display the iso-surface
%   figure, patch(FV,'facecolor',[1 0 0],'edgecolor','none'); view(3);camlight
%   % Invert Face rotation
%   FV.faces=[FV.faces(:,3) FV.faces(:,2) FV.faces(:,1)];
%
%   % Make a material structure
%   material(1).type='newmtl';
%   material(1).data='skin';
%   material(2).type='Ka';
%   material(2).data=[0.8 0.4 0.4];
%   material(3).type='Kd';
%   material(3).data=[0.8 0.4 0.4];
%   material(4).type='Ks';
%   material(4).data=[1 1 1];
%   material(5).type='illum';
%   material(5).data=2;
%   material(6).type='Ns';
%   material(6).data=27;
%
%   % Make OBJ structure
%   clear OBJ
%   OBJ.vertices = FV.vertices;
%   OBJ.vertices_normal = N;
%   OBJ.material = material;
%   OBJ.objects(1).type='g';
%   OBJ.objects(1).data='skin';
%   OBJ.objects(2).type='usemtl';
%   OBJ.objects(2).data='skin';
%   OBJ.objects(3).type='f';
%   OBJ.objects(3).data.vertices=FV.faces;
%   OBJ.objects(3).data.normal=FV.faces;
%   write_wobj(OBJ,'skinMRI.obj');
%
% Function is written by D.Kroon University of Twente (June 2010)
% Optimization by Y. Queau (TU Munich) for faster write of large files

if(exist('fullfilename','var')==0)
    [filename, filefolder] = uiputfile('*.obj', 'Write obj-file');
    fullfilename = [filefolder filename];
end
[filefolder,filename] = fileparts( fullfilename);

comments=cell(1,4);
comments{1}=' Produced by Matlab Write Wobj exporter ';
comments{2}='';

fid = fopen(fullfilename,'Wb');
write_comment(fid,comments);

if(isfield(OBJ,'material')&&~isempty(OBJ.material))
    filename_mtl=fullfile(filefolder,[filename '.mtl']);
    fprintf(fid,'mtllib %s\n',[filename '.mtl']);
    write_MTL_file(filename_mtl,OBJ.material)
    
end

%progressbar(0);
if(isfield(OBJ,'vertices')&&~isempty(OBJ.vertices))
	disp('Writing vertices');
	%progressbar(0/5);
    fast_write_vertices(fid,OBJ.vertices,'v');
end


if(isfield(OBJ,'vertices_point')&&~isempty(OBJ.vertices_point))
    disp('Writing vertices points');
    %progressbar(1/5);
    fast_write_vertices(fid,OBJ.vertices_point,'vp');
end


if(isfield(OBJ,'vertices_normal')&&~isempty(OBJ.vertices_normal))
    disp('Writing vertices normals');
    %progressbar(2/5);
    fast_write_vertices(fid,OBJ.vertices_normal,'vn');
end


if(isfield(OBJ,'vertices_texture')&&~isempty(OBJ.vertices_texture))
    disp('Writing vertices texture');
    %progressbar(3/5);
    fast_write_vertices(fid,OBJ.vertices_texture,'vt');
end

for i=1:length(OBJ.objects)
    type=OBJ.objects(i).type;
    data=OBJ.objects(i).data;
    switch(type)
        case 'usemtl'
            fprintf(fid,'usemtl %s\n',data);
        case 'f'
			disp('Writing faces');
			%progressbar(4/5);
            check1=(isfield(OBJ,'vertices_texture')&&~isempty(OBJ.vertices_texture));
            check2=(isfield(OBJ,'vertices_normal')&&~isempty(OBJ.vertices_normal));
            
            chr = repmat('f',[size(data.vertices,1) 1]); 
			C1 = cellstr(chr); 
			C = C1.'; 
			

            if(check1&&check2)
				C2 = num2cell(data.vertices(:,1));
				C(2,:) = C2.';
				C3 = num2cell(data.texture(:,1));
				C(3,:) = C3.';
				C4 = num2cell(data.normal(:,1));
				C(4,:) = C4.';
				C5 = num2cell(data.vertices(:,2));
				C(5,:) = C5.';
				C6 = num2cell(data.texture(:,2));
				C(6,:) = C6.';
				C7 = num2cell(data.normal(:,2));
				C(7,:) = C7.';
				C8 = num2cell(data.vertices(:,3));
				C(8,:) = C8.';
				C9 = num2cell(data.texture(:,3));
				C(9,:) = C9.';
				C10 = num2cell(data.normal(:,3));
				C(10,:) = C10.';
				C11 = num2cell(data.vertices(:,4));
				C(11,:) = C11.';
				C12 = num2cell(data.texture(:,4));
				C(12,:) = C12.';
				C13 = num2cell(data.normal(:,4));
				C(13,:) = C13.';
				fprintf(fid,'%s %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d\n', C{:});	            
                %~ for j=1:size(data.vertices,1)
					%~ progressbar([],j/size(data.vertices,1));
                    %~ fprintf(fid,'f %d/%d/%d',data.vertices(j,1),data.texture(j,1),data.normal(j,1));
                    %~ fprintf(fid,' %d/%d/%d', data.vertices(j,2),data.texture(j,2),data.normal(j,2));
                    %~ fprintf(fid,' %d/%d/%d', data.vertices(j,3),data.texture(j,3),data.normal(j,3));
                    %~ fprintf(fid,' %d/%d/%d\n', data.vertices(j,4),data.texture(j,4),data.normal(j,4));
                %~ end
            elseif(check1)
				C2 = num2cell(data.vertices(:,1));
				C(2,:) = C2.';
				C3 = num2cell(data.texture(:,1));
				C(3,:) = C3.';
				C4 = num2cell(data.vertices(:,2));
				C(4,:) = C4.';
				C5 = num2cell(data.texture(:,2));
				C(5,:) = C5.';
				C6 = num2cell(data.vertices(:,3));
				C(6,:) = C6.';
				C7 = num2cell(data.texture(:,3));
				C(7,:) = C7.';
				C8 = num2cell(data.vertices(:,4));
				C(8,:) = C8.';
				C9 = num2cell(data.texture(:,4));
				C(9,:) = C9.';
				fprintf(fid,'%s %d//%d %d//%d %d//%d %d//%d\n', C{:});	            
                %~ for j=1:size(data.vertices,1)
					%~ progressbar([],j/size(data.vertices,1));
                    %~ fprintf(fid,'f %d/%d',data.vertices(j,1),data.texture(j,1));
                    %~ fprintf(fid,' %d/%d', data.vertices(j,2),data.texture(j,2));
                    %~ fprintf(fid,' %d/%d', data.vertices(j,3),data.texture(j,3));
                    %~ fprintf(fid,' %d/%d\n', data.vertices(j,4),data.texture(j,4));
                %~ end
            elseif(check2)
				C2 = num2cell(data.vertices(:,1));
				C(2,:) = C2.';
				C3 = num2cell(data.normals(:,1));
				C(3,:) = C3.';
				C4 = num2cell(data.vertices(:,2));
				C(4,:) = C4.';
				C5 = num2cell(data.normals(:,2));
				C(5,:) = C5.';
				C6 = num2cell(data.vertices(:,3));
				C(6,:) = C6.';
				C7 = num2cell(data.normals(:,3));
				C(7,:) = C7.';
				C8 = num2cell(data.vertices(:,4));
				C(8,:) = C8.';
				C9 = num2cell(data.normals(:,4));
				C(9,:) = C9.';
				fprintf(fid,'%s %d//%d %d//%d %d//%d %d//%d\n', C{:});	
                %~ for j=1:size(data.vertices,1)
					%~ progressbar([],j/size(data.vertices,1));
                    %~ fprintf(fid,'f %d//%d',data.vertices(j,1),data.normal(j,1));
                    %~ fprintf(fid,' %d//%d', data.vertices(j,2),data.normal(j,2));
                    %~ fprintf(fid,' %d//%d', data.vertices(j,3),data.normal(j,3));
                    %~ fprintf(fid,' %d//%d\n', data.vertices(j,4),data.normal(j,4));
                %~ end
            else
				C2 = num2cell(data.vertices(:,1));
				C(2,:) = C2.';
				C3 = num2cell(data.vertices(:,2));
				C(3,:) = C3.';
				C4 = num2cell(data.vertices(:,3));
				C(4,:) = C4.';
				C5 = num2cell(data.vertices(:,4));
				C(5,:) = C5.';
				fprintf(fid,'%s %d %d %d\n', C{:});	
                %~ for j=1:size(data.vertices,1)
					%~ progressbar([],j/size(data.vertices,1));
                    %~ fprintf(fid,'f %d %d %d %d\n',data.vertices(j,1),data.vertices(j,2),data.vertices(j,3),data.vertices(j,4));
                %~ end
            end
            %progressbar(1,1);
        otherwise
            fprintf(fid,'%s ',type);
            if(iscell(data))
                for j=1:length(data)
                    if(ischar(data{j}))
                        fprintf(fid,'%s ',data{j});
                    else
                        fprintf(fid,'%0.5g ',data{j});
                    end
                end 
            elseif(ischar(data))
                 fprintf(fid,'%s ',data);
            else
                for j=1:length(data)
                    fprintf(fid,'%0.5g ',data(j));
                end      
            end
            fprintf(fid,'\n');
    end
end
fclose(fid);
%progressbar(1)
end









function write_MTL_file(filename,material)
fid = fopen(filename,'Wb');
comments=cell(1,2);
comments{1}=' Produced by Matlab Write Wobj exporter ';
comments{2}='';
write_comment(fid,comments);

for i=1:length(material)
    type=material(i).type;
    data=material(i).data;
    switch(type)
        case('newmtl')
            fprintf(fid,'%s ',type);
            fprintf(fid,'%s\n',data);
        case{'Ka','Kd','Ks'}
            fprintf(fid,'%s ',type);
            fprintf(fid,'%5.5f %5.5f %5.5f\n',data);
        case('illum')
            fprintf(fid,'%s ',type);
            fprintf(fid,'%d\n',data);
        case {'Ns','Tr','d'}
            fprintf(fid,'%s ',type);
            fprintf(fid,'%5.5f\n',data);
        otherwise
            fprintf(fid,'%s ',type);
            if(iscell(data))
                for j=1:length(data)
                    if(ischar(data{j}))
                        fprintf(fid,'%s ',data{j});
                    else
                        fprintf(fid,'%0.5g ',data{j});
                    end
                end 
            elseif(ischar(data))
                fprintf(fid,'%s ',data);
            else
                for j=1:length(data)
                    fprintf(fid,'%0.5g ',data(j));
                end      
            end
            fprintf(fid,'\n');
    end
end

comments=cell(1,2);
comments{1}='';
comments{2}=' EOF';
write_comment(fid,comments);
fclose(fid);
end








function write_comment(fid,comments)
for i=1:length(comments), fprintf(fid,'# %s\n',comments{i}); end
end







function fast_write_vertices(fid,V,type)
	chr = repmat(type,[size(V,1) 1]); 
	C1 = cellstr(chr); 
	C = C1.'; 
	switch size(V,2) 
		case 1
			C2 = num2cell(V(:,1));
			C(2,:) = C2.'; 
			
			fprintf(fid,'%s %5.5f\n', C{:});		
		case 2
			C2 = num2cell(V(:,1));
			C(2,:) = C2.'; 

			C3 = num2cell(V(:,2));
			C(3,:) = C3.'; 

			fprintf(fid,'%s %5.5f %5.5f\n', C{:});		
		case 3 
			C2 = num2cell(V(:,1));
			C(2,:) = C2.'; 

			C3 = num2cell(V(:,2));
			C(3,:) = C3.'; 

			C4 = num2cell(V(:,3));
			C(4,:) = C4.'; 
			
			fprintf(fid,'%s %5.5f %5.5f %5.5f\n', C{:});				
		otherwise
	end
	switch(type)
		case 'v'
			fprintf(fid,'# %d vertices \n', size(V,1));
		case 'vt'
			fprintf(fid,'# %d texture verticies \n', size(V,1));
		case 'vn'
			fprintf(fid,'# %d normals \n', size(V,1));
		otherwise
			fprintf(fid,'# %d\n', size(V,1));			
	end
end



function write_vertices(fid,V,type)
	switch size(V,2) 
		case 1
			for i=1:size(V,1)
				%progressbar([],i/size(V,1));
				fprintf(fid,'%s %5.5f\n', type, V(i,1));
			end
		case 2
			for i=1:size(V,1)
				%progressbar([],i/size(V,1));
				fprintf(fid,'%s %5.5f %5.5f\n', type, V(i,1), V(i,2));
			end
		case 3 
			for i=1:size(V,1)
				%progressbar([],i/size(V,1));
				fprintf(fid,'%s %5.5f %5.5f %5.5f\n', type, V(i,1), V(i,2), V(i,3));
			end
		otherwise
	end
	switch(type)
		case 'v'
			fprintf(fid,'# %d vertices \n', size(V,1));
		case 'vt'
			fprintf(fid,'# %d texture verticies \n', size(V,1));
		case 'vn'
			fprintf(fid,'# %d normals \n', size(V,1));
		otherwise
			fprintf(fid,'# %d\n', size(V,1));			
	end
end

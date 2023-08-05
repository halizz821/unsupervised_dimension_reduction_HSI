clc
clear all
%% Parameter
numfeatures= 10; % number of features to be extracted by WFE. This is the total number of bands of the new image

%% load image
load('Indian_pines_corrected.mat')
im= indian_pines_corrected;

%% Scale the image

 for i=1:size(im,3)
     bn=im(:,:,i);
     
     mi=(min(min(bn)));
     ma=(max(max(bn)));
     bn=(bn-mi)/(ma-mi);
     im_vec_n(:,i)=bn(:); % Scaled image
 end

%% WFE

new_image = WFE(im_vec_n,numfeatures); % run WFE to extract 'numfeatures' number of features. These features are the new bands of image
Final_image_WFE= reshape(new_image,size(im,1),size(im,2),numfeatures); % reshape 'new_image' so that it has the same spatial dimension of the input image


%% FFE

new_image = FFE(im_vec_n,numfeatures); % run WFE to extract 'numfeatures' number of features. These features are the new bands of image
Final_image_FFE= reshape(new_image,size(im,1),size(im,2),numfeatures); % reshape 'new_image' so that it has the same spatial dimension of the input image


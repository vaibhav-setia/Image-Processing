Deblur an Image Using Blind Deconvolution
Create a sample image with noise.

% Set the random number generator back to its default settings for
% consistency in results.
rng default;

I = im2double(imread('002.png'));
PSF = fspecial('gaussian',5,5);
V = .00001;
BlurredNoisy = imnoise(imfilter(I,PSF),'gaussian',0,V);
%BlurredNoisy = im2double(imread('002.png'));
Create a weight array to specify which pixels are included in processing.

WT = zeros(size(I));
WT(5:end-4,5:end-4) = 1;
INITPSF = ones(size(PSF));
Perform blind deconvolution.

[J P] = deconvblind(BlurredNoisy,INITPSF,20,10*sqrt(V),WT);
Display the results.

subplot(221);imshow(BlurredNoisy);
title('A = Blurred and Noisy');
subplot(222);imshow(PSF,[]);
title('True PSF');
figure;imshow(J);
title('Deblurred Image');
subplot(224);imshow(P,[]);
dlmwrite('filename.txt',P)
title('Recovered PSF');
Copyright 2015 The MathWorks, Inc.

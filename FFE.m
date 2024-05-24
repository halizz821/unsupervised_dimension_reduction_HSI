function new_image = FFE(image_scaled,numfeatures)
% This the put forward unsupervised feature extraction method, called FFE
% in our paper:

% @Article{rs15153855,
% AUTHOR = {Alizadeh Moghaddam, Sayyed Hamed and Gazor, Saeed and Karami, Fahime and Amani, Meisam and Jin, Shuanggen},
% TITLE = {An Unsupervised Feature Extraction Using Endmember Extraction and Clustering Algorithms for Dimension Reduction of Hyperspectral Images},
% JOURNAL = {Remote Sensing},
% VOLUME = {15},
% YEAR = {2023},
% NUMBER = {15},
% ARTICLE-NUMBER = {3855},
% URL = {https://www.mdpi.com/2072-4292/15/15/3855},
% ISSN = {2072-4292},
% DOI = {10.3390/rs15153855}
% }

%%%%%%%%%%%%%%%%% Input Parameter
% numfeatures= The number of features to be extracted by WFE. This is the total number of bands of the new image
% image_scaled= scaled image in the shape of (total_pixel,total bands)

%%%%%%%%%%%%%%%%% Output
% new_image= new image with new extracted features. It has the dimension of (total_pixel,numfeatures)


 %% VD estimation using Hysime

noise_type = 'poisson';
verbose='on';
[w ,Rn] = estNoise(image_scaled',noise_type,verbose);
[q ,~]=hysime(image_scaled',w,Rn,verbose) ;  % q is the estimated VD

%% Endmember Extraction using VCA

U = hyperVca( image_scaled', q ); % U is the extraction endmembers in the dimension of [total_bands, total_endmember]

%% Cluster bands using Fuzzy C-Means
options = [2 1000 1e-5 0];
[~,UU,~] = fcm(U,numfeatures,options);
%% feature extraction using WFE
new_image=wmean(image_scaled,UU);

function wm=wmean(training_vec,fuzzines)
m=(training_vec*fuzzines');
a=sum(fuzzines,2)';

wm=bsxfun(@rdivide,m,a);

%% The following functions are got from "Open Source MATLAB Hyperspectral Toolbox"
% written by Isaac Gerg
% https://github.com/isaacgerg/matlabHyperspectralToolbox

function [varargout]=hysime(varargin);
%
% HySime: Hyperspectral signal subspace estimation
%
% [kf,Ek]=hysime(y,w,Rw,verbose);
%
% Input:
%        y  hyperspectral data set (each column is a pixel)
%           with (L x N), where L is the number of bands
%           and N the number of pixels
%        w  (L x N) matrix with the noise in each pixel
%        Rw noise correlation matrix (L x L)
%        verbose [optional] (on)/off
% Output
%        kf signal subspace dimension
%        Ek matrix which columns are the eigenvectors that span 
%           the signal subspace
%
%  Copyright: José Nascimento (zen@isel.pt)
%             & 
%             José Bioucas-Dias (bioucas@lx.it.pt)
%
%  For any comments contact the authors

error(nargchk(3, 4, nargin))
if nargout > 2, error('too many output parameters'); end
verbose = 1; % default value

y = varargin{1}; % 1st parameter is the data set
[L N] = size(y);
if ~numel(y),error('the data set is empty');end
n = varargin{2}; % the 2nd parameter is the noise
[Ln Nn] = size(n);
Rn = varargin{3}; % the 3rd parameter is the noise correlation matrix
[d1 d2] = size(Rn);
if nargin == 4, verbose = ~strcmp(lower(varargin{4}),'off');end

if Ln~=L | Nn~=N,  % n is an empty matrix or with different size
   error('empty noise matrix or its size does not agree with size of y\n'),
end
if (d1~=d2 | d1~=L)
   fprintf('Bad noise correlation matrix\n'),
   Rn = n*n'/N; 
end    


x = y - n;

if verbose,fprintf(1,'Computing the correlation matrices\n');end
[L N]=size(y);
Ry = y*y'/N;   % sample correlation matrix 
Rx = x*x'/N;   % signal correlation matrix estimates 
if verbose,fprintf(1,'Computing the eigen vectors of the signal correlation matrix\n');end
[E,D]=svd(Rx); % eigen values of Rx in decreasing order, equation (15)
dx = diag(D);

if verbose,fprintf(1,'Estimating the number of endmembers\n');end
Rn=Rn+sum(diag(Rx))/L/10^5*eye(L);

Py = diag(E'*Ry*E); %equation (23)
Pn = diag(E'*Rn*E); %equation (24)
cost_F = -Py + 2 * Pn; %equation (22)
kf = sum(cost_F<0);
[dummy,ind_asc] = sort( cost_F ,'ascend');
Ek = E(:,ind_asc(1:kf));
if verbose,fprintf(1,'The signal subspace dimension is: k = %d\n',kf);end

% only for plot purposes, equation (19)
Py_sort =  trace(Ry) - cumsum(Py(ind_asc));
Pn_sort = 2*cumsum(Pn(ind_asc));
cost_F_sort = Py_sort + Pn_sort;

%indice=1:50;
%figure
%   set(gca,'FontSize',12,'FontName','times new roman')
%   semilogy(indice,cost_F_sort(indice),'-',indice,Py_sort(indice),':',indice,Pn_sort(indice),'-.', 'Linewidth',2,'markersize',5)
%   xlabel('k');ylabel('mse(k)');title('HySime')
%   legend('Mean Squared Error','Projection Error','Noise Power')


varargout(1) = {kf};
if nargout == 2, varargout(2) = {Ek};end
return
%end of function [varargout]=hysime(varargin);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [varargout]=estNoise(varargin);

% estNoise : hyperspectral noise estimation.
% This function infers the noise in a 
% hyperspectral data set, by assuming that the 
% reflectance at a given band is well modelled 
% by a linear regression on the remaining bands.
%
% [w Rw]=estNoise(r)
% [w Rw]=estNoise(r,noise_type,verbose)
% Input:
%    r: is LxN matrix with the hyperspectral data set 
%       where L is the number of bands 
%       and N is the number of pixels
%    noise_type: [optional] ('additive')|'poisson'
%    verbose: [optional] ('on')|'off'
% Output
%    w is the noise estimates for every pixel (LxN)
%    Rw is the noise correlation matrix estimates (LxL)
%
%  Copyright: José Nascimento (zen@isel.pt)
%             & 
%             José Bioucas-Dias (bioucas@lx.it.pt)
%
%  For any comments contact the authors

error(nargchk(1, 3, nargin))
if nargout > 2, error('too many output parameters'); end
y = varargin{1};
if ~isnumeric(y), error('the data set must an L x N matrix'); end
noise_type = 'additive'; % default value
verbose = 1; verb ='on'; % default value
for i=2:nargin 
   switch lower(varargin{i}) 
       case {'additive'}, noise_type = 'additive';
       case {'poisson'}, noise_type = 'poisson';
       case {'on'}, verbose = 1; verb = 'on';
       case {'off'}, verbose = 0; verb = 'off';
       otherwise, error('parameter [%d] is unknown',i);
   end
end

[L N] = size(y);
if L<2, error('Too few bands to estimate the noise.'); end

if verbose, fprintf(1,'Noise estimates:\n'); end

if strcmp(noise_type,'poisson')
       sqy = sqrt(y.*(y>0));          % prevent negative values
       [u Ru] = estAdditiveNoise(sqy,verb); % noise estimates
       x = (sqy - u).^2;            % signal estimates 
       w = sqrt(x).*u*2;
       Rw = w*w'/N; 
else % additive noise
       [w Rw] = estAdditiveNoise(y,verb); % noise estimates        
end

varargout(1) = {w};
if nargout == 2, varargout(2) = {Rw}; end
return
% end of function [varargout]=estNoise(varargin);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Internal Function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w,Rw]=estAdditiveNoise(r,verbose);

small = 1e-6;
verbose = ~strcmp(lower(verbose),'off');
[L N] = size(r);
% the noise estimation algorithm
w=zeros(L,N);
if verbose, 
   fprintf(1,'computing the sample correlation matrix and its inverse\n');
end
RR=r*r';     % equation (11)
RRi=inv(RR+small*eye(L)); % equation (11)
if verbose, fprintf(1,'computing band    ');end;
for i=1:L
    if verbose, fprintf(1,'\b\b\b%3d',i);end;
    % equation (14)
    XX = RRi - (RRi(:,i)*RRi(i,:))/RRi(i,i);
    RRa = RR(:,i); RRa(i)=0; % this remove the effects of XX(:,i)
    % equation (9)
    beta = XX * RRa; beta(i)=0; % this remove the effects of XX(i,:)
    % equation (10)
    w(i,:) = r(i,:) - beta'*r; % note that beta(i)=0 => beta(i)*r(i,:)=0
end
if verbose, fprintf(1,'\ncomputing noise correlation matrix\n');end
Rw=diag(diag(w*w'/N));
return
% end of function [w,Rw]=estAdditiveNoise(r,verbose);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ U, indicies, snrEstimate ] = hyperVca( M, q )
%HYPERVCA Vertex Component Analysis algorithm
%   hyperVca performs the vertex component analysis algorithm to find pure
% pixels in an HSI scene
%
% Usage
%   [ U, indicies, snrEstimate ] = hyperVca( M, q )
% Inputs
%   M - HSI data as 2D matrix (p x N).
%   q - Number of endmembers to find.
% Outputs
%   U - Matrix of endmembers (p x q).
%   indicies - Indicies of pure pixels in U
%   snrEstimate - SNR estimate of data [dB]
%
% References
%   J. M. P. Nascimento and J. M. B. Dias, Vertex component analysis: A 
% fast algorithm to unmix hyperspectral data, IEEE Transactions on 
% Geoscience and Remote Sensing, vol. 43, no. 4, apr 2005.

%Initialization.
N = size(M, 2);
L = size(M, 1);

% Compute SNR estimate.  Units are dB.
% Equation 13.
% Prefer using mean(X, dim).  I believe this should be faster than doing
% mean(X.') since matlab doesnt have to worry about the matrix
% transposition.
rMean = mean(M, 2);
RZeroMean = M - repmat(rMean, 1, N);
% This is essentially doing PCA here since we have zero mean data.
%  RZeroMean*RZeroMean.'/N -> covariance matrix estimate.
[Ud, Sd, Vd] = svds(RZeroMean*RZeroMean.'/N, q);
Rd = Ud.'*(RZeroMean);
P_R = sum(M(:).^2)/N;
P_Rp = sum(Rd(:).^2)/N + rMean.'*rMean;
SNR = abs(10*log10( (P_Rp - (q/L)*P_R) / (P_R - P_Rp) ));
snrEstimate = SNR;

fprintf('SNR estimate [dB]: %g\n', SNR);

% Determine which projection to use.
SNRth = 15 + 10*log(q) + 8;
%SNRth = 15 + 10*log(q);
if (SNR > SNRth) 
    d = q;
    [Ud, Sd, Vd] = svds((M*M.')/N, d);
    Xd = Ud.'*M;
    u = mean(Xd, 2);
    Y =  Xd ./ repmat( sum( Xd .* repmat(u,[1 N]) ) ,[d 1]);
    %for j=1:N
    %    Y(:,j) = Xd(:,j) / (Xd(:,j).'*u);
    %end
else
    d = q-1;
    r_bar = mean(M.').';
    Ud = pca(M, d);
    %Ud = Ud(:, 1:d);
    R_zeroMean = M - repmat(r_bar, 1, N);
    Xd = Ud.' * R_zeroMean;
    % Preallocate memory for speed.
    c = zeros(N, 1);
    for j=1:N
        c(j) = norm(Xd(:,j));
    end
    c = repmat(max(c), 1, N);
    Y = [Xd; c];
end
e_u = zeros(q, 1);
e_u(q) = 1;
A = zeros(q, q);
% idg - Doesnt match.
A(:, 1) = e_u;
I = eye(q);
k = zeros(N, 1);
for i=1:q
    w = rand(q, 1);
    % idg - Oppurtunity for speed up here.
    tmpNumerator =  (I-A*pinv(A))*w;
    %f = ((I - A*pinv(A))*w) /(norm( tmpNumerator ));
    f = tmpNumerator / norm(tmpNumerator);

    v = f.'*Y;
    k = abs(v);
    [dummy, k] = max(k);
    A(:,i) = Y(:,k);
    indicies(i) = k;
end
if (SNR > SNRth)
    U = Ud*Xd(:,indicies);
else
    U = Ud*Xd(:,indicies) + repmat(r_bar, 1, q);
end
return;

function [U] = pca(X, d)
    N = size(X, 2);
    xMean = mean(X, 2);
    XZeroMean = X - repmat(xMean, 1, N);     
    [U,S,V] = svds((XZeroMean*XZeroMean.')/N, d);
return;





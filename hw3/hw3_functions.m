% --------------- function definitions -------------------------------

%method 1: this function sums the RGB axis into one
function x = sumRGB(vidFrames,height,width,num_frames)
    x = zeros(height,width,num_frames);
    for j =1 : num_frames
        x(:,:,j) = sum(vidFrames(:,:,:,j),3);
    end
end 

%method 2: this function returns the gradient
function x = grad_space(vidFrames, height, width, num_frames)
    x = zeros(height, width, num_frames);
    for j = 1 : num_frames
        [vFx, vFy] = gradient(vidFrames(:,:,j));
        x(:,:,j) = abs(vFx)+abs(vFy);
    end
end

%method 3: change in time
function x = grad_time(vidFrames, height, width, num_frames)
    x = zeros(height, width, num_frames-1);
    for j = 2 : num_frames
        x(:,:,j-1) = vidFrames(:,:,j) - vidFrames(:,:,j-1);
    end
end

%method 4: sum neighbors
function x = sum_neighbors(vidFrames, height, width, num_frames)
    x = zeros(height-1,width-1,num_frames);
    for j = 1 : num_frames
        for i = 1 : height-1
            for k = 1 : width-1
                x(i,k,j) = vidFrames(i,k,j) + vidFrames(i+1,k,j) + vidFrames(i,k+1,j);
            end
        end
    end 
end

%find position of brightest pixel
function [x,y] = findlight(vidFrames,height,width,num_frames)
    x = zeros(1,num_frames); y = zeros(1,num_frames);
    for j = 1 : num_frames
        [~, i] = max(vidFrames(:,:,j),[],'all','linear');
        [y(j),x(j)] = ind2sub([height width], i);
    end
end

%perform diagonalization and projection
function Y = PCA(x1,y1,x2,y2,x3,y3,k,r)
    X = [x1; y1; x2; y2; x3; y3];
    [~,n] = size(X);
    mn = mean(X,2);
    X = X - repmat(mn,1,n);

    Cx = (1/(n-1))*X*X';
    [V,D] = eig(Cx);
    lambda = diag(D);

    [~, m_arrange] = sort(-1*lambda);
    lambda = lambda(m_arrange);
    V=V(:,m_arrange);

    Y=V'*X;

    figure(k)
    subplot(2,2,r), plot(Y(2,:))
    
   if r == 1
       xlabel('I. Ideal')
   elseif r == 2
       xlabel('II. Noisy')
   elseif r == 3
       xlabel('III. Horizontal')
   else
       xlabel('IV. Horizontal & Rotation')
   end
    
end
function pc2 = rotatePointCloudAlongZ(pc, direction)
    % This function rotates the point cloud of the highest eigenvector  
    % to be along z-direction. 
    % For the 2nd highest eigenvector, the user can choose to align 
    % along x or y direction 
    
    % parameters: 
    % inputs: 
    % pc: point cloud of size N * 3 size
    % direction: 'x' or 'y' (align the 2nd highest eigenvector along 'x' or
    % 'y' direction 
    % output:
    % pc2: rotated point cloud 
    %% Bring the point cloud center to the origin
    
    pc = pc - mean(pc);
    
    %% This section align u normal vector along z-direction
    
    % Obtain the eigenvector of the highest eigenvalue
    u = pcaEig(pc, 'max');
       
    % Calculate the angles of the normal vector 
    [alpha, beta] = unitVectorToAngle(u);
    
    % Align the point cloud along x-axis followed by aligning along z-axis
    % YOU CAN REMOVE THE PI IF YOU WANT TO FLIP THE POINT CLOUD ALONG
    % Z-DIRECTION 
    [~, Ry, Rz] = rotationalMatrix(-alpha, pi-beta);
    pc2 = rotatePC(pc, Ry, Rz);
    
    %% This section align v normal vector along x or y direction
    
    switch direction
        case 'x'
            offset = 0;
        case 'y'
            offset = pi/2;
    end
    
    % Obtain the eigenvector of the 2nd highest eigenvalue 
    v = pcaEig(pc2, 'middle');
    
    % Calculate the angle of the projected v-vector along the xy-plane
    % with respect to the x-axis
    [alpha, ~] = unitVectorToAngle(v);
    
    % Calculate the rotational matrix for the angle
    [~, Ry, Rz] = rotationalMatrix(offset - alpha, 0);
    
    % Rotate the point cloud 
    pc2 = rotatePC(pc2, Ry, Rz);
    
end
function [Rx, Ry, Rz] = rotationalMatrix(alpha, beta)
    
    Rx = [1 0 0; 0 cos(beta) -sin(beta); 0 sin(beta) cos(beta)];
    Ry = [cos(beta) 0 sin(beta); 0 1 0; -sin(beta) 0 cos(beta)];
    Rz = [cos(alpha) -sin(alpha) 0; sin(alpha) cos(alpha) 0; 0 0 1];
    
end
function u = pcaEig(pc, magnitude)
    %% Obtain the covariance matrix
    
    covariance = cov([pc(:, 1) pc(:, 2) pc(:, 3)]);
    
    %% Calculate the eigenvectors and obtain the normal vector
    
    [V, D] = eig(covariance);
    diagonalEigenvalues = diag(D);
    
    %% Output the normal vector 
    
    % Sort the eigenvectors based on size of eigenvalues 
    [~, I] = sort(diagonalEigenvalues);
    V = V(:, I);
    
    switch magnitude
        case 'max'
            % Choose the eigenvector of the highest eigenvalue
            u = V(:, 3); 
        case 'middle'
            % Choose the eigenvector of the middle eigenvalue
            u = V(:, 2); 
        case 'min'
            % Choose the eigenvector of the lowest eigenvalue
            u = V(:, 1); 
    end
    
end
function [alpha, beta] = unitVectorToAngle(u)
    % Rotational angle between the projected u on the xy plane and the x-axis
    alpha = atan2(u(2), u(1)); 
    % Rotational angle between the u vector and the z-axis
    beta = atan2(sqrt(u(1)^2 + u(2)^2), u(3));
    
end
function pc2 = rotatePC(pc, Ry, Rz)
    % Convert the point cloud to 3 * N format
    matrix = pc';
    % rotation around z axis to align point cloud along x axis 
    matrix2 = Rz*matrix;
    matrix2 = Ry*matrix2;
    % Ouput the point cloud 
    pc2 = matrix2';
end
function [visible_idx] = HPR_operator(point_cloud, camera_center, r)
%% Katz, Sagi, Ayellet Tal, and Ronen Basri. "Direct visibility of point sets." ACM SIGGRAPH 2007 papers. 2007. 24-es. 
% point_cloud: a N(num points)*M(dimension) point cloud
% camera_center: (x,y,z) camera location
% r: a parameter to calculate the radius R

[N,M] = size(point_cloud);
central_point_cloud = point_cloud-repmat(camera_center,[N 1]);
normal = sqrt(dot(central_point_cloud,central_point_cloud,2));
R = repmat(max(normal)*(10^r),[N 1]);
flip_cloud = central_point_cloud+2*repmat(R-normal,[1 M]).*central_point_cloud./repmat(normal,[1 M]);
visible_idx = unique(convhulln(double([flip_cloud;zeros(1,M)])));
visible_idx(visible_idx==N+1)=[];
end
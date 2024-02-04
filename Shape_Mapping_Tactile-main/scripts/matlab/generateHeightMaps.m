clc;
clear;

%% object path
obj = 'power_drill';
path_processed_data = '../../processed_data/';
path_data = '../../data/';
path_ply = strcat(path_processed_data,obj,'/',obj,'.ply');
path_sample = strcat(path_data,obj,'/',obj,'_50sampled.mat');
path_save = strcat(path_processed_data,obj,'/gt_height_map_mat/');

%% load point cloud

% ptCloud = pcread('../../processed_data/mustard_bottle/mustard_bottle.ply');
ptCloud = pcread(path_ply);

point_cloud = ptCloud.Location;
[N,M] = size(point_cloud);
point_cloud_hom = [point_cloud,ones(N,1)];
% pcshow(ptCloud);
% hold on;

%% load sample info

% sample_info = load('../../data/mustard_bottle/mustard_bottle_50sampled.mat');
sample_info = load(path_sample);

%% set parameters
pressing_depth = 1.5;
pixmm = 0.0295;
h = 480;
w = 640;


%% generate height maps
for k = 1:sample_info.numSamples
    cur_point = sample_info.samplePoints(k,:);
    cur_normal = -1*sample_info.sampleNormals(k,:);
    % generate camera pose
    z_axis = [0 0 1];
    if isequal(z_axis,cur_normal)
        R = [1 0 0; 0 1 0; 0 0 1];
    else
        v = cross(z_axis,cur_normal);
        s = norm(v);
        c = dot(z_axis,cur_normal);
        R = eye(3) + skewMat(v) + skewMat(v)^2 * (1-c)/(s^2);
    end
    T_wp = [R' [0 ; 0 ; 0]; cur_point 1]';

    T_pw = inv(T_wp);

    % pcshow(point_cloud);
    point_cloud_cam = T_pw*(point_cloud_hom');
    point_cloud_3d = point_cloud_cam./point_cloud_cam(4,:);
    point_cloud_3d = point_cloud_3d*1000; % in mm
    point_cloud_idx = find( point_cloud_3d(3,:) > -1*pressing_depth & point_cloud_3d(3,:) < pressing_depth & point_cloud_3d(1,:)/pixmm > -1 * w/2 & point_cloud_3d(1,:)/pixmm <= w/2 & point_cloud_3d(2,:)/pixmm > -1*h/2 & point_cloud_3d(2,:)/pixmm <= h/2);
    point_cloud_valid = point_cloud(point_cloud_idx,:);
%     pcshow(point_cloud_valid);
%     hold on;
%     % camera visualization
%     l = 0.01;
%     A = [0 0 0 1; l 0 0 1; 0 0 0 1; 0 l 0 1; 0 0 0 1; 0 0 l 1]';
%     % for i = 1:20:n
%     B = T_wp*A;
%     n = 100;
%     cam_pc = zeros(n*3,3);
%     cam_color = zeros(n*3,3);
%     % x
%     xr = B(1,2)-B(1,1);
%     yr = B(2,2)-B(2,1);
%     zr = B(3,2)-B(3,1);
%     for i = 1:n
%         x = B(1,1) + i*xr/n;
%         y = B(2,1) + i*yr/n;
%         z = B(3,1) + i*zr/n;
%         cam_pc(i,:) = [x y z];
%         cam_color(i,:) = [255 0 0];
%     end
%     % y
%     xr = B(1,4)-B(1,3);
%     yr = B(2,4)-B(2,3);
%     zr = B(3,4)-B(3,3);
%     for i = 1:n
%         x = B(1,3) + i*xr/n;
%         y = B(2,3) + i*yr/n;
%         z = B(3,3) + i*zr/n;
%         cam_pc(n+i,:) = [x y z];
%         cam_color(n+i,:) = [0 255 0];
%     end
%     % z
%     xr = B(1,6)-B(1,5);
%     yr = B(2,6)-B(2,5);
%     zr = B(3,6)-B(3,5);
%     for i = 1:n
%         x = B(1,5) + i*xr/n;
%         y = B(2,5) + i*yr/n;
%         z = B(3,5) + i*zr/n;
%         cam_pc(2*n+i,:) = [x y z];
%         cam_color(2*n+i,:) = [0 0 255];
%     end
%     pcshow(cam_pc,cam_color);
%     hold on;
    % visibility check and filtering
    r = 3.8;
    camera_center = cur_point;
    visible_idx = HPR_operator(point_cloud_valid, camera_center, r);
%     visible_cloud = point_cloud_valid(visible_idx,:);

%     pcshow(visible_cloud);
%     hold on;

    % heightmap
    point_cloud_3d_valid = point_cloud_3d(:,point_cloud_idx)/pixmm;
    visible_3d = point_cloud_3d_valid(1:3,visible_idx);
    max_z = max(visible_3d(3,:));
    pcshow(visible_3d');
%     hold on;

    % generate 2d image
    image=zeros(480,640,'single'); %initialize
    visible_3d(1,:) = visible_3d(1,:) + 320;
    visible_3d(2,:) = visible_3d(2,:) + 240;
    max_z = max(visible_3d(3,:));
    visible_3d(3,:) = -1 * visible_3d(3,:) + max_z;
    pt3_locate = int64(visible_3d(1:2,:));
    pt3_locate(1,pt3_locate(1,:) == 0 ) = 1;
    pt3_locate(2,pt3_locate(2,:) == 0 ) = 1;
    for i = 1:size(pt3_locate,2)
        image(pt3_locate(2,i),pt3_locate(1,i),1) = visible_3d(3,i);
    end
    save(strcat(path_save,num2str(k-1),'.mat'),'image');
end

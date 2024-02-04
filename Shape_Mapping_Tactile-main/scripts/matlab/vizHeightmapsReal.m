function vizHeightmapsReal(dir, obj, pauseFlag, dataSource)
    % load gelsight data
    
    gelsightFile = fullfile(dir, 'ycbSight', obj, 'gelsight_data.mat'); load(gelsightFile);
    
    if strcmp(dataSource, 'fcrn')
        genFile = fullfile(dir, 'generated_data', 'fcrn', obj, 'fcrn_data.mat'); 
        genData = load(genFile); gen_heightmaps = genData.fcrn_heightmaps; gen_normalmaps = genData.fcrn_normalmaps; 
    elseif strcmp(dataSource, 'mlp')
        genFile = fullfile(dir, 'generated_data', 'mlp', obj, 'mlp_data.mat'); 
        genData = load(genFile); gen_heightmaps = genData.mlp_heightmaps; gen_normalmaps = genData.mlp_normalmaps; 
    elseif strcmp(dataSource, 'lookup')
        genFile = fullfile(dir, 'generated_data', 'lookup', obj, 'lookup_data.mat'); 
        genData = load(genFile); gen_heightmaps = genData.lookup_heightmaps; gen_normalmaps = genData.lookup_normalmaps; 
    end
    
    elev = 30;

    N = size(poses, 1);

    figure('Name', 'Gelsight output');

    s1 = subplot(3, 2, [1 3 5]);
    hold on;
    axis equal off
    shading interp % Make surface look smooth
    camlight; lighting phong % Shine light on surface
    % colormap(s1,gray);
    view(-37.5, elev);
    xlim manual; ylim manual; zlim manual;

    pixmm = 0.0295; % heightmap (pix) -> 3D conversion (mm)
    max_depth = 1.0; %mm
    max_z = 1.0/pixmm; % pix

    fprintf("Obj: %s, Method: %s, ", obj, dataSource);
    
    %% plot the depth map 
    
    world2azure = toPose(world2azure); 
    world2object = toPose(world2object).inv; 
    gripper2gelsight = toPose(gripper2gelsight);
    
    sensorPoses = {}; 
    sensorPositions = zeros(N, 3);
    fprintf('Plotting sensor readings...\n');
    
    %% get poses 
    for i = 1:N
        poses(i, [4, 5, 6, 7]) = poses(i, [7, 4, 5, 6]);        % x y z w -> w x y z
        sensorPose = SE3(quat2rotm(poses(i, 4:end)), poses(i, 1:3));
                
        gelsightTrans = SE3(eye(3), gripper2gelsight.t); 
        sensorPose = sensorPose * gelsightTrans;
        gelsightRot = SE3(gripper2gelsight.R, zeros(1, 3)); 
        sensorPose = sensorPose * gelsightRot;    
        
        sensorPose = world2object * sensorPose;
        
        sensorPoses{i} = sensorPose; 
        sensorPositions(i, :) = sensorPose.t';
    end
    
    %% plot depthmap 
    world_depth_map = pointCloud((world2azure * depth_map')');
    object_depth_map = pointCloud((world2object * world_depth_map.Location')');

%     xmin = -15.0/100.0; xmax = 15.0/100.0; 
%     ymin = -15.0/100.0; ymax = 15.0/100.0; 
%     zmin = (3.9 + 3)/100.0;  zmax = 35.0/100.0;
    
    xmin = min(sensorPositions(:, 1)); xmax = max(sensorPositions(:, 1)); 
    ymin = min(sensorPositions(:, 2));  ymax = max(sensorPositions(:, 2)); 
    zmin = min(sensorPositions(:, 3));   zmax = max(sensorPositions(:, 3)); 
    
    roi = [xmin xmax ymin ymax zmin zmax] + [-1 1 -1 1 -2 3]*1e-2;
    indices = findPointsInROI(object_depth_map,roi);
    object_depth_map = select(object_depth_map,indices);
    p = pcshow(object_depth_map, 'MarkerSize', 20);

    set(gcf,'color','w'); set(gca,'color','w');

%     h(1)= plot3(0, 0, 0, 'o', 'Color','b','MarkerSize',5,'MarkerFaceColor','#D9FFFF', 'DisplayName','World origin');
    
    
    %% plto sensor measurements 
    for i = 1:N
        fprintf('%d ', i);
        %% plot sensor pose
        subplot(3, 2, [1 3 5]);
        
        %% reshape column matrices to data
        heightmap = reshape(gen_heightmaps(:, i),[480,640]);
        normalmap = reshape(gen_normalmaps(:, 3*(i-1) + 1 : 3*(i-1) + 3),[480*640, 3]);            
        contactmask = reshape(est_contactmasks(:, i),[480,640]);

        tactileimage = reshape(tactileimages(:, 3*(i-1) + 1 : 3*(i-1) + 3),[480,640, 3]);
        tactileimage = rescale(tactileimage); 

        rotation = SO3(quat2rotm(poses(i, 4:end)));
        normalmap = (rotation * normalmap')';

        %% ground truth heightmap
        heightmap = -1 * (heightmap - max_z); % invert gelsight heightmap
        heightmapValid = heightmap .* contactmask; % apply contact mask
        [x,y] = meshgrid((1:size(heightmapValid,2)) - size(heightmapValid,2)/2,...
                         (1:size(heightmapValid,1)) - size(heightmapValid,1)/2);
        heightmap_3d = [x(:), y(:), heightmapValid(:)];
        normalmap(heightmap_3d(:,3) == 0, :) = []; % apply contact mask
        heightmap_3d(heightmap_3d(:,3) == 0, :) = []; % apply contact mask
        local_point_cloud = heightmap_3d'*pixmm/1000; % heightmap (pix) -> 3D conversion (m)
        world_point_cloud = sensorPoses{i} * local_point_cloud;     % convert to global cooridnates via sensorPose
        p0 = pcshow(world_point_cloud', 'g');
        
        skip = 10; ratio = 1e-3;
        sensorPoses{i}.plot('rgb', 'length', 0.01, 'labels', '   ', 'thick', 1);
        quiver3(world_point_cloud(1,1:skip:end),world_point_cloud(2,1:skip:end),world_point_cloud(3,1:skip:end),...
        ratio*normalmap(1:skip:end,1)', ratio*normalmap(1:skip:end,2)', ratio*normalmap(1:skip:end,3)', 'Color', 'c');
        set(gcf,'color','w');set(gca,'color','w');
        drawnow;


        if pauseFlag
            %% plot GelSight image
            s2 = subplot(3, 2, 2);
            imshow(tactileimage, []);
            title(strcat('Sensor reading #', num2str(i)))

            %% plot groundtruth heightmap
            s3 = subplot(3, 2, 4);
            imshow(heightmap, []);
            title(strcat('Heightmap #', num2str(i)))
            colormap(s3, jet);

            %% plot contact mask
            s4 = subplot(3, 2, 6);
            imshow(contactmask, []);
            title(strcat('Contactmask #', num2str(i)))
            pause(1);
        end
    end
    

    %% save visualized model
    fig = figure('visible', 'off');
    h = copyobj(s1,fig);
    set(h, 'pos', [0.23162 0.2233 0.72058 0.63107]);
    savename = strcat('gelsight_', dataSource, '.png');
    savepath = fullfile(dir, 'generated_data', dataSource, obj, savename);
    fprintf("Save path: %s\n", savepath);
    exportgraphics(h,savepath, 'Resolution',1000);
    close(fig);
    close all;
end

function p = toPose(xyzquat)
    R = quat2rotm(xyzquat([7, 4, 5, 6])); 
    t = xyzquat(1:3);
    p = SE3(R, t);
end

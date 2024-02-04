function vizHeightMaps(obj, datasetName, pauseFlag, dataSource)
    % load gelsight data
    shapeMappingRootPath = getenv('SHAPE_MAPPING_ROOT');

    gelsightFile = fullfile(shapeMappingRootPath, 'gelsight_data', datasetName, obj, 'gelsight_data.mat'); load(gelsightFile);
    
    if strcmp(dataSource, 'fcrn')
        genFile = fullfile(shapeMappingRootPath, 'generated_data', datasetName, 'fcrn', obj, 'fcrn_data.mat'); 
        genData = load(genFile); gen_heightmaps = genData.fcrn_heightmaps; gen_normalmaps = genData.fcrn_normalmaps; 
    elseif strcmp(dataSource, 'mlp')
        genFile = fullfile(shapeMappingRootPath, 'generated_data', datasetName, 'mlp', obj, 'mlp_data.mat'); 
        genData = load(genFile); gen_heightmaps = genData.mlp_heightmaps; gen_normalmaps = genData.mlp_normalmaps; 
    elseif strcmp(dataSource, 'lookup')
        genFile = fullfile(shapeMappingRootPath, 'generated_data', datasetName, 'lookup', obj, 'lookup_data.mat'); 
        genData = load(genFile); gen_heightmaps = genData.lookup_heightmaps; gen_normalmaps = genData.lookup_normalmaps; 
    end
    
    modelFile = fullfile(shapeMappingRootPath, 'models', obj, 'textured.mat'); load(modelFile);
    elev = 30;

    N = size(gt_heightmaps, 2);

    figure('Name', 'Gelsight output');

    s1 = subplot(3, 2, [1 3 5]);
    hold on;
    trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3), 'EdgeColor', 'none', 'FaceAlpha', 0.5);
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
    
    if strcmp(dataSource, 'depth_map')
        %% plot the depth map 
        p1 = pcshow(depth_map, 'c', 'MarkerSize', 10);
        set(gcf,'color','w'); set(gca,'color','w');
    else
        for i = 1:N
            %% plot sensor pose
            subplot(3, 2, [1 3 5]);
            poses(i, [4, 5, 6, 7]) = poses(i, [7, 4, 5, 6]);        % x y z w -> w x y z
            sensorPose = SE3(quat2rotm(poses(i, 4:end)), poses(i, 1:3));
            sensorPose.plot('rgb', 'length', 0.01, 'labels', '   ', 'thick', 1);
    %         plot3(samplePoints(i, 1), samplePoints(i, 2), samplePoints(i, 3), 'yo');

            %% reshape column matrices to data
            if strcmp(dataSource, 'gt')
                heightmap = reshape(gt_heightmaps(:, i),[480,640]);
                normalmap = reshape(gt_normalmaps(:, 3*(i-1) + 1 : 3*(i-1) + 3),[480*640, 3]);
                contactmask = reshape(gt_contactmasks(:, i),[480,640]);
            else            
                heightmap = reshape(gen_heightmaps(:, i),[480,640]);
                normalmap = reshape(gen_normalmaps(:, 3*(i-1) + 1 : 3*(i-1) + 3),[480*640, 3]);            
                contactmask = reshape(est_contactmasks(:, i),[480,640]);
            end

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
            world_point_cloud = sensorPose * local_point_cloud;     % convert to global cooridnates via sensorPose
            p0 = pcshow(world_point_cloud', 'g');

            skip = 5000; ratio = 1e-3;

            set(gcf,'color','w');set(gca,'color','w');

            if pauseFlag

                quiver3(world_point_cloud(1,1:skip:end),world_point_cloud(2,1:skip:end),world_point_cloud(3,1:skip:end),...
                ratio*normalmap(1:skip:end,1)', ratio*normalmap(1:skip:end,2)', ratio*normalmap(1:skip:end,3)', 'Color', 'c');

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
    end
    %% save visualized model
    fig = figure('visible', 'off');
    h = copyobj(s1,fig);
    set(h, 'pos', [0.23162 0.2233 0.72058 0.63107]);
    if strcmp(dataSource, 'gt')
        savename = 'gelsight_gt.png';
        savepath = fullfile(shapeMappingRootPath, 'gelsight_data', datasetName, obj, savename);
    elseif strcmp(dataSource, 'depth_map')
        savename = 'depth_map.png';
        savepath = fullfile(shapeMappingRootPath, 'gelsight_data', datasetName, obj, savename);
    else
        savename = strcat('gelsight_', dataSource, '.png');
        savepath = fullfile(shapeMappingRootPath, 'generated_data', datasetName, dataSource, obj, savename);
    end

    fprintf("Save path: %s\n", savepath);
    exportgraphics(h,savepath, 'Resolution',1000);
    close(fig);
    close all;
end
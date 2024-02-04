function sampleModelPoints(obj, numSamples, saveFlag)
    shapeMappingRootPath = getenv('SHAPE_MAPPING_ROOT');

    keySet = ["002_master_chef_can","003_cracker_box","004_sugar_box", "005_tomato_soup_can",...
              "006_mustard_bottle","007_tuna_fish_can","008_pudding_box","009_gelatin_box",...
              "010_potted_meat_can","011_banana","012_strawberry","013_apple","014_lemon",...
              "017_orange","019_pitcher_base","021_bleach_cleanser","024_bowl","025_mug",...
              "029_plate","035_power_drill","036_wood_block","037_scissors","042_adjustable_wrench",...
              "043_phillips_screwdriver",	"048_hammer","055_baseball","056_tennis_ball",...
              "072-a_toy_airplane","072-b_toy_airplane","077_rubiks_cube"];
    valueSet = ["top_circle","top_line","top_line", "top_circle",...
              "top_circle","top_circle","top_line","top_line",...
              "top_line","top_line","top_line","top_line","top_line",...
              "top_line","top_circle","top_circle","top_line","top_line",...
              "top_line","top_circle","top_circle","top_line","top_line",...
              "top_line",	"top_line","top_line","top_line",...
              "top_line","top_line","top_line"];
    topMap = containers.Map(keySet,valueSet);

    dataPath = fullfile(shapeMappingRootPath, 'models', obj, 'textured.mat');
    load(dataPath); XObj = vertices; YObj = normals; YObj = normr(YObj); TriObj = faces;
    count = length(XObj);
       
    %% subsample dense point cloud 
    cloud = pointCloud(XObj, 'Normal', YObj);
    mLimit = mean([cloud.XLimits(2) - cloud.XLimits(1), cloud.YLimits(2) - cloud.YLimits(1), cloud.ZLimits(2) - cloud.ZLimits(1)]); 
    newCloud = pointCloud([0 0 0]);
    scaleFactor = 1;
    while newCloud.Count < numSamples
        newCloud = pcdownsample(cloud,'gridAverage',mLimit/(scaleFactor*100));
        scaleFactor = scaleFactor + 1;
    end
    cloud = newCloud;
    
    %% Sample contact points
    
    if (strcmp(obj, "010_potted_meat_can"))
        rCloud = cloud.Location;
    elseif (strcmp(obj, "004_sugar_box"))
        rCloud = rotatePointCloudAlongZ(cloud.Location, 'x'); % rotate cloud along positive Z
    else
        rCloud = rotatePointCloudAlongZ(cloud.Location, 'y'); % rotate cloud along positive Z
    end
    
    
    radius = 0.2;
    
    if (strcmp(obj, "004_sugar_box"))
        l = (max(rCloud(:,1)) - min(rCloud(:,1)))/2 - 1e-2;
        b = (max(rCloud(:,2)) - min(rCloud(:,2)))/2 - 1e-2;  
    elseif (strcmp(obj, "010_potted_meat_can"))
        l = (max(rCloud(:,1)) - min(rCloud(:,1)))/2 - 1.5e-2;
        b = (max(rCloud(:,2)) - min(rCloud(:,2)))/2 - 1.5e-2;  
    elseif (strcmp(obj, "005_tomato_soup_can"))
        l = (max(rCloud(:,1)) - min(rCloud(:,1)))/2 - 1.5e-2;
        b = (max(rCloud(:,2)) - min(rCloud(:,2)))/2 - 1.5e-2;
    else
        l = (max(rCloud(:,1)) - min(rCloud(:,1)))/2 - 2e-2;
        b = (max(rCloud(:,2)) - min(rCloud(:,2)))/2 - 2e-2;       
    end
    
    % N heights, M angles, K top/bottom = 5*8 + 5 + 5 = 50 contacts
    if ( strcmp(obj, "021_bleach_cleanser") || strcmp(obj, "036_wood_block") )
        N = 6; M = 8;
        zHeights = linspace(min(rCloud(:,3)) + 1e-2, max(rCloud(:,3)) - 1e-2, N);
    else
        N = 5; M = 8;
        zHeights = linspace(min(rCloud(:,3)) + 1e-2, max(rCloud(:,3)) - 1e-2, N);
    end
    
    %% get side points = N*M points
    safePts = [];
    % Plot a circle.
    for z = zHeights
        if (strcmp(obj, "004_sugar_box"))
           angles = deg2rad([0    30    90   150   180   210   270   330 ]);
        elseif (strcmp(obj, "010_potted_meat_can"))
           angles = deg2rad([0    45    90   135   180   215   270   315 ]);
        else
            angles = linspace(0, 2*pi, M + 1); 
            angles = angles(1:M);   
        end

        xCenter = mean(rCloud(:, 1));
        yCenter = mean(rCloud(:, 2));
        x = radius * cos(angles) + xCenter; 
        y = radius * sin(angles) + yCenter;
        newPts = [x', y', ones(size(x'))*z];
        safePts = [safePts; newPts];
    end

    c = cos(deg2rad(45));
    
    if (strcmp(topMap(obj), 'top_line'))
        if (strcmp(obj, "010_potted_meat_can"))
            xVals = linspace(min(rCloud(:,1)) + 1.5e-2,max(rCloud(:,1)) - 1.5e-2,5);
            yVals = linspace(min(rCloud(:,2)) + 1.5e-2,max(rCloud(:,2)) - 1.5e-2,2);
        elseif (strcmp(obj, "004_sugar_box"))
            xVals = linspace(min(rCloud(:,1)) + 1e-2,max(rCloud(:,1)) - 1e-2,5);
            yVals = linspace(min(rCloud(:,2)) + 1e-2,max(rCloud(:,2)) - 1e-2,2);
        else
             xVals = linspace(min(rCloud(:,1)),max(rCloud(:,1)),5 + 2);
             xVals = xVals(2:5 + 1);
             yVals = linspace(min(rCloud(:,2)),max(rCloud(:,2)),2 + 2);
             yVals = yVals(2:2 + 1);           
        end
            
        newPts = [xVals(1), yVals(1), radius; 
                xVals(2), yVals(1), radius; 
                xVals(3), yVals(1), radius; 
                xVals(4), yVals(1), radius; 
                xVals(5), yVals(1), radius; 
                xVals(1), yVals(2), radius; 
                xVals(2), yVals(2), radius; 
                xVals(3), yVals(2), radius; 
                xVals(4), yVals(2), radius; 
                xVals(5), yVals(2), radius; 
                xVals(1), yVals(1), -radius; 
                xVals(2), yVals(1), -radius; 
                xVals(3), yVals(1), -radius; 
                xVals(4), yVals(1), -radius; 
                xVals(5), yVals(1), -radius; 
                xVals(1), yVals(2), -radius; 
                xVals(2), yVals(2), -radius; 
                xVals(3), yVals(2), -radius; 
                xVals(4), yVals(2), -radius; 
                xVals(5), yVals(2), -radius;];
    elseif ( strcmp(obj, "021_bleach_cleanser") || strcmp(obj, "036_wood_block") ) 
          newPts = [mean(rCloud(:,1)) + 1e-2, mean(rCloud(:,2)), radius; 
                  mean(rCloud(:,1)) + 1e-2, mean(rCloud(:,2)), -radius;
                  mean(rCloud(:,1)) - 1e-2, mean(rCloud(:,2)), radius; 
                  mean(rCloud(:,1)) - 1e-2, mean(rCloud(:,2)), -radius;
                  c*l, c*b, radius; 
                  c*l,  -c*b, radius; 
                  -c*l,  -c*b, radius; 
                  -c*l, c*b, radius; 
                  c*l, c*b, -radius;
                  c*l,  -c*b, -radius;
                  -c*l,  -c*b, -radius;
                  -c*l,  c*b, -radius;];      
    else
        newPts = [mean(rCloud(:,1)) + 1e-2, mean(rCloud(:,2)), radius; 
                  mean(rCloud(:,1)) + 1e-2, mean(rCloud(:,2)), -radius;
                  mean(rCloud(:,1)) - 1e-2, mean(rCloud(:,2)), radius; 
                  mean(rCloud(:,1)) - 1e-2, mean(rCloud(:,2)), -radius;
                  c*l, c*b, radius; 
                  c*l,  -c*b, radius; 
                  -c*l,  -c*b, radius; 
                  -c*l, c*b, radius; 
                  c*l, c*b, -radius;
                  c*l,  -c*b, -radius;
                  -c*l,  -c*b, -radius;
                  -c*l,  c*b, -radius;
                  l, mean(rCloud(:,2)), radius; 
                  -l,  mean(rCloud(:,2)), radius; 
                  mean(rCloud(:,1)), b, radius; 
                  mean(rCloud(:,1)), -b, radius; 
                  l, mean(rCloud(:,2)), -radius; 
                  -l,  mean(rCloud(:,2)), -radius; 
                  mean(rCloud(:,1)), b, -radius; 
                  mean(rCloud(:,1)), -b, -radius;];
    end
    safePts = [safePts; newPts];
    
    %% get top/bottom points = K + K
%     top = prctile(rCloud(:, 3),90); bot = prctile(rCloud(:, 3),10);
%     top_id = find(rCloud(:, 3) > top);    bot_id = find(rCloud(:, 3) < bot);
   
%     if isTopBottom
%         %% check for angles 
%         while 1
%             samp_id = datasample(top_id,K,'Replace',false);
%             samp_n = cloud.Normal(samp_id, :);
%             flag = true;
%             for i = 1:K
%                 ni = samp_n(i, :);
%                 for j = 1:K
%                     nj = samp_n(j, :);
%                     ang = rad2deg(atan2(norm(cross(ni,nj)),dot(ni,nj)));
%                     if (ang > 60) flag = false; break; end
%                 end
%                 if (flag == false) break; end
%             end
%             if (flag == true) break; end
%         end
%         sampleTopID = samp_id';
%         
%         while 1
%             samp_id = datasample(bot_id,K,'Replace',false);
%             samp_n = cloud.Normal(samp_id, :);
%             flag = true;
%             for i = 1:K
%                 ni = samp_n(i, :);
%                 for j = 1:K
%                     nj = samp_n(j, :);
%                     ang = rad2deg(atan2(norm(cross(ni,nj)),dot(ni,nj)));
%                     if (ang > 60) flag = false; break; end
%                 end
%                 if (flag == false) break; end
%             end
%             if (flag == true) break; end
%         end
%         sampleBotID = samp_id';   
%     else
%         skip = round(size(top_id, 1)/K); rIdx = 1:skip:size(top_id, 1);  sampleTopID = top_id(rIdx)'; 
%         skip = round(size(bot_id, 1)/K); rIdx = 1:skip:size(bot_id, 1);  sampleBotID = bot_id(rIdx)';   
%     end
    sampleSideID = [];
    for i = 1:size(safePts, 1)
        % get closest point on the vector
        if (i <= M*N)
            origin = [mean(rCloud(:, 1)), mean(rCloud(:, 2)), safePts(i, 3)]; 
        else
            origin = [safePts(i, 1), safePts(i, 2), mean(rCloud(:, 3))]; 
        end
        dist = point_to_line_distance(rCloud, origin, safePts(i, :));
        [~, pos] = mink(dist, 100);
        p = pos(1);
        % prevent point on other side of object 
        for j = 1:length(pos)
        if pdist2(safePts(i, :), rCloud(pos(j), :)) < pdist2(safePts(i, :), origin)
            p = pos(j);
            break;
        end
        end
        sampleSideID = [sampleSideID, p];
    end

    samplePoints = cloud.Location([sampleSideID], :); 
    sampleNormals = cloud.Normal([sampleSideID], :); 
    [samplePoints, idx] = sortrows(samplePoints,3);
    sampleNormals = sampleNormals(idx, :);
    
    %% plot 
    f = figure; axis equal off; hold on; view(3);
    trisurf(TriObj, XObj(:,1), XObj(:,2), XObj(:,3), 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    shading interp % Make surface look smooth
    camlight; lighting phong % Shine light on surface
    colormap(gca,gray);
    plotPointsAndNormals3D(samplePoints, sampleNormals, 0.1);
    
%     plot3( mean(rCloud(:,1)),  mean(rCloud(:,2)),  mean(rCloud(:,3)), 'ro');
%     rotPlot = plot3( rCloud(1:50:end,1),  rCloud(1:50:end,2),  rCloud(1:50:end,3), 'r.');
%     plot3( rCloud(sampleSideID,1),  rCloud(sampleSideID,2),  rCloud(sampleSideID,3), 'b.', 'MarkerSize', 20);
%     plot3( safePts(:,1),  safePts(:,2),  safePts(:,3), 'g.');
%     plot3( sampleTopPoints(:,1),  sampleTopPoints(:,2),  sampleTopPoints(:,3), 'k.',  'MarkerSize', 20);
%     plot3( sampleBotPoints(:,1),  sampleBotPoints(:,2),  sampleBotPoints(:,3), 'k.',  'MarkerSize', 20);

    [modelPath,modelName,ext] = fileparts(dataPath);
    disp(strcat(modelPath, '/', modelName, '_', num2str(numSamples), 'sampled',  '.mat'));
    
    if saveFlag
        % eg: /home/suddhu/projects/Shape_Mapping_Tactile/data/002_master_chef_can/google_512k/textured_120sampled.mat
        save(strcat(modelPath, '/', modelName, '_', num2str(numSamples), 'sampled',  '.mat'), 'modelName', 'vertices', 'normals', 'faces',...
                        'numSamples', 'samplePoints', 'sampleNormals');
        f = gcf;
        exportgraphics(f,strcat(modelPath, '/', modelName, '_', num2str(numSamples), 'sampled',  '.pdf')...
            ,'Resolution',500);
    else
         uiwait(f);
    end
    close all;
end
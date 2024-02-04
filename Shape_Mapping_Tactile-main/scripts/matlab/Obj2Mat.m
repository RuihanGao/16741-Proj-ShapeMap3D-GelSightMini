function [DataOut] = Obj2Mat(objname, swap)
%     vizObj(objname);
    [modelPath,modelName,~] = fileparts(objname);
    [objectPath,~,~] = fileparts(modelPath);

    [V,F] = read_vertices_and_faces_from_obj_file(objname);
    
    if (swap)
        V=[1 0 0;0 0 1; 0 -1 0]'*V';  % change the y and z axis
        V=V';
    end
    
    data = convert_vf_to_data( V, F );
    DataOut = data'; % N x D
    
    vertices = DataOut(:, 1:3); 
    normals = DataOut(:, 4:6); 
    faces = F;
    save(strcat(objectPath, '/', modelName, '.mat'), 'modelName', 'vertices', 'normals', 'faces');
    disp(strcat(objectPath, '/', modelName, '.mat'));
end
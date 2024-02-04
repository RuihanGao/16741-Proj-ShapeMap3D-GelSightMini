function vizObj(path)
    [V,F] = read_vertices_and_faces_from_obj_file(path);
    trisurf(F,V(:,1),V(:,2),V(:,3), 'EdgeColor', 'none');
    axis equal off
    view(3);
    shading interp % Make surface look smooth
    camlight; lighting phong % Shine light on surface
    colormap(gca,gray)
end



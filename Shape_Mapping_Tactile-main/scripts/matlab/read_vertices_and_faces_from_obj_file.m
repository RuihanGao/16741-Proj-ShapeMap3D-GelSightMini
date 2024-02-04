
function [V,F] = read_vertices_and_faces_from_obj_file(filename)
  % Reads a .obj mesh file and outputs the vertex and face list
  % assumes a 3D triangle mesh and ignores everything but:
  % v x y z and f i j k lines
  % Input:
  %  filename  string of obj file's path
  %
  % Output:
  %  V  number of vertices x 3 array of vertex positions
  %  F  number of faces x 3 array of face indices
  %
  V = zeros(0,3);
  F = zeros(0,3);
  vertex_index = 1;
  face_index = 1;
  fid = fopen(filename,'rt');
  line = fgets(fid);
  vcount = 0; fcount = 0;
  while ischar(line)
    vertex = sscanf(line,'v %f %f %f');
    face = sscanf(line,'x %d %d %d');   % this may need to be changed according to the file formate;
    
    face_long = sscanf(line,'f %d//%d %d//%d %d//%d');

    if(length(face_long) ~= 6)
    face_long = sscanf(line,'f %d/%d %d/%d %d/%d');
        if(length(face_long)<3)
        face_long = sscanf(line,'f %d/%d/%d %d/%d/%d %d/%d/%d');
        end   
    end
    
    
    % see if line is vertex command if so add to vertices
    if(size(vertex)>0)
      vcount = vcount + 1;
      V(vertex_index,:) = vertex;
      vertex_index = vertex_index+1;
    % see if line is simple face command if so add to faces
    elseif(size(face)>0)
      fcount = fcount + 1;
      F(face_index,:) = face;
      face_index = face_index+1;
    % see if line is a long face command if so add to faces
    elseif(size(face_long)>0)
      fcount = fcount + 1;
      % remove normal and texture indices
      if(length(face_long) == 6)
      face_long = face_long(1:2:end);
      elseif(length(face_long) == 9)
      face_long = face_long(1:3:end);
      else
         error('format of obj file need to check'); 
      end
      F(face_index,:) = face_long;
      face_index = face_index+1;
      
    else
%        fprintf('Ignored: %s',line);
    end

    line = fgets(fid);
    if ((mod(vcount, 10000) == 0) && vcount) || ((mod(fcount, 10000) == 0) && fcount)
        disp('vertex: ' + string(vcount) + ' face: ' + string(fcount))
    end
  end
  fclose(fid);
end


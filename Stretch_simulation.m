function Stretch_simulation

% Common parameters
L = 1; % Length 
W = 1; % Width 
d = 0.05; % displacement 
nu = 0.48; % Poisson's ratio

%-------------------------
% --- CASE 1: One vertex per edge
%-------------------------
fields_case1 = run_case([L/2 0; L/2 W], d, nu, L, W);

%-------------------------
% --- CASE 2:  Two vertices per edge
%-------------------------
fields_case2 = run_case([L/3 0; L*2/3 0; L/3 W; L*2/3 W], d, nu, L, W);


%-------------------------
% --- Plot both cases in one figure
%-------------------------
titles = {'\epsilon_{xx}', '\epsilon_{yy}', '\sigma_{xx}', '\sigma_{yy}'};

figure('Units','centimeters','Position',[2 2 28 14]);

for iCase = 1:2
    if iCase == 1
        fields = fields_case1;
    else
        fields = fields_case2;
    end

    for k = 1:4
        colOffset = (iCase-1)*2; 
        row = ceil(k/2); 
        col = mod(k-1,2) + 1 + colOffset;
        subplot(2,4,(row-1)*4 + col)

        pdeplot(fields.Mesh, ...
            XYData = fields.Data{k}, ...
            Deformation = fields.Displacement, ...
            DeformationScaleFactor=1, ...
            ColorMap="jet");
        axis equal tight
        title(titles{k}, 'Interpreter', 'tex', 'FontSize', 10)
        colorbar
    end
end

%-------------------------
% --- Add big labels for each half
%-------------------------
annotation('textbox', [0.06 0.94 0.48 0.05], ...
    'String', 'One clasping point', ...
    'FontSize', 12, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', ...
    'EdgeColor', 'none');

annotation('textbox', [0.48 0.94 0.48 0.05], ...
    'String', 'Two clasping points', ...
    'FontSize', 12, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', ...
    'EdgeColor', 'none');

%-------------------------
% --- Print center point values to console
%-------------------------
disp('--- Center point values at (L/2,W/2) ---');
disp('Case 1 (one clasping point):');
print_center_values(fields_case1,L,W);
disp('Case 2 (two clasping points):');
print_center_values(fields_case2,L,W);

end

%-------------------------
% --- Helper function to run a case
%-------------------------
function out = run_case(extraVertices, d, nu, L, W)

nodes = [0 0; L 0; L W; 0 W];
elements = [1 2 3; 1 3 4];
geometry = fegeometry(nodes, elements);

geometry = addVertex(geometry, "Coordinates", extraVertices);

model = femodel('AnalysisType', 'structuralStatic', 'Geometry', geometry);
model = generateMesh(model,'Hmax',0.05);
model.MaterialProperties = materialProperties(YoungsModulus=1,PoissonsRatio=nu);

nExtra = size(extraVertices,1);
if nExtra == 4
    model.VertexBC(5) = vertexBC(YDisplacement=-d,XDisplacement=0);
    model.VertexBC(6) = vertexBC(YDisplacement=-d, XDisplacement=0);
    model.VertexBC(7) = vertexBC(YDisplacement=d,XDisplacement=0);
    model.VertexBC(8) = vertexBC(YDisplacement=d, XDisplacement=0);
elseif nExtra == 2
    model.VertexBC(5) = vertexBC(YDisplacement=-d,XDisplacement=0);
    model.VertexBC(6) = vertexBC(YDisplacement=d,XDisplacement=0);
end

R = solve(model);

out.Mesh = R.Mesh;
out.Displacement = R.Displacement;
out.Data = {R.Strain.exx,R.Strain.eyy,R.Stress.sxx,R.Stress.syy};
out.Result = R;

end

%-------------------------
% --- Helper function to print center point values
%-------------------------
function print_center_values(fields,L,W)
pt = [L/2, W/2];
strainAtMidpoint = interpolateStrain(fields.Result, L/2, W/2)
stressAtMidpoint = interpolateStress(fields.Result, L/2, W/2)
max_sxx = max(fields.Result.Stress.sxx);
max_syy = max(fields.Result.Stress.syy);
max_sxy = max(fields.Result.Stress.sxy);

fprintf('Maximum stresses:\n');
fprintf('  sigma_xx = %.4f\n', max_sxx);
fprintf('  sigma_yy = %.4f\n', max_syy);
fprintf('  sigma_xy = %.4f\n', max_sxy);
end


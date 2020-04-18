data = load('./DataSet/Localization/participant1/103/U/envNoClutterscans.mat');
bins = load('./DataSet/Localization/participant1/109/U/range_bins.mat');
scan = load('./DataSet/Localization/participant1/109/U/t_stmp.mat');

[X,Y] = meshgrid(bins.Rbin_1033, scan.T_stmp_1033);
Z = data.envNoClutterscansV_1033;
surf(X, Y, Z, 'edgecolor', 'none');
xlabel('Fast Time/Range');
ylabel('Slow Time/Scan Number');
zlabel('Magnitude');
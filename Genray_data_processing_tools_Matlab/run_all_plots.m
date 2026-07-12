function result = run_all_plots(jsonlFile, outDir, suffix)
%RUN_ALL_PLOTS Generate the standard GENRAY figures from one JSONL file.
% Every figure is saved as both PNG and MATLAB FIG.
if nargin < 1 || isempty(jsonlFile), jsonlFile = 'genray_scan_from_nc.jsonl'; end
if nargin < 2 || isempty(outDir), outDir = fullfile(fileparts(jsonlFile), 'figs_matlab'); end
if nargin < 3, suffix = ''; end
if ~isfile(jsonlFile), error('JSONL not found: %s', jsonlFile); end
if ~isfolder(outDir), mkdir(outDir); end
R = read_jsonl_records(jsonlFile);
if isempty(R), error('JSONL contains no records.'); end

freq = arrayfun(@(r) getnum(r, {'frqncy','frequency','freq'}, NaN), R)';
mode = arrayfun(@(r) getnum(r, {'ioxm','mode'}, NaN), R)';
alpha = arrayfun(@(r) getnum(r, {'alfast','alpha'}, NaN), R)';
beta = arrayfun(@(r) getnum(r, {'betast','beta'}, NaN), R)';
ratio = arrayfun(@(r) getratio(r), R)';
rho = arrayfun(@(r) getnum(r, {'rho_peak','rhoPeak'}, NaN), R)';
cur = arrayfun(@(r) getnum(r, {'driven_current_ka','current_ka','parallel_cur_total'}, NaN), R)';
freqs = unique(freq(isfinite(freq)));

% 1) alpha-beta absorption ratio and rho peak surfaces
for f = freqs(:)'
  for m = unique(mode(freq==f & isfinite(mode)))'
    q = freq==f & mode==m & isfinite(alpha) & isfinite(beta);
    if ~any(q), continue; end
    fig = figure('Visible','off','Color','w','Position',[100 100 1200 500]);
    subplot(1,2,1); plot_surface(alpha(q),beta(q),ratio(q)); title(sprintf('%.3g GHz %s absorption',f,mode_name(m))); xlabel('\alpha'); ylabel('\beta'); colorbar;
    subplot(1,2,2); plot_surface(alpha(q),beta(q),rho(q)); title(sprintf('%.3g GHz %s rho peak',f,mode_name(m))); xlabel('\alpha'); ylabel('\beta'); colorbar;
    save_fig_png(fig, fullfile(outDir,sprintf('alpha_beta_rhoPeak_powRatio_%gGHz_%s%s',f,mode_name(m),suffix))); close(fig);
  end
end

% 2) O/X 2x2 comparison for each frequency
for f = freqs(:)'
  fig = figure('Visible','off','Color','w','Position',[100 100 1100 800]);
  modes = [1 -1];
  for row = 1:2
    q = freq==f & mode==modes(row) & isfinite(alpha) & isfinite(beta);
    subplot(2,2,2*row-1); if any(q), plot_surface(alpha(q),beta(q),ratio(q)); end
    title(sprintf('%.3g GHz %s absorption',f,mode_name(modes(row)))); xlabel('\alpha'); ylabel('\beta'); colorbar;
    subplot(2,2,2*row); if any(q), plot_surface(alpha(q),beta(q),rho(q)); end
    title(sprintf('%.3g GHz %s rho peak',f,mode_name(modes(row)))); xlabel('\alpha'); ylabel('\beta'); colorbar;
  end
  save_fig_png(fig, fullfile(outDir,sprintf('alpha_beta_freq_%gGHz_OX_2x2%s',f,suffix))); close(fig);
end

% 3) current contour maps (1 degree grid)
for f = freqs(:)'
  for m = unique(mode(freq==f & isfinite(mode)))'
    q = freq==f & mode==m & isfinite(alpha) & isfinite(beta) & isfinite(cur);
    if ~any(q), continue; end
    fig = figure('Visible','off','Color','w'); plot_surface(alpha(q),beta(q),cur(q)); hold on;
    ax = gca; ax.XMinorTick='on'; ax.YMinorTick='on'; ax.XAxis.MinorTickValues=floor(min(alpha(q))):ceil(max(alpha(q))); ax.YAxis.MinorTickValues=floor(min(beta(q))):ceil(max(beta(q))); grid on; ax.GridAlpha=.18; ax.MinorGridAlpha=.12; title(sprintf('%.3g GHz %s driven current',f,mode_name(m))); xlabel('\alpha'); ylabel('\beta'); colorbar;
    save_fig_png(fig, fullfile(outDir,sprintf('alpha_beta_current_drive_%gGHz_%s%s',f,mode_name(m),suffix))); close(fig);
  end
end

% 4) compact peak/current summary
fig = figure('Visible','off','Color','w'); scatter(freq,ratio,18,mode,'filled'); xlabel('Frequency (GHz)'); ylabel('Absorption ratio'); title('GENRAY absorption summary'); grid on; colorbar;
save_fig_png(fig, fullfile(outDir,['absorption_summary' suffix])); close(fig);
result = struct('records',numel(R),'outDir',outDir);
fprintf('Completed MATLAB plots: %d records -> %s\n',numel(R),outDir);
end

function plot_surface(x,y,z)
  q=isfinite(x)&isfinite(y)&isfinite(z); x=x(q); y=y(q); z=z(q);
  scatter(x,y,22,z,'filled'); hold on;
  if numel(x)>3
    try
      F=scatteredInterpolant(x,y,z,'natural','none');
      [X,Y]=meshgrid(linspace(min(x),max(x),81),linspace(min(y),max(y),81));
      surf(X,Y,F(X,Y),'EdgeColor','none','FaceAlpha',.75); view(2);
    catch
      % Keep the raw scatter plot when interpolation is underdetermined.
    end
  end
  hold off;
end
function s=mode_name(m), if m==0||m==1, s='Xm'; elseif m==2||m==-1, s='Om'; else, s=sprintf('mode%d',m); end, end
function v=getnum(r,names,default), v=default; for i=1:numel(names), if isfield(r,names{i}), x=r.(names{i}); if isnumeric(x)&&isscalar(x), v=double(x); elseif ischar(x)||isstring(x), v=str2double(x); end; if isfinite(v), return; end, end, end, end
function v=getratio(r)
v=getnum(r,{'pow_ratio','absorption_ratio','p_total_over_p_inj','pabs_ratio'},NaN);
if ~isfinite(v)
  p=getnum(r,{'power_total_1e10'},NaN); pin=getnum(r,{'power_inj_total_1e10'},NaN);
  if isfinite(p)&&isfinite(pin)&&pin~=0, v=p/pin; end
end
end

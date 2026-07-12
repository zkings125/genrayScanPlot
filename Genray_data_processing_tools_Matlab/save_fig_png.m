function save_fig_png(fig, baseName)
%SAVE_FIG_PNG Save editable FIG and raster PNG with the same stem.
savefig(fig,[baseName '.fig']);
exportgraphics(fig,[baseName '.png'],'Resolution',180);
end

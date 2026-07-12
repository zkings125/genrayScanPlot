function records = read_jsonl_records(fileName)
%READ_JSONL_RECORDS Read JSONL one line at a time (no NC data caching).
fid=fopen(fileName,'r'); if fid<0, error('Cannot open %s',fileName); end
c={}; cleaner=onCleanup(@()fclose(fid));
while true
  line=fgetl(fid); if ~ischar(line), break; end
  if isempty(strtrim(line)), continue; end
  try
    c{end+1}=jsondecode(line); %#ok<AGROW>
  catch ME
    warning(ME.identifier,'%s',ME.message);
  end
end
if isempty(c), records=struct([]); else, records=[c{:}]; end
end

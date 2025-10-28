# Additional Refactor Targets

1. **Fix SaveDialog export option injection order**  
   The helper `_apply_export_opts` builds ffmpeg command arguments by appending new flags with `list.extend`. When we add options like `-loop`, `-q:v`, or the fallback `-filter:v fps=…`, they are appended *after* the output pathname because the command list already ends with the destination file. ffmpeg treats anything after the output as another output specification, so setting a custom FPS currently breaks the encode step. Refactor the routine to compute the output argument index and insert/replace options *before* it, or rebuild the command with structured segments instead of list mutations. 【F:ui/home_page.py†L973-L1009】

2. **Normalize AddPage OSD key derivation**  
   `_show_selected` tries to derive the OSD key by splitting the file path on the hard-coded substring `"./animes\"`. This only succeeds on Windows-style paths and leaves full absolute paths on POSIX, so duplicate detection and the “already exists?” prompt silently fail on Linux/macOS. Switch to `os.path.relpath`/`os.path.basename` (possibly combined with the configured output directory) to generate a portable identifier. 【F:ui/add_page.py†L145-L170】

3. **Make duplicate-name suffixing predictable**  
   When a name collision occurs, `_duplicate` recursively appends `_c{num}` to the *current* string, which yields awkward keys like `name_c0_c1` if multiple duplicates exist. Extract the original stem and increment a numeric counter (`name_c1`, `name_c2`, …) or keep a per-stem counter so the naming scheme stays readable and deterministic. 【F:ui/add_page.py†L177-L180】


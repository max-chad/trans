from pathlib import Path

file_path = Path(r"c:\Users\max_chad\Music\transcriber\ui\main_window.py")
lines = file_path.read_text(encoding='utf-8').splitlines(keepends=True)

# We want to remove lines 227 to 421 (1-based).
# Indices to remove: 226 to 420 (inclusive).
# So we keep 0..225 and 421..end.

new_lines = lines[:226] + lines[421:]

file_path.write_text("".join(new_lines), encoding='utf-8')
print(f"Removed lines 227-421. New line count: {len(new_lines)}")

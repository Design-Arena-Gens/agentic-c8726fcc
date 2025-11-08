"""Generate a stlite-based static bundle for deployment."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


PYTHON_FILES = [
    Path("streamlit_app.py"),
    Path("app/__init__.py"),
    Path("app/data.py"),
    Path("app/model.py"),
    Path("app/simulation.py"),
    Path("app/analytics.py"),
]


def load_files() -> dict[str, str]:
    files: dict[str, str] = {}
    for file_path in PYTHON_FILES:
        files[str(file_path)] = file_path.read_text()
    return files


def build_index_html(files: dict[str, str]) -> str:
    files_json = json.dumps(files)
    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Wind Turbine Failure Lab</title>
    <style>
      html, body {{
        margin: 0;
        height: 100%;
        background: #0b172a;
        color: #fff;
        font-family: 'Inter', system-ui, sans-serif;
      }}
      #root {{
        height: 100%;
      }}
      header.banner {{
        background: linear-gradient(135deg, rgba(14,116,144,0.9), rgba(30,64,175,0.85));
        padding: 1.5rem;
        text-align: center;
        color: #f8fafc;
      }}
      header.banner h1 {{
        margin: 0;
        font-size: 1.8rem;
        letter-spacing: 0.04em;
      }}
      header.banner p {{
        margin: 0.5rem 0 0;
        font-size: 0.95rem;
        opacity: 0.85;
      }}
      .app-container {{
        height: calc(100% - 150px);
      }}
    </style>
  </head>
  <body>
    <header class="banner">
      <h1>Wind Turbine Failure Protection Lab</h1>
      <p>Interactive Streamlit experience running entirely in your browser via stlite.</p>
    </header>
    <div id="root" class="app-container"></div>
    <script type="module">
      import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/mountable@0.68.4/dist/stlite.mjs";

      const files = {files_json};

      mount(document.getElementById("root"), {{
        requirements: ["streamlit==1.38.0", "pandas==2.1.4", "numpy==1.26.4", "plotly==5.18.0", "pyyaml==6.0.1"],
        entrypoint: "streamlit_app.py",
        files
      }});
    </script>
  </body>
</html>
"""
    return textwrap.dedent(html)


def main() -> None:
    files = load_files()
    public_dir = Path("public")
    public_dir.mkdir(exist_ok=True)
    index_html = build_index_html(files)
    (public_dir / "index.html").write_text(index_html)


if __name__ == "__main__":
    main()

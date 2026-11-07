"""
Script to update notebooks with dynamic Colab / Local paths and execute them sequentially,
capturing all cell outputs, charts, and tables directly inside the .ipynb files.
"""

import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

COLAB_DRIVE_SNIPPET = '''# Setup for Google Colab (runs automatically if inside Colab)
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    if not Path('/content/drive').exists():
        drive.mount('/content/drive')
        print("Google Drive mounted at /content/drive")
'''

FIND_ROOT_SNIPPET_NB1 = '''from pathlib import Path
import re
import numpy as np
import pandas as pd
from IPython.display import display

# Automatic project root detection (works in local project root, notebooks directory, or Google Colab)
def find_project_root():
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path('/content/drive/MyDrive/project'),
        Path('/content/drive/MyDrive/Detecting Silent Failures of Uncertainty Estimators'),
        Path('/content/drive/MyDrive'),
        Path('/content'),
    ]
    for cand in candidates:
        if (cand / 'datasets' / 'commonsense_qa').exists() and (cand / 'datasets' / 'arc_challenge').exists():
            return cand
    for cand in candidates:
        if (cand / 'datasets').exists():
            return cand
    return Path.cwd()

ROOT = find_project_root()
DATASET_DIRS = {
    'CommonsenseQA': ROOT / 'datasets' / 'commonsense_qa',
    'ARC-Challenge': ROOT / 'datasets' / 'arc_challenge',
}
OUTPUT = ROOT / 'outputs'
OUTPUT.mkdir(parents=True, exist_ok=True)
print(f"Project ROOT: {ROOT}")
print(f"Output directory: {OUTPUT}")

for name, folder in DATASET_DIRS.items():
    files = sorted(folder.rglob('*.parquet')) if folder.exists() else []
    print(f'{name}: {len(files)} Parquet file(s) at {folder}')
    for f in files:
        print('  ', f.relative_to(folder), f'({f.stat().st_size/1024**2:.3f} MiB)')
    if not files:
        raise FileNotFoundError(f'No Parquet files for {name}. Update DATASET_DIRS: {folder}')
'''

FIND_ROOT_SNIPPET_NB2 = '''from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from IPython.display import display

# Automatic project root detection (works in local project root, notebooks directory, or Google Colab)
def find_project_root():
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path('/content/drive/MyDrive/project'),
        Path('/content/drive/MyDrive/Detecting Silent Failures of Uncertainty Estimators'),
        Path('/content/drive/MyDrive'),
        Path('/content'),
    ]
    for cand in candidates:
        if (cand / 'outputs' / 'numeric_features.csv').exists():
            return cand
    for cand in candidates:
        if (cand / 'datasets' / 'commonsense_qa').exists():
            return cand
    return Path.cwd()

ROOT = find_project_root()
OUTPUT = ROOT / 'outputs'
CHARTS = OUTPUT / 'charts'
CHARTS.mkdir(parents=True, exist_ok=True)
path = OUTPUT / 'numeric_features.csv'
if not path.exists():
    raise FileNotFoundError('First run 01_prepare_numeric_features.ipynb (numeric_features.csv missing)')
data = pd.read_csv(path)
FEATURES = ['question_words', 'question_chars', 'avg_option_words', 'option_count']
missing = set(['dataset', 'split', *FEATURES]) - set(data.columns)
if missing:
    raise ValueError(f'Missing columns in numeric_features.csv: {missing}')
print(f'Project ROOT: {ROOT}')
print('Loaded', len(data), 'rows')
display(data.groupby(['dataset', 'split']).size().rename('samples').to_frame())
'''

FIND_ROOT_SNIPPET_NB3 = '''from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Automatic project root detection (works in local project root, notebooks directory, or Google Colab)
def find_project_root():
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path('/content/drive/MyDrive/project'),
        Path('/content/drive/MyDrive/Detecting Silent Failures of Uncertainty Estimators'),
        Path('/content/drive/MyDrive'),
        Path('/content'),
    ]
    for cand in candidates:
        if (cand / 'outputs' / 'numeric_features.csv').exists():
            return cand
    for cand in candidates:
        if (cand / 'datasets' / 'commonsense_qa').exists():
            return cand
    return Path.cwd()

ROOT = find_project_root()
OUTPUT = ROOT / 'outputs'
CHARTS = OUTPUT / 'charts'
CHARTS.mkdir(parents=True, exist_ok=True)
path = OUTPUT / 'numeric_features.csv'
if not path.exists():
    raise FileNotFoundError('First run 01_prepare_numeric_features.ipynb')

data = pd.read_csv(path)
FEATURES = ['question_words', 'question_chars', 'avg_option_words', 'option_count']
FIT_ON_TRAIN_ONLY = True
print(f'Project ROOT: {ROOT}')
print('Rows:', len(data), '| train-only fitting:', FIT_ON_TRAIN_ONLY)
'''

FIND_ROOT_SNIPPET_NB4 = '''from pathlib import Path
import io, zipfile, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, balanced_accuracy_score,
                             classification_report)

# Automatic project root detection (works in local project root, notebooks directory, or Google Colab)
def find_project_root():
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path('/content/drive/MyDrive/project'),
        Path('/content/drive/MyDrive/Detecting Silent Failures of Uncertainty Estimators'),
        Path('/content/drive/MyDrive'),
        Path('/content'),
        Path('/mnt/data'),
    ]
    # 1. Check direct candidate paths
    for cand in candidates:
        if (cand / 'outputs' / 'numeric_features.csv').exists() or (cand / 'outputs.zip').exists():
            return cand
    # 2. Check subdirectories of /content and /content/drive/MyDrive (in case repo was cloned or stored in a subfolder)
    for parent_dir in [Path('/content'), Path('/content/drive/MyDrive')]:
        if parent_dir.exists():
            try:
                for sub in parent_dir.iterdir():
                    if sub.is_dir() and ((sub / 'outputs' / 'numeric_features.csv').exists() or (sub / 'outputs.zip').exists()):
                        return sub
            except Exception:
                pass
    # 3. Fallback to candidate paths with datasets
    for cand in candidates:
        if (cand / 'datasets' / 'commonsense_qa').exists():
            return cand
    return Path.cwd()

BASE_DIR = find_project_root()
csv_path = BASE_DIR / 'outputs' / 'numeric_features.csv'
zip_path = BASE_DIR / 'outputs.zip'

if csv_path.exists():
    df = pd.read_csv(csv_path)
elif zip_path.exists():
    with zipfile.ZipFile(zip_path) as archive:
        df = pd.read_csv(io.BytesIO(archive.read('outputs/numeric_features.csv')))
elif (Path.cwd() / 'numeric_features.csv').exists():
    df = pd.read_csv(Path.cwd() / 'numeric_features.csv')
    BASE_DIR = Path.cwd()
else:
    raise FileNotFoundError(
        f"فایل outputs/numeric_features.csv یا outputs.zip در مسیرهای بررسی شده پیدا نشد.\\n"
        f"مسیر فعلی: {Path.cwd()}\\n"
        f"مسیر بررسی شده: {BASE_DIR}\\n\\n"
        f"علت خطا و راه‌حل‌ها:\\n"
        f"1. اگر در Google Colab هستید، ابتدا نوت‌بوک 01_prepare_numeric_features.ipynb را اجرا کنید تا خروجی‌ها ایجاد شوند.\\n"
        f"2. یا فایل outputs.zip یا پوشه outputs را مستقیماً در Colab آپلود کنید.\\n"
        f"3. اگر فایل‌ها در Google Drive هستند، مطمئن شوید درایو Mount شده است و در صورت نیاز متغیر BASE_DIR را دستی آدرس‌دهی کنید:\\n"
        f"   BASE_DIR = Path('/content/drive/MyDrive/نام_پوشه_پروژه')\\n"
        f"4. اگر مخزن گیت را clone کرده‌اید، با دستور cd% وارد پوشه مخزن شوید: cd /content/Detecting...%"
    )

out = BASE_DIR / 'outputs' / 'classification'
out.mkdir(parents=True, exist_ok=True)
required = {'dataset', 'split', 'question_words', 'question_chars', 'avg_option_words', 'option_count'}
assert required.issubset(df.columns), f'فیلدهای مفقود: {required - set(df.columns)}'
print(f'Project ROOT: {BASE_DIR}')
print('Loaded:', len(df), 'rows')
print(pd.crosstab(df['dataset'], df['split']))
print('Results folder:', out)
'''

def update_notebook_cells(nb_path: Path, nb_id: int):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Check if first code cell has colab snippet, if not prepend it
    has_colab_cell = False
    for cell in nb.cells:
        if cell.cell_type == "code" and "IN_COLAB" in cell.source:
            has_colab_cell = True
            break

    if not has_colab_cell:
        colab_cell = nbformat.v4.new_code_cell(COLAB_DRIVE_SNIPPET)
        colab_cell.metadata = {"id": f"colab_mount_{nb_id}"}
        # Insert after the first markdown or pip cell
        nb.cells.insert(1, colab_cell)

    # Update path resolution snippet
    if nb_id == 1:
        snippet = FIND_ROOT_SNIPPET_NB1
    elif nb_id == 2:
        snippet = FIND_ROOT_SNIPPET_NB2
    elif nb_id == 3:
        snippet = FIND_ROOT_SNIPPET_NB3
    else:
        snippet = FIND_ROOT_SNIPPET_NB4

    # Find the cell with ROOT=Path.cwd() or ROOT = Path.cwd()
    updated = False
    for cell in nb.cells:
        if cell.cell_type == "code" and ("ROOT = Path.cwd()" in cell.source or "ROOT=Path.cwd()" in cell.source or "find_project_root" in cell.source or "PROJECT_ROOT" in cell.source):
            cell.source = snippet
            updated = True
            break

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"[{nb_path.name}] Updated Colab & ROOT path resolution (Updated: {updated})")


def run_notebook(nb_path: Path):
    print(f"\n==================================================")
    print(f"Executing: {nb_path.name}")
    print(f"==================================================")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"SUCCESS: {nb_path.name} executed and saved with cell outputs.")


if __name__ == "__main__":
    notebooks = [
        (NOTEBOOKS_DIR / "01_prepare_numeric_features.ipynb", 1),
        (NOTEBOOKS_DIR / "02_statistics_and_distributions.ipynb", 2),
        (NOTEBOOKS_DIR / "03_three_normalizations.ipynb", 3),
        (NOTEBOOKS_DIR / "04_pearson_bbn_classification.ipynb", 4),
    ]

    # 1. Update all notebooks with robust path resolution
    for nb_path, nb_id in notebooks:
        update_notebook_cells(nb_path, nb_id)

    # 2. Execute all notebooks sequentially
    for nb_path, nb_id in notebooks:
        run_notebook(nb_path)

    print("\nAll notebooks executed successfully.")

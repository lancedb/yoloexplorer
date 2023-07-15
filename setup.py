from pathlib import Path
from setuptools import find_packages, setup

# import pkg_resources as pkg

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / 'README.md').read_text(encoding='utf-8')
REQUIREMENTS = [
"lancedb",
"duckdb",
"scikit-learn",
"streamlit",
"streamlit-image-select",
"ultralytics@git+https://github.com/ultralytics/ultralytics.git@embeddings"
]

def get_version():
    return "0.0.1.dev1"


setup(
    name="yoloexplorer",
    version=get_version(),
    python_requires=">=3.8",
    description="",
    # long_description=README,
    install_requires=REQUIREMENTS,
    long_description_content_type="text/markdown",
    author="dev@lance",
    author_email="contact@lancedb.com",
    packages=find_packages(),  # required
    include_package_data=True,
)

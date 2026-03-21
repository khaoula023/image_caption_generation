import os 
from pathlib import Path
import logging

logging.basicConfig(
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

list_of_files = [
    # CI / Repo
    ".github/workflows/ci.yaml",
    ".gitignore",
    "README.md",
    
    # Core Package
    "src/__init__.py",
    "src/logger.py",
    "src/exception.py",

    # Components (modular pipeline)
    "src/components/__init__.py",
    "src/components/data_loader.py",
    "src/components/preprocessing.py",
    "src/components/evaluator.py",

    # Pipeline
    "src/pipeline/__init__.py",
    "src/pipeline/train_pipeline.py",
    "src/pipeline/eval_pipeline.py",

    # Utilities
    "src/utils/__init__.py",
    "src/utils/common.py",
    "src/utils/metrics.py",
    "src/utils/visualization.py",

    # Configuration
    "src/config/__init__.py",
    "src/config/configuration.py",

    # Entities / Schemas
    "src/entity/__init__.py",
    "src/entity/artifacts_entity.py",

    # Constants
    "src/constants/__init__.py",

    # Research / Experiments
    "research/experiments.ipynb",
    "research/analysis.ipynb",

    # Deployment
    "Dockerfile",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")
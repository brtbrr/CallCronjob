import json
import os
from pathlib import Path
from typing import Any


def get_project_dir(team: str, project: str) -> Path:
    project_path = Path('dags') / 'repo' / team / project
    if project_path.exists():
        return project_path

    return Path(os.getcwd())


def load() -> dict[str, Any]:
    project_dir = get_project_dir(team='******', project='*******')
    config_path = project_dir / 'models' / 'config.json'
    with open(config_path.absolute(), 'r') as read_file:
        return json.load(read_file)

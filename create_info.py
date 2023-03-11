import pandas as pd
from pathlib import Path

entries = []
for path in [Path("Testing"), Path("TrainingValidation")]:
    for filepath in path.glob('*.csv'):
        filename = filepath.name
        split_by_underscore = filename.split("_")
        label = "_".join(split_by_underscore[:-2])
        split = str(filepath.parent)
        entry = {
            "filename": filename,
            "label": label,
            "split": split
        }
        entries.append(entry)
result = pd.DataFrame(entries)
result.to_csv("train_validation_info30.csv")
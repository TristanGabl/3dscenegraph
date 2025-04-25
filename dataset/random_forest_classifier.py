import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1) LOAD (assumes one JSON/sample per line in data.jsonl)
rows = []
with open('helper_repos/3RScan/relationship_training_data.jsonl', 'r') as f:
    for line in f:
        sample = json.loads(line)
        src = sample['input']['source']
        tgt = sample['input']['target']
        # flatten into one dict
        dist = np.hypot(np.hypot(src['x'] - tgt['x'], src['y'] - tgt['y']), src['z'] - tgt['z'])
        if dist > 100:
            continue
        feat = {
            'src_name':   src['name'],
            'tgt_name':   tgt['name'],
            'dx':         src['x'] - tgt['x'],
            'dy':         src['y'] - tgt['y'],
            'dz':         src['z'] - tgt['z'],
            'dist':       dist,
            'dsize_x':    src['size_x'] - tgt['size_x'],
            'dsize_y':    src['size_y'] - tgt['size_y'],
            'dsize_z':    src['size_z'] - tgt['size_z'],
            'output':     sample['output']
        }
        rows.append(feat)

df = pd.DataFrame(rows[:30000])
unique_src_names = df['src_name'].unique()
print("Unique source names:", unique_src_names)

# 2) SPLIT
X = df.drop('output', axis=1)
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1, 
                                                    random_state=42,
                                                    stratify=y)

# 3) PREPROCESSING: one-hot on names, scale numeric diffs
cat_cols = ['src_name', 'tgt_name']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer([
    ('ohe',    OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('scale',  StandardScaler(),                   num_cols),
])

# 4) PIPELINE & TRAIN
clf = Pipeline([
    ('prep', preprocessor),
    ('rf',   RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)),
])
clf.fit(X_train, y_train)

# 5) EVALUATE
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

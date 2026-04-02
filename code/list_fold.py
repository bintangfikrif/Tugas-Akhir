import pandas as pd
from sklearn.model_selection import GroupKFold
from config import Config

df = pd.read_csv('label/labels.csv')
if 'subject_id' not in df.columns:
    df['subject_id'] = df['filename'].apply(lambda x: x.split('-')[0])

gkf = GroupKFold(n_splits=Config.N_SPLITS)
for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['label'], groups=df['subject_id'])):
    train_subjs = sorted(df.loc[train_idx, 'subject_id'].unique())
    val_subjs = sorted(df.loc[val_idx, 'subject_id'].unique())
    print(f'Fold {fold}:')
    print('  train:', train_subjs)
    print('  val:  ', val_subjs)
    print()
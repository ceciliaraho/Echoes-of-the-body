def assign_labels(df, labels_dict):
   
    df = df.copy()
    df['label'] = 'unlabeled'

    for label, (start, end) in labels_dict.items():
        mask = (df['time_from_start'] >= start) & (df['time_from_start'] < end)
        df.loc[mask, 'label'] = label

    return df

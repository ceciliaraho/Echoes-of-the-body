# labeling.py

def assign_labels(df, labels_dict):
    """
    Aggiunge una colonna 'label' al DataFrame in base ai secondi indicati nel dizionario.

    Parametri:
        df (pd.DataFrame): deve contenere la colonna 'time_from_start'
        labels_dict (dict): chiavi = nome fase, valori = (start_time, end_time) in secondi

    Ritorna:
        df con colonna 'label'
    """
    df = df.copy()
    df['label'] = 'unlabeled'

    for label, (start, end) in labels_dict.items():
        mask = (df['time_from_start'] >= start) & (df['time_from_start'] < end)
        df.loc[mask, 'label'] = label

    return df

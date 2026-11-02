"""Merge VAD outputs with image annotations and build the final long and wide datasets."""
from pathlib import Path
import re
import sys
import pandas as pd
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / 'merged_outputs'
QWEN_FILE = PROJECT_DIR / 'output_qwen3.8-27B_vad_stats' / 'qwen3.8-27B_vad_stats_ALL.csv'
OBLITERATED_FILE = PROJECT_DIR / 'output_Qwen3.8-27B-OBLITERATED_vad_stats' / 'qwen3.8-27B-OBLITERATED_vad_stats_ALL.csv'
ANNOTATION_FILE = PROJECT_DIR / 'image_annotations.csv'
EXPECTED_ANNOTATIONS = 1500
EXPECTED_QWEN_ROWS = 411713
EXPECTED_OBLITERATED_ROWS = 412225
AI_IMAGE_PATTERN = re.compile('_(?:Negativo|Positivo|Neutro)_\\d+\\.(?:jpg|jpeg|png|webp|avif)$', re.IGNORECASE)
VAD_COLUMNS = ['vad_valence_mean', 'vad_valence_std', 'vad_valence_median', 'vad_valence_max', 'vad_arousal_mean', 'vad_arousal_std', 'vad_arousal_median', 'vad_arousal_max', 'vad_dominance_mean', 'vad_dominance_std', 'vad_dominance_median', 'vad_dominance_max']

def normalize_filename(value):
    """
    Estrae solo il nome del file da un eventuale path.
    Funziona sia con slash Windows sia Linux.
    """
    if pd.isna(value):
        return ''
    text = str(value).strip().strip('"').strip("'")
    if not text:
        return ''
    parts = re.split('[\\\\/]', text)
    return parts[-1].strip()

def normalize_text(value):
    """
    Normalizzazione usata per le chiavi di merge.
    """
    return str(value).strip().casefold()

def normalize_gender(value):
    """
    Normalizza il genere percepito in M/F.
    """
    text = str(value).strip().upper()
    mapping = {'M': 'M', 'MALE': 'M', 'MAN': 'M', 'UOMO': 'M', 'F': 'F', 'FEMALE': 'F', 'WOMAN': 'F', 'DONNA': 'F'}
    return mapping.get(text, text)

def normalize_race(value):
    """
    Normalizza l'appartenenza razziale percepita in W/NW.
    """
    text = str(value).strip().upper().replace(' ', '').replace('-', '').replace('_', '')
    mapping = {'W': 'W', 'WHITE': 'W', 'NW': 'NW', 'NONWHITE': 'NW', 'NOTWHITE': 'NW'}
    return mapping.get(text, text)

def classify_image_source(filename):
    """
    Regola definitiva:

    - filename con suffisso *_Negativo_N / *_Positivo_N / *_Neutro_N
      => AI

    - qualsiasi altro filename
      => REAL
    """
    if AI_IMAGE_PATTERN.search(filename):
        return 'AI'
    return 'REAL'

def ensure_file_exists(path, label):
    if not path.exists():
        raise FileNotFoundError(f'File non trovato per {label}:\n{path}')

def save_csv(df, path):
    """
    Salva CSV con separatore ; e UTF-8 BOM.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=';', index=False, encoding='utf-8-sig')

def load_annotations():
    print()
    print('======================================')
    print(' ANNOTAZIONI IMMAGINI')
    print('======================================')
    print(f'File annotazioni: {ANNOTATION_FILE}')
    ensure_file_exists(ANNOTATION_FILE, 'annotazioni immagini')
    df = pd.read_csv(ANNOTATION_FILE, sep=None, engine='python', dtype=str, keep_default_na=False, encoding='utf-8-sig')
    required_columns = ['filename', 'relative_path', 'perceived_gender', 'perceived_racial_group']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError('Nel file delle annotazioni mancano queste colonne:\n' + '\n'.join((f'  - {col}' for col in missing_columns)))
    ann = df[['filename', 'relative_path', 'perceived_gender', 'perceived_racial_group']].copy()
    ann['image_filename'] = ann['filename'].apply(normalize_filename)
    ann['gender_perceived'] = ann['perceived_gender'].apply(normalize_gender)
    ann['race_perceived'] = ann['perceived_racial_group'].apply(normalize_race)
    ann['image_source'] = ann['image_filename'].apply(classify_image_source)
    normalized_paths = ann['relative_path'].astype(str).str.replace('\\', '/', regex=False).str.strip('/')
    path_parts = normalized_paths.str.split('/')
    path_lengths = path_parts.str.len()
    if (path_lengths < 4).any():
        problematic = ann.loc[path_lengths < 4, ['relative_path']]
        raise ValueError(f'Alcuni relative_path non contengono almeno categoria/professione/sentiment/filename:\n{problematic.head(20).to_string(index=False)}')
    ann['annotation_category'] = path_parts.str[-4].str.strip()
    ann['annotation_profession'] = path_parts.str[-3].str.strip()
    ann['annotation_sentiment'] = path_parts.str[-2].str.strip()
    ann['merge_category'] = ann['annotation_category'].apply(normalize_text)
    ann['merge_sentiment'] = ann['annotation_sentiment'].apply(normalize_text)
    merge_keys = ['merge_category', 'merge_sentiment', 'image_filename']
    empty_filename = (ann['image_filename'].str.strip() == '').sum()
    if empty_filename > 0:
        raise ValueError(f'Ci sono {empty_filename} annotazioni senza filename.')
    duplicated = ann.duplicated(subset=merge_keys, keep=False)
    if duplicated.any():
        problematic = ann.loc[duplicated, merge_keys + ['annotation_profession', 'relative_path']]
        raise ValueError(f'La chiave categoria + sentiment + filename non è univoca.\n\n{problematic.head(30).to_string(index=False)}')
    invalid_gender = ~ann['gender_perceived'].isin(['M', 'F'])
    if invalid_gender.any():
        values = ann.loc[invalid_gender, 'gender_perceived'].value_counts()
        raise ValueError(f'Sono presenti valori non validi per gender_perceived:\n{values.to_string()}')
    invalid_race = ~ann['race_perceived'].isin(['W', 'NW'])
    if invalid_race.any():
        values = ann.loc[invalid_race, 'race_perceived'].value_counts()
        raise ValueError(f'Sono presenti valori non validi per race_perceived:\n{values.to_string()}')
    if len(ann) != EXPECTED_ANNOTATIONS:
        raise ValueError(f'Numero annotazioni inatteso: {len(ann):,}. Attese: {EXPECTED_ANNOTATIONS:,}.')
    print()
    print(f'Annotazioni caricate: {len(ann):,}')
    print('Filename unici:', f"{ann['image_filename'].nunique():,}")
    print('Chiavi immagine uniche:', f'{ann[merge_keys].drop_duplicates().shape[0]:,}')
    print()
    print('Distribuzione genere percepito:')
    print(ann['gender_perceived'].value_counts().to_string())
    print()
    print('Distribuzione appartenenza razziale percepita:')
    print(ann['race_perceived'].value_counts().to_string())
    print()
    print('Distribuzione AI / REAL:')
    print(ann['image_source'].value_counts().to_string())
    return ann

def load_vad_dataset(path, model_label):
    ensure_file_exists(path, model_label)
    print()
    print('Caricamento:')
    print(path)
    df = pd.read_csv(path, sep=';', low_memory=False, encoding='utf-8-sig')
    required_columns = ['image', 'category', 'profession', 'sentiment', 'combination_id'] + VAD_COLUMNS
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f'Nel dataset {model_label} mancano queste colonne:\n' + '\n'.join((f'  - {col}' for col in missing)))
    return df

def annotate_model_dataset(input_path, annotations, model_label, expected_rows):
    print()
    print('======================================')
    print(f' {model_label}')
    print('======================================')
    df = load_vad_dataset(input_path, model_label)
    print(f'Righe originali: {len(df):,}')
    if len(df) != expected_rows:
        raise ValueError(f'Numero righe inatteso per {model_label}: {len(df):,}. Attese: {expected_rows:,}.')
    df['image_filename'] = df['image'].apply(normalize_filename)
    df['merge_category'] = df['category'].apply(normalize_text)
    df['merge_sentiment'] = df['sentiment'].apply(normalize_text)
    annotation_columns = ['merge_category', 'merge_sentiment', 'image_filename', 'gender_perceived', 'race_perceived', 'image_source']
    merged = df.merge(annotations[annotation_columns], on=['merge_category', 'merge_sentiment', 'image_filename'], how='left', validate='many_to_one')
    if len(merged) != len(df):
        raise RuntimeError(f'Il numero di righe è cambiato durante il merge di {model_label}: {len(df):,} -> {len(merged):,}')
    missing_annotations = merged[['gender_perceived', 'race_perceived', 'image_source']].isna().any(axis=1)
    missing_count = int(missing_annotations.sum())
    print('Righe senza annotazione:', f'{missing_count:,}')
    if missing_count > 0:
        missing_images = merged.loc[missing_annotations, ['category', 'profession', 'sentiment', 'image_filename']].drop_duplicates()
        print()
        print('Immagini senza annotazione:')
        print(missing_images.head(50).to_string(index=False))
        raise ValueError(f'{model_label}: merge incompleto. {missing_count:,} righe non hanno annotazioni.')
    duplicated = merged.duplicated(subset=['image', 'combination_id']).sum()
    print('Duplicati image + combination_id:', f'{duplicated:,}')
    if duplicated > 0:
        raise ValueError(f'{model_label}: trovati {duplicated:,} duplicati su image + combination_id.')
    merged['model_variant'] = model_label
    merged = merged.drop(columns=['merge_category', 'merge_sentiment'])
    print('Immagini rappresentate:', f"{merged['image'].nunique():,}")
    print()
    print('Genere percepito nelle righe:')
    print(merged['gender_perceived'].value_counts().to_string())
    print()
    print('Appartenenza razziale percepita nelle righe:')
    print(merged['race_perceived'].value_counts().to_string())
    print()
    print('AI / REAL nelle righe:')
    print(merged['image_source'].value_counts().to_string())
    return merged

def build_long_dataset(qwen_df, obliterated_df):
    print()
    print('======================================')
    print(' CREAZIONE DATASET LONG')
    print('======================================')
    long_df = pd.concat([qwen_df, obliterated_df], axis=0, ignore_index=True)
    expected = len(qwen_df) + len(obliterated_df)
    if len(long_df) != expected:
        raise RuntimeError('Errore nella concatenazione LONG.')
    print('Righe Qwen:', f'{len(qwen_df):,}')
    print('Righe Obliterated:', f'{len(obliterated_df):,}')
    print('Righe LONG:', f'{len(long_df):,}')
    return long_df

def build_wide_dataset(qwen_df, obliterated_df):
    print()
    print('======================================')
    print(' CREAZIONE DATASET WIDE')
    print('======================================')
    keys = ['image', 'combination_id']
    q_dup = qwen_df.duplicated(subset=keys).sum()
    o_dup = obliterated_df.duplicated(subset=keys).sum()
    if q_dup > 0:
        raise ValueError(f'Qwen ha {q_dup:,} duplicati nelle chiavi del merge WIDE.')
    if o_dup > 0:
        raise ValueError(f'Obliterated ha {o_dup:,} duplicati nelle chiavi del merge WIDE.')
    shared_columns = ['image', 'image_filename', 'category', 'profession', 'sentiment', 'combination_id', 'system_prompt_id', 'dimension', 'system_condition', 'target_group', 'orientation', 'system_language', 'system_prompt', 'prompt_id', 'prompt_condition', 'paraphrase', 'prompt_language', 'user_prompt', 'occupation', 'decoding', 'temperature', 'top_p', 'seed', 'gender_perceived', 'race_perceived', 'image_source']
    shared_columns = [col for col in shared_columns if col in qwen_df.columns and col in obliterated_df.columns]
    qwen_columns = shared_columns + ['response'] + VAD_COLUMNS
    qwen_part = qwen_df[qwen_columns].copy()
    qwen_rename = {'response': 'qwen_response'}
    for col in VAD_COLUMNS:
        qwen_rename[col] = f'qwen_{col}'
    qwen_part = qwen_part.rename(columns=qwen_rename)
    obl_columns = keys + ['response'] + VAD_COLUMNS
    obl_part = obliterated_df[obl_columns].copy()
    obl_rename = {'response': 'obliterated_response'}
    for col in VAD_COLUMNS:
        obl_rename[col] = f'obliterated_{col}'
    obl_part = obl_part.rename(columns=obl_rename)
    wide = qwen_part.merge(obl_part, on=keys, how='inner', validate='one_to_one')
    qwen_keys = qwen_df[keys].drop_duplicates()
    obl_keys = obliterated_df[keys].drop_duplicates()
    qwen_only = qwen_keys.merge(obl_keys, on=keys, how='left', indicator=True)
    qwen_only_count = int((qwen_only['_merge'] == 'left_only').sum())
    obl_only = obl_keys.merge(qwen_keys, on=keys, how='left', indicator=True)
    obl_only_count = int((obl_only['_merge'] == 'left_only').sum())
    print('Osservazioni comuni:', f'{len(wide):,}')
    print('Presenti solo in Qwen:', f'{qwen_only_count:,}')
    print('Presenti solo in Obliterated:', f'{obl_only_count:,}')
    return (wide, qwen_only_count, obl_only_count)

def build_summary(annotations, qwen_df, obliterated_df, long_df, wide_df, qwen_only, obliterated_only):
    summary = pd.DataFrame([{'annotations_images': len(annotations), 'annotations_unique_filenames': annotations['image_filename'].nunique(), 'gender_M_images': int((annotations['gender_perceived'] == 'M').sum()), 'gender_F_images': int((annotations['gender_perceived'] == 'F').sum()), 'race_W_images': int((annotations['race_perceived'] == 'W').sum()), 'race_NW_images': int((annotations['race_perceived'] == 'NW').sum()), 'AI_images': int((annotations['image_source'] == 'AI').sum()), 'REAL_images': int((annotations['image_source'] == 'REAL').sum()), 'qwen_rows': len(qwen_df), 'qwen_images': qwen_df['image'].nunique(), 'obliterated_rows': len(obliterated_df), 'obliterated_images': obliterated_df['image'].nunique(), 'long_rows': len(long_df), 'wide_common_rows': len(wide_df), 'qwen_only_rows': qwen_only, 'obliterated_only_rows': obliterated_only}])
    return summary

def main():
    print()
    print('======================================')
    print(' MERGE FINALE DATASET')
    print(' QWEN + OBLITERATED + ANNOTAZIONI')
    print('======================================')
    annotations = load_annotations()
    qwen = annotate_model_dataset(QWEN_FILE, annotations, 'Qwen3.8-27B', EXPECTED_QWEN_ROWS)
    obliterated = annotate_model_dataset(OBLITERATED_FILE, annotations, 'Qwen3.8-27B-OBLITERATED', EXPECTED_OBLITERATED_ROWS)
    print()
    print('======================================')
    print(' SALVATAGGIO DATASET ANNOTATI')
    print('======================================')
    qwen_output = OUTPUT_DIR / 'qwen3.8-27B_vad_stats_ALL_annotated.csv'
    obliterated_output = OUTPUT_DIR / 'qwen3.8-27B-OBLITERATED_vad_stats_ALL_annotated.csv'
    save_csv(qwen, qwen_output)
    save_csv(obliterated, obliterated_output)
    print(f'Qwen annotato:\n{qwen_output}')
    print()
    print(f'Obliterated annotato:\n{obliterated_output}')
    long_df = build_long_dataset(qwen, obliterated)
    long_output = OUTPUT_DIR / 'merged_long_qwen_obliterated.csv'
    save_csv(long_df, long_output)
    print()
    print(f'LONG salvato:\n{long_output}')
    wide_df, qwen_only, obliterated_only = build_wide_dataset(qwen, obliterated)
    wide_output = OUTPUT_DIR / 'merged_wide_qwen_obliterated.csv'
    save_csv(wide_df, wide_output)
    print()
    print(f'WIDE salvato:\n{wide_output}')
    summary = build_summary(annotations, qwen, obliterated, long_df, wide_df, qwen_only, obliterated_only)
    summary_output = OUTPUT_DIR / 'merge_summary.csv'
    save_csv(summary, summary_output)
    print()
    print('======================================')
    print(' RIEPILOGO')
    print('======================================')
    print(summary.T.to_string(header=False))
    print()
    print(f'Riepilogo salvato:\n{summary_output}')
    print()
    print('======================================')
    print(' CONTROLLI FINALI')
    print('======================================')
    print('Annotazioni mancanti Qwen:', int(qwen[['gender_perceived', 'race_perceived', 'image_source']].isna().any(axis=1).sum()))
    print('Annotazioni mancanti Obliterated:', int(obliterated[['gender_perceived', 'race_perceived', 'image_source']].isna().any(axis=1).sum()))
    print('NaN VAD Qwen:', int(qwen[VAD_COLUMNS].isna().sum().sum()))
    print('NaN VAD Obliterated:', int(obliterated[VAD_COLUMNS].isna().sum().sum()))
    print()
    print('======================================')
    print(' MERGE COMPLETATO CORRETTAMENTE')
    print('======================================')
    print()
    print('File prodotti:')
    print(f'  1. {qwen_output}')
    print(f'  2. {obliterated_output}')
    print(f'  3. {long_output}')
    print(f'  4. {wide_output}')
    print(f'  5. {summary_output}')
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print()
        print('======================================')
        print(' ERRORE')
        print('======================================')
        print(exc)
        sys.exit(1)

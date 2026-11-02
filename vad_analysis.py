"""Compute NRC-VAD statistics for model responses using spaCy tokenization and surface/lemma lexicon lookup."""
from pathlib import Path
import argparse
import csv
import json
import math
import os
from statistics import median
import pandas as pd
import spacy
from tqdm import tqdm
PIPELINE_VERSION = 2
PROJECT_DIR = Path(__file__).resolve().parent
DATASETS = {'qwen': {'name': 'Qwen3.8-27B', 'input_dir': PROJECT_DIR / 'output_qwen3.8-27B', 'output_dir': PROJECT_DIR / 'output_qwen3.8-27B_vad_stats', 'final_filename': 'qwen3.8-27B_vad_stats_ALL.csv'}, 'obliterated': {'name': 'Qwen3.8-27B-OBLITERATED', 'input_dir': PROJECT_DIR / 'output_Qwen3.8-27B-OBLITERATED', 'output_dir': PROJECT_DIR / 'output_Qwen3.8-27B-OBLITERATED_vad_stats', 'final_filename': 'qwen3.8-27B-OBLITERATED_vad_stats_ALL.csv'}}
LEXICON_CANDIDATES = [PROJECT_DIR / 'NRC-VAD-Lexicon' / 'NRC-VAD-Lexicon' / 'NRC-VAD-Lexicon.txt', PROJECT_DIR / 'NRC-VAD-Lexicon' / 'NRC-VAD-Lexicon.txt']
TEXT_COLUMN = 'response'
ERROR_PREFIX = 'ERRORE:'
CHUNK_SIZE = 5000
SPACY_BATCH_SIZE = 512
SPACY_MODEL = 'en_core_web_sm'
EXPECTED_CSV_COUNT = 150
EXPECTED_SOURCE_ROWS = 412500
VAD_COLUMNS = ['vad_valence_mean', 'vad_valence_std', 'vad_valence_median', 'vad_valence_max', 'vad_arousal_mean', 'vad_arousal_std', 'vad_arousal_median', 'vad_arousal_max', 'vad_dominance_mean', 'vad_dominance_std', 'vad_dominance_median', 'vad_dominance_max']

def parse_arguments():
    """
    Permette di scegliere quale dataset elaborare.

    Esempi:
        python vad_analysis.py qwen
        python vad_analysis.py obliterated
    """
    parser = argparse.ArgumentParser(description='Calcola le statistiche NRC-VAD per uno dei dataset Qwen.')
    parser.add_argument('dataset', choices=DATASETS.keys(), help="Dataset da elaborare: 'qwen' oppure 'obliterated'.")
    return parser.parse_args()

def find_lexicon_path():
    """
    Restituisce il percorso del NRC-VAD inglese completo.
    """
    for candidate in LEXICON_CANDIDATES:
        if candidate.exists():
            return candidate
    expected = '\n'.join((f'  - {path}' for path in LEXICON_CANDIDATES))
    raise FileNotFoundError(f'\nNon trovo NRC-VAD-Lexicon.txt.\nHo controllato questi percorsi:\n{expected}\n')

def write_json_atomic(path: Path, data: dict):
    """
    Scrive un file JSON tramite file temporaneo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)

def qc_path_for(output_dir: Path, relative_csv: Path):
    """
    Percorso del file QC associato a un CSV.
    """
    return output_dir / '_qc' / relative_csv.with_suffix('.json')

def load_vad_lexicon(path: Path):
    """
    Carica il NRC-VAD Lexicon v1 (2018).

    Il file contiene:
        Word
        Valence
        Arousal
        Dominance
    """
    print()
    print('Caricamento NRC-VAD:')
    print(path)
    df = pd.read_csv(path, sep='\t', header=None, names=['Word', 'Valence', 'Arousal', 'Dominance'], encoding='utf-8')
    df['Word'] = df['Word'].astype(str).str.strip().str.casefold()
    for column in ['Valence', 'Arousal', 'Dominance']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=['Word', 'Valence', 'Arousal', 'Dominance'])
    for column in ['Valence', 'Arousal', 'Dominance']:
        minimum = df[column].min()
        maximum = df[column].max()
        if minimum < 0 or maximum > 1:
            raise ValueError(f"La colonna {column} contiene valori fuori dall'intervallo 0-1. Controlla il lessico selezionato.")
    lexicon = {row.Word: (float(row.Valence), float(row.Arousal), float(row.Dominance)) for row in df.itertuples(index=False)}
    if not lexicon:
        raise ValueError('Il NRC-VAD risulta vuoto.')
    print(f'NRC-VAD caricato correttamente: {len(lexicon):,} parole')
    return lexicon

def lookup_token(token, lexicon):
    """
    Cerca il token nel NRC-VAD.

    1. Forma superficiale.
    2. Lemma spaCy come fallback.
    3. Nessun match -> token ignorato.
    """
    if not token.is_alpha:
        return None
    surface = token.text.casefold().strip()
    if surface in lexicon:
        return lexicon[surface]
    lemma = token.lemma_.casefold().strip()
    if lemma and lemma != surface and (lemma in lexicon):
        return lexicon[lemma]
    return None

def calculate_statistics(values):
    """
    Calcola:
        media
        deviazione standard della popolazione
        mediana
        massimo
    """
    n = len(values)
    mean_value = sum(values) / n
    variance = sum(((value - mean_value) ** 2 for value in values)) / n
    std_value = math.sqrt(variance)
    median_value = median(values)
    max_value = max(values)
    return (mean_value, std_value, median_value, max_value)

def score_document(doc, lexicon):
    """
    Calcola le 12 statistiche VAD della risposta:

        4 Valence
        4 Arousal
        4 Dominance

    Restituisce anche i conteggi utilizzati nel QC.
    """
    valence_values = []
    arousal_values = []
    dominance_values = []
    matched_tokens = 0
    alphabetic_tokens = 0
    for token in doc:
        if token.is_alpha:
            alphabetic_tokens += 1
        values = lookup_token(token, lexicon)
        if values is None:
            continue
        valence, arousal, dominance = values
        valence_values.append(valence)
        arousal_values.append(arousal)
        dominance_values.append(dominance)
        matched_tokens += 1
    if matched_tokens == 0:
        return (*[float('nan')] * 12, 0, alphabetic_tokens)
    valence_stats = calculate_statistics(valence_values)
    arousal_stats = calculate_statistics(arousal_values)
    dominance_stats = calculate_statistics(dominance_values)
    return (*valence_stats, *arousal_stats, *dominance_stats, matched_tokens, alphabetic_tokens)

def output_is_complete(output_file: Path, qc_file: Path):
    """
    Un file viene considerato già elaborato soltanto se:

    - esiste il CSV;
    - esiste il QC;
    - il QC appartiene alla pipeline attuale;
    - è stato usato lo stesso filtro degli errori;
    - sono presenti tutte le colonne VAD.
    """
    if not output_file.exists() or not qc_file.exists():
        return False
    try:
        with qc_file.open('r', encoding='utf-8') as handle:
            qc = json.load(handle)
        if qc.get('pipeline_version') != PIPELINE_VERSION:
            return False
        if qc.get('error_prefix') != ERROR_PREFIX:
            return False
        columns = pd.read_csv(output_file, sep=';', nrows=0, encoding='utf-8-sig').columns
        return all((column in columns for column in VAD_COLUMNS))
    except Exception:
        return False

def load_qc(qc_file: Path):
    """
    Carica il report QC di un file già elaborato.
    """
    with qc_file.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def process_csv(input_file: Path, output_file: Path, qc_file: Path, nlp, lexicon):
    """
    Elabora un CSV.

    Le righe in cui response inizia con "ERRORE:"
    vengono escluse prima di spaCy e non entrano
    nel dataset VAD finale.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    qc_file.parent.mkdir(parents=True, exist_ok=True)
    input_columns = list(pd.read_csv(input_file, sep=';', nrows=0, encoding='utf-8-sig').columns)
    if TEXT_COLUMN not in input_columns:
        raise KeyError(f"La colonna '{TEXT_COLUMN}' non è presente in:\n{input_file}")
    temporary_file = output_file.with_suffix(output_file.suffix + '.tmp')
    if temporary_file.exists():
        temporary_file.unlink()
    first_chunk = True
    input_rows = 0
    valid_rows = 0
    skipped_error_rows = 0
    total_matched = 0
    total_alphabetic = 0
    rows_without_matches = 0
    reader = pd.read_csv(input_file, sep=';', chunksize=CHUNK_SIZE, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    for chunk in reader:
        input_rows += len(chunk)
        responses = chunk[TEXT_COLUMN].fillna('').astype(str)
        error_mask = responses.str.strip().str.startswith(ERROR_PREFIX, na=False)
        current_error_rows = int(error_mask.sum())
        skipped_error_rows += current_error_rows
        valid_chunk = chunk.loc[~error_mask].copy()
        if valid_chunk.empty:
            continue
        texts = valid_chunk[TEXT_COLUMN].fillna('').astype(str).tolist()
        scores = []
        docs = nlp.pipe(texts, batch_size=SPACY_BATCH_SIZE)
        for doc in docs:
            scores.append(score_document(doc, lexicon))
        score_df = pd.DataFrame(scores, columns=[*VAD_COLUMNS, '_matched_tokens', '_alphabetic_tokens'], index=valid_chunk.index)
        for column in VAD_COLUMNS:
            valid_chunk[column] = score_df[column]
        valid_rows += len(valid_chunk)
        total_matched += int(score_df['_matched_tokens'].sum())
        total_alphabetic += int(score_df['_alphabetic_tokens'].sum())
        rows_without_matches += int((score_df['_matched_tokens'] == 0).sum())
        encoding = 'utf-8-sig' if first_chunk else 'utf-8'
        valid_chunk.to_csv(temporary_file, sep=';', index=False, mode='w' if first_chunk else 'a', header=first_chunk, encoding=encoding, quoting=csv.QUOTE_MINIMAL, float_format='%.6f')
        first_chunk = False
    if valid_rows + skipped_error_rows != input_rows:
        if temporary_file.exists():
            temporary_file.unlink()
        raise RuntimeError(f'Incoerenza nel conteggio delle righe:\n{input_file}\nInput: {input_rows:,}\nValide: {valid_rows:,}\nErrori: {skipped_error_rows:,}')
    if first_chunk:
        output_columns = list(input_columns)
        for column in VAD_COLUMNS:
            if column not in output_columns:
                output_columns.append(column)
        pd.DataFrame(columns=output_columns).to_csv(temporary_file, sep=';', index=False, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary_file, output_file)
    lexical_coverage = total_matched / total_alphabetic if total_alphabetic > 0 else float('nan')
    qc_data = {'pipeline_version': PIPELINE_VERSION, 'error_prefix': ERROR_PREFIX, 'input_file': str(input_file), 'output_file': str(output_file), 'input_rows': input_rows, 'rows': valid_rows, 'error_rows_skipped': skipped_error_rows, 'matched_tokens': total_matched, 'alphabetic_tokens': total_alphabetic, 'lexical_coverage': lexical_coverage, 'rows_without_vad_matches': rows_without_matches}
    write_json_atomic(qc_file, qc_data)
    return qc_data

def final_file_is_complete(final_file: Path, meta_file: Path, expected_rows: int):
    """
    Controlla il CSV concatenato finale.
    """
    if not final_file.exists() or not meta_file.exists():
        return False
    try:
        with meta_file.open('r', encoding='utf-8') as handle:
            meta = json.load(handle)
        if meta.get('pipeline_version') != PIPELINE_VERSION:
            return False
        if meta.get('rows') != expected_rows:
            return False
        columns = pd.read_csv(final_file, sep=';', nrows=0, encoding='utf-8-sig').columns
        return all((column in columns for column in VAD_COLUMNS))
    except Exception:
        return False

def concatenate_outputs(output_files, final_file: Path, expected_rows: int, source_rows: int, skipped_error_rows: int):
    """
    Concatena i 150 CSV VAD.

    Il numero finale di righe corrisponde a:

        righe sorgente - righe di errore
    """
    meta_file = final_file.with_suffix(final_file.suffix + '.meta.json')
    if final_file_is_complete(final_file, meta_file, expected_rows):
        print()
        print(f'CSV concatenato già completo: {final_file.name}')
        return final_file
    print()
    print('Concatenazione dei CSV elaborati...')
    print(f'File finale: {final_file}')
    print(f'Righe valide attese: {expected_rows:,}')
    temporary_file = final_file.with_suffix(final_file.suffix + '.tmp')
    if temporary_file.exists():
        temporary_file.unlink()
    first_chunk = True
    total_rows = 0
    expected_columns = None
    for csv_file in tqdm(output_files, desc='Concatenazione'):
        current_columns = list(pd.read_csv(csv_file, sep=';', nrows=0, encoding='utf-8-sig').columns)
        if expected_columns is None:
            expected_columns = current_columns
        elif current_columns != expected_columns:
            raise RuntimeError(f'I CSV elaborati non hanno tutti le stesse colonne.\nFile problematico: {csv_file}')
        reader = pd.read_csv(csv_file, sep=';', chunksize=CHUNK_SIZE, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        for chunk in reader:
            total_rows += len(chunk)
            encoding = 'utf-8-sig' if first_chunk else 'utf-8'
            chunk.to_csv(temporary_file, sep=';', index=False, mode='w' if first_chunk else 'a', header=first_chunk, encoding=encoding, quoting=csv.QUOTE_MINIMAL)
            first_chunk = False
    if first_chunk:
        pd.DataFrame(columns=expected_columns or []).to_csv(temporary_file, sep=';', index=False, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL)
    if total_rows != expected_rows:
        if temporary_file.exists():
            temporary_file.unlink()
        raise RuntimeError(f'Il CSV concatenato conterrebbe {total_rows:,} righe invece delle {expected_rows:,} attese.')
    if source_rows - skipped_error_rows != total_rows:
        if temporary_file.exists():
            temporary_file.unlink()
        raise RuntimeError(f'Il conteggio delle righe non coincide.\nRighe sorgente: {source_rows:,}\nErrori esclusi: {skipped_error_rows:,}\nRighe valide: {total_rows:,}')
    os.replace(temporary_file, final_file)
    write_json_atomic(meta_file, {'pipeline_version': PIPELINE_VERSION, 'rows': total_rows, 'source_rows': source_rows, 'error_rows_skipped': skipped_error_rows, 'source_csv_count': len(output_files), 'final_file': str(final_file)})
    print(f'Concatenazione completata: {total_rows:,} righe valide.')
    print(f'Risposte di errore escluse: {skipped_error_rows:,}')
    return final_file

def process_dataset(dataset, nlp, lexicon):
    """
    Elabora un solo modello per esecuzione.
    """
    name = dataset['name']
    input_dir = dataset['input_dir']
    output_dir = dataset['output_dir']
    final_file = output_dir / dataset['final_filename']
    print()
    print('======================================')
    print(f' {name}')
    print('======================================')
    print(f'Input : {input_dir}')
    print(f'Output: {output_dir}')
    print()
    if not input_dir.exists():
        raise FileNotFoundError(f'\nNon trovo la cartella del dataset selezionato:\n{input_dir}\n')
    csv_files = sorted((path for path in input_dir.rglob('*.csv') if '__MACOSX' not in path.parts and '.ipynb_checkpoints' not in path.parts and (not path.name.startswith('._'))))
    print(f'CSV trovati: {len(csv_files)}')
    if len(csv_files) != EXPECTED_CSV_COUNT:
        raise RuntimeError(f'Per {name} erano attesi {EXPECTED_CSV_COUNT} CSV, ma ne sono stati trovati {len(csv_files)}.')
    print(f'Numero di CSV corretto ({EXPECTED_CSV_COUNT}).')
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_results = []
    output_files = []
    skipped_files = 0
    for input_file in tqdm(csv_files, desc=f'VAD - {name}'):
        relative_path = input_file.relative_to(input_dir)
        output_file = output_dir / relative_path
        qc_file = qc_path_for(output_dir, relative_path)
        output_files.append(output_file)
        if output_is_complete(output_file, qc_file):
            skipped_files += 1
            qc_results.append(load_qc(qc_file))
            continue
        qc_data = process_csv(input_file=input_file, output_file=output_file, qc_file=qc_file, nlp=nlp, lexicon=lexicon)
        qc_results.append(qc_data)
    qc_df = pd.DataFrame(qc_results)
    qc_summary_file = output_dir / 'vad_qc_summary.csv'
    qc_df.to_csv(qc_summary_file, sep=';', index=False, encoding='utf-8-sig', float_format='%.6f')
    total_source_rows = int(qc_df['input_rows'].sum())
    total_valid_rows = int(qc_df['rows'].sum())
    total_error_rows = int(qc_df['error_rows_skipped'].sum())
    total_matched = int(qc_df['matched_tokens'].sum())
    total_alphabetic = int(qc_df['alphabetic_tokens'].sum())
    total_no_matches = int(qc_df['rows_without_vad_matches'].sum())
    global_coverage = total_matched / total_alphabetic if total_alphabetic > 0 else float('nan')
    print()
    print(f'CSV già completati e saltati: {skipped_files}')
    print(f'Righe sorgente complessive: {total_source_rows:,}')
    print(f'Risposte di errore escluse: {total_error_rows:,}')
    print(f'Righe valide analizzate: {total_valid_rows:,}')
    print(f'Copertura lessicale globale: {global_coverage:.2%}')
    print(f'Risposte valide senza alcun match VAD: {total_no_matches:,}')
    if total_source_rows != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(f'Per {name} risultano {total_source_rows:,} righe sorgente invece delle {EXPECTED_SOURCE_ROWS:,} attese.')
    if total_valid_rows + total_error_rows != total_source_rows:
        raise RuntimeError(f'Il conteggio globale non coincide.\nRighe sorgente: {total_source_rows:,}\nRighe valide: {total_valid_rows:,}\nErrori esclusi: {total_error_rows:,}')
    for input_file, output_file in zip(csv_files, output_files):
        relative_path = input_file.relative_to(input_dir)
        qc_file = qc_path_for(output_dir, relative_path)
        if not output_is_complete(output_file, qc_file):
            raise RuntimeError(f'Output non completo: {output_file}')
    concatenate_outputs(output_files=output_files, final_file=final_file, expected_rows=total_valid_rows, source_rows=total_source_rows, skipped_error_rows=total_error_rows)
    print()
    print(f'Dataset {name} completato.')
    print(f'CSV definitivo: {final_file}')
    return final_file

def main():
    args = parse_arguments()
    dataset = DATASETS[args.dataset]
    print()
    print('======================================')
    print(' ANALISI NRC-VAD')
    print(' MEDIA - STD - MEDIANA - MASSIMO')
    print('======================================')
    print()
    print(f"Dataset selezionato: {dataset['name']}")
    if not dataset['input_dir'].exists():
        raise FileNotFoundError(f"\nManca la cartella di input selezionata:\n{dataset['input_dir']}\n")
    lexicon_path = find_lexicon_path()
    lexicon = load_vad_lexicon(lexicon_path)
    print()
    print(f'Caricamento modello spaCy: {SPACY_MODEL}')
    try:
        nlp = spacy.load(SPACY_MODEL, disable=['parser', 'ner'])
    except OSError as exc:
        raise RuntimeError('\nIl modello spaCy en_core_web_sm non è installato.\nEsegui:\n\nC:\\python313\\python.exe -m spacy download en_core_web_sm\n') from exc
    print('Modello spaCy caricato correttamente.')
    final_file = process_dataset(dataset, nlp, lexicon)
    print()
    print('======================================')
    print(' ELABORAZIONE COMPLETATA')
    print('======================================')
    print()
    print('CSV definitivo prodotto:')
    print(f'  - {final_file}')
    print()
if __name__ == '__main__':
    main()

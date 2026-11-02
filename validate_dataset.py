"""Optional consistency checker for the final merged dataset."""
from pathlib import Path
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import json
import re
import unicodedata
import zipfile
import pandas as pd
DATASET = Path('merged_outputs/merged_long_qwen_obliterated.zip')
OUTPUT_DIR = Path('validation')
CHUNK_SIZE = 10000
EXPECTED_RESPONSES = 275
COLUMN_NAMES = {'model': ['model', 'model_name', 'model_id'], 'occupation': ['occupation', 'occupation_name', 'job', 'job_title'], 'prompt_id': ['prompt_id', 'user_prompt_id', 'compiled_prompt_id'], 'prompt_condition': ['prompt_condition', 'user_prompt_condition', 'prompt_type', 'condition'], 'system_id': ['system_prompt_id', 'sys_prompt_id', 'system_id', 'system_condition'], 'image': ['image', 'image_path', 'image_id', 'image_filename', 'image_name', 'filename'], 'gender': ['gender_perceived', 'perceived_gender', 'gender', 'gender_label'], 'race': ['race_perceived', 'perceived_race', 'race', 'race_label'], 'visual_condition': ['sentiment', 'image_condition', 'visual_condition', 'image_sentiment'], 'source': ['image_source', 'source_type', 'source', 'image_origin', 'image_provenance']}
PROFESSIONS = {'Responsabilita e prestigio': ['surgeon', 'dentist', 'chief executive', 'sales manager', 'construction manager', 'chef', 'veterinarian', 'producer or director', 'legislator'], 'Cura, servizio e basso status': ['dishwasher', 'personal care aide', 'preschool teacher', 'elementary school teacher', 'kindergarten teacher', 'phlebotomist', 'home health aide', 'healthcare social worker', 'special education teacher', 'baker'], 'Manuali, forza e sicurezza': ['carpenter', 'plumber', 'construction laborer', 'security guard', 'firefighter', 'electrician', 'worker in farming, fishing or forestry', 'blockmason'], 'STEM': ['computer programmer', 'web developer', 'information security analyst', 'mechanical engineer', 'electrical engineer', 'computer and information systems manager', 'aerospace engineer', 'computer hardware engineer', 'civil engineer'], 'Stereotipi occupazionali': ['hairdresser, hairstylist or cosmetologist', 'registered nurse', 'childcare worker', 'barber', 'police officer', 'aircraft pilot', 'executive secretary', 'laundry worker', 'maid', 'dressmaker', 'manicurist or pedicurist', 'massage therapist', 'flight attendant', 'receptionist or information clerk']}

def normalize(text):
    """Uniforma maiuscole, accenti e punteggiatura."""
    text = unicodedata.normalize('NFKD', str(text))
    text = ''.join((char for char in text if not unicodedata.combining(char)))
    text = text.lower().replace('&', ' and ')
    return ' '.join(re.findall('[a-z0-9]+', text))

def profession_groups():
    """Associa ogni professione al proprio macrogruppo."""
    groups = {}
    for group, professions in PROFESSIONS.items():
        for profession in professions:
            key = normalize(profession)
            if key in groups:
                raise ValueError(f'Professione duplicata: {profession}')
            groups[key] = group
    if len(groups) != 50:
        raise ValueError(f'Attese 50 professioni, trovate {len(groups)}.')
    return groups

@contextmanager
def open_dataset(path):
    """Legge il CSV, anche quando si trova dentro uno ZIP."""
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            files = [name for name in archive.namelist() if name.lower().endswith('.csv') and (not name.startswith('__MACOSX/'))]
            if len(files) != 1:
                raise ValueError('Lo ZIP deve contenere un solo CSV.')
            with archive.open(files[0]) as file:
                yield file
    else:
        with open(path, 'rb') as file:
            yield file

def get_columns(path):
    """Legge l'intestazione e riconosce i campi."""
    with open_dataset(path) as file:
        header = pd.read_csv(file, sep=';', nrows=0, encoding='utf-8-sig').columns.tolist()
    available = {name.lower().strip(): name for name in header}
    mapping = {}
    for field, alternatives in COLUMN_NAMES.items():
        for alternative in alternatives:
            if alternative in available:
                mapping[field] = available[alternative]
                break
    return (header, mapping)

def save_csv(filename, records):
    """Salva una tabella nella cartella di controllo."""
    if not records:
        return
    with open(OUTPUT_DIR / filename, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()), delimiter=';')
        writer.writeheader()
        writer.writerows(records)

def image_key(occupation, image):
    """Identifica l'immagine anche tramite la professione."""
    path = image.strip().replace('\\', '/')
    return (normalize(occupation), path)

def validate():
    if not DATASET.exists():
        raise FileNotFoundError(f'Non trovo il dataset: {DATASET}')
    OUTPUT_DIR.mkdir(exist_ok=True)
    headers, mapping = get_columns(DATASET)
    print('\nColonne disponibili:')
    for name in headers:
        print(' -', name)
    print('\nColonne riconosciute:')
    for field, name in mapping.items():
        print(f' {field}: {name}')
    required = ['model', 'image', 'occupation']
    missing = [name for name in required if name not in mapping]
    if missing:
        raise ValueError('Colonne obbligatorie non riconosciute: ' + ', '.join(missing) + '\nControlla COLUMN_NAMES.')
    groups = profession_groups()
    total_rows = 0
    rows_by_model = Counter()
    rows_by_group = Counter()
    unknown_professions = Counter()
    distributions = {field: Counter() for field in COLUMN_NAMES}
    images = defaultdict(lambda: {'rows': 0, 'prompts': set(), 'systems': set(), 'combinations': set(), 'duplicates': 0, 'gender': set(), 'race': set(), 'source': set(), 'visual_condition': set()})
    image_annotations = defaultdict(lambda: {'gender': set(), 'race': set(), 'source': set(), 'visual_condition': set()})
    with open_dataset(DATASET) as file:
        reader = pd.read_csv(file, sep=';', usecols=list(mapping.values()), chunksize=CHUNK_SIZE, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        for chunk in reader:
            chunk = chunk.rename(columns={original: field for field, original in mapping.items()})
            for row in chunk.itertuples(index=False):
                model = getattr(row, 'model').strip()
                occupation = getattr(row, 'occupation').strip()
                image = getattr(row, 'image').strip()
                prompt = getattr(row, 'prompt_id', '').strip()
                system = getattr(row, 'system_id', '').strip()
                if 'prompt_id' not in mapping:
                    prompt = ''
                values = {field: str(getattr(row, field, '')).strip() for field in distributions if field in mapping}
                total_rows += 1
                rows_by_model[model] += 1
                group = groups.get(normalize(occupation))
                if group is None:
                    unknown_professions[occupation] += 1
                else:
                    rows_by_group[model, group] += 1
                for field, value in values.items():
                    distributions[field][value] += 1
                key = image_key(occupation, image)
                model_key = (model, key)
                info = images[model_key]
                info['rows'] += 1
                if prompt:
                    info['prompts'].add(prompt)
                if system:
                    info['systems'].add(system)
                if prompt and system:
                    combination = (prompt, system)
                    if combination in info['combinations']:
                        info['duplicates'] += 1
                    else:
                        info['combinations'].add(combination)
                for field in ['gender', 'race', 'source', 'visual_condition']:
                    value = values.get(field, '')
                    if value:
                        info[field].add(value)
                        image_annotations[key][field].add(value)
            if total_rows % 100000 < len(chunk):
                print(f'Righe esaminate: {total_rows:,}')
    summary = []
    image_report = []
    problems = []
    for (model, key), info in images.items():
        occupation, image = key
        image_report.append({'model': model, 'occupation': occupation, 'image': image, 'rows': info['rows'], 'expected': EXPECTED_RESPONSES, 'missing_rows': max(EXPECTED_RESPONSES - info['rows'], 0), 'duplicate_combinations': info['duplicates'], 'unique_prompts': len(info['prompts']), 'unique_systems': len(info['systems'])})
        if info['rows'] != EXPECTED_RESPONSES:
            problems.append({'model': model, 'occupation': occupation, 'image': image, 'problem': 'Numero di risposte diverso da 275', 'detail': info['rows']})
        if info['duplicates']:
            problems.append({'model': model, 'occupation': occupation, 'image': image, 'problem': 'Combinazioni prompt duplicate', 'detail': info['duplicates']})
    for key, annotations in image_annotations.items():
        for field, values in annotations.items():
            if len(values) > 1:
                problems.append({'model': 'ALL', 'occupation': key[0], 'image': key[1], 'problem': f'Etichette discordanti: {field}', 'detail': ' | '.join(sorted(values))})
    for model, count in rows_by_model.items():
        model_images = sum((1 for current_model, _ in images if current_model == model))
        summary.append({'model': model, 'valid_rows': count, 'distinct_images': model_images, 'expected_rows_for_images': model_images * EXPECTED_RESPONSES})
    group_report = [{'model': model, 'macrogroup': group, 'responses': count} for (model, group), count in sorted(rows_by_group.items())]
    distribution_report = [{'variable': field, 'value': value, 'count': count} for field, counter in distributions.items() for value, count in counter.most_common()]
    unknown_report = [{'occupation': name, 'rows': count} for name, count in unknown_professions.most_common()]
    save_csv('model_summary.csv', summary)
    save_csv('macrogroup_summary.csv', group_report)
    save_csv('image_inventory.csv', image_report)
    save_csv('distributions.csv', distribution_report)
    save_csv('problems.csv', problems)
    save_csv('unknown_professions.csv', unknown_report)
    with open(OUTPUT_DIR / 'columns.json', 'w', encoding='utf-8') as file:
        json.dump({'all_columns': headers, 'recognized_columns': mapping, 'expected_responses_per_image': EXPECTED_RESPONSES, 'total_rows': total_rows}, file, indent=2, ensure_ascii=False)
    print('\nVALIDAZIONE COMPLETATA')
    print(f'Righe totali: {total_rows:,}')
    print(f'Immagini modello: {len(images):,}')
    print(f'Professioni non riconosciute: {len(unknown_professions)}')
    print(f'Problemi segnalati: {len(problems)}')
    if 'prompt_id' not in mapping or 'system_id' not in mapping:
        print('\nATTENZIONE: impossibile controllare le combinazioni duplicate senza entrambi gli ID.')
    print(f'\nReport salvati in: {OUTPUT_DIR.resolve()}')
if __name__ == '__main__':
    validate()

"""RQ4: compare real and AI-generated images while controlling for profession and visual condition."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / 'merged_outputs' / 'merged_long_qwen_obliterated.csv'
OUTPUT_DIR = ROOT / 'analysis_outputs' / 'rq4'
SYSTEM_PROMPT_FILTER = 'NO_SYS'
EXPECTED_PARAPHRASES = 5
PROMPT_ORDER = ['baseline_neutral', 'occupation_neutral', 'positive_modifier', 'negative_modifier']
PROMPT_LABELS = {'baseline_neutral': 'Baseline', 'occupation_neutral': 'Occupational', 'positive_modifier': 'Positive', 'negative_modifier': 'Negative'}
SENTIMENT_ORDER = ['NEUTRAL', 'POSITIVE', 'NEGATIVE']
SOURCE_ORDER = ['REAL', 'AI']
SOURCE_LABELS = {'REAL': 'Real', 'AI': 'AI'}
DIMENSIONS = ['valence', 'arousal', 'dominance']
DIMENSION_LABELS = {'valence': 'Valence', 'arousal': 'Arousal', 'dominance': 'Dominance'}
STATISTICS = ['mean', 'std', 'median', 'max']
VAD_COLUMNS = [f'vad_{dimension}_{statistic}' for dimension in DIMENSIONS for statistic in STATISTICS]
MODEL_COLORS = {'Qwen Standard': '#3978B8', 'Qwen Obliterated': '#D66545'}
SOURCE_STYLES = {'REAL': {'level': 1.0, 'hatch': ''}, 'AI': {'level': 0.58, 'hatch': '//'}}

def model_name(value):
    if 'OBLITERATED' in str(value).upper():
        return 'Qwen Obliterated'
    return 'Qwen Standard'

def normalize_source(value):
    value = str(value).strip().upper()
    aliases = {'REAL': 'REAL', 'WEB': 'REAL', 'HUMAN': 'REAL', 'AI': 'AI', 'GENERATED': 'AI', 'GENERATED_AI': 'AI', 'SYNTHETIC': 'AI', 'ARTIFICIAL': 'AI'}
    if value not in aliases:
        raise ValueError(f'Valore inatteso in image_source: {value!r}')
    return aliases[value]

def normalize_sentiment(value):
    value = str(value).strip().upper()
    aliases = {'NEUTRAL': 'NEUTRAL', 'NEUTRO': 'NEUTRAL', 'POSITIVE': 'POSITIVE', 'POSITIVO': 'POSITIVE', 'NEGATIVE': 'NEGATIVE', 'NEGATIVO': 'NEGATIVE'}
    if value not in aliases:
        raise ValueError(f'Valore inatteso in sentiment: {value!r}')
    return aliases[value]

def shade(color, level):
    rgb = np.array(to_rgb(color))
    return tuple(1 - level * (1 - rgb))

def save_csv(df, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, sep=';', index=False, encoding='utf-8-sig')
    return path

def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f'{stem}.png'
    pdf = OUTPUT_DIR / f'{stem}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0.15)
    fig.savefig(pdf, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    return (png, pdf)

def format_signed(value, digits=4):
    value = round(float(value), digits)
    if value == 0:
        return f'{0:.{digits}f}'.replace('.', ',')
    return f'{value:+.{digits}f}'.replace('.', ',')

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f'Non trovo il dataset LONG:\n{INPUT_FILE}')
    required_columns = ['image', 'profession', 'model_variant', 'system_prompt_id', 'prompt_id', 'prompt_condition', 'image_source', 'sentiment', *VAD_COLUMNS]
    df = pd.read_csv(INPUT_FILE, sep=';', usecols=required_columns, low_memory=False, encoding='utf-8-sig')
    df = df[(df['system_prompt_id'] == SYSTEM_PROMPT_FILTER) & df['prompt_condition'].isin(PROMPT_ORDER)].copy()
    df['model_label'] = df['model_variant'].map(model_name)
    df['image_source'] = df['image_source'].map(normalize_source)
    df['sentiment'] = df['sentiment'].map(normalize_sentiment)
    if set(df['image_source'].unique()) != set(SOURCE_ORDER):
        raise ValueError('Nel subset RQ4 non sono presenti entrambe le provenienze REAL e AI.')
    missing_sentiments = set(SENTIMENT_ORDER) - set(df['sentiment'].unique())
    if missing_sentiments:
        raise ValueError('Mancano condizioni visive: ' + ', '.join(sorted(missing_sentiments)))
    for column in VAD_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        if df[column].isna().any():
            raise ValueError(f'{column} contiene valori mancanti o non numerici.')
        if df[column].min() < 0 or df[column].max() > 1:
            raise ValueError(f'{column} contiene valori fuori dalla scala 0-1.')
    return df

def select_complete_images(df):
    counts = df.groupby(['model_label', 'image', 'prompt_condition'], observed=True).agg(n_rows=('prompt_id', 'size'), n_prompt_ids=('prompt_id', 'nunique')).reset_index()
    counts['condition_complete'] = (counts['n_rows'] == EXPECTED_PARAPHRASES) & (counts['n_prompt_ids'] == EXPECTED_PARAPHRASES)
    image_check = counts.groupby(['model_label', 'image'], observed=True).agg(n_conditions=('prompt_condition', 'nunique'), n_complete=('condition_complete', 'sum')).reset_index()
    complete = image_check[(image_check['n_conditions'] == len(PROMPT_ORDER)) & (image_check['n_complete'] == len(PROMPT_ORDER))][['model_label', 'image']].copy()
    filtered = df.merge(complete, on=['model_label', 'image'], how='inner')
    report_rows = []
    for model in ['Qwen Standard', 'Qwen Obliterated']:
        before = df.loc[df['model_label'] == model, 'image'].nunique()
        after = filtered.loc[filtered['model_label'] == model, 'image'].nunique()
        report_rows.append({'model_label': model, 'images_before': int(before), 'images_complete': int(after), 'images_excluded': int(before - after)})
    report = pd.DataFrame(report_rows)
    save_csv(report, 'rq4_completeness_report.csv')
    return (filtered, report)

def build_image_level(df):
    image_level = df.groupby(['model_label', 'image', 'profession', 'image_source', 'sentiment', 'prompt_condition'], as_index=False, observed=True).agg(**{column: (column, 'mean') for column in VAD_COLUMNS})
    save_csv(image_level, 'rq4_image_level.csv')
    return image_level

def select_matched_professions(image_level):
    base = image_level[image_level['prompt_condition'] == 'occupation_neutral'].copy()
    counts = base.groupby(['model_label', 'profession', 'sentiment', 'image_source'], observed=True)['image'].nunique().reset_index(name='n_images')
    presence = counts.assign(present=lambda x: x['n_images'] > 0).pivot_table(index=['model_label', 'profession', 'sentiment'], columns='image_source', values='present', aggfunc='max', fill_value=False).reset_index()
    for source in SOURCE_ORDER:
        if source not in presence.columns:
            presence[source] = False
    presence['sentiment_matched'] = presence[SOURCE_ORDER].all(axis=1)
    profession_check = presence.groupby(['model_label', 'profession'], observed=True).agg(n_sentiments=('sentiment', 'nunique'), n_matched_sentiments=('sentiment_matched', 'sum')).reset_index()
    profession_check['eligible'] = (profession_check['n_sentiments'] == len(SENTIMENT_ORDER)) & (profession_check['n_matched_sentiments'] == len(SENTIMENT_ORDER))
    eligible = profession_check[profession_check['eligible']][['model_label', 'profession']].copy()
    save_csv(counts, 'rq4_source_counts_by_profession_sentiment.csv')
    save_csv(profession_check, 'rq4_matched_professions_report.csv')
    return (eligible, profession_check)

def build_profession_source_values(image_level, eligible):
    data = image_level.merge(eligible, on=['model_label', 'profession'], how='inner')
    cell_level = data.groupby(['model_label', 'profession', 'sentiment', 'image_source', 'prompt_condition'], as_index=False, observed=True).agg(**{column: (column, 'mean') for column in VAD_COLUMNS}, n_images=('image', 'nunique'))
    expected_cells = len(SENTIMENT_ORDER) * len(SOURCE_ORDER) * len(PROMPT_ORDER)
    cell_check = cell_level.groupby(['model_label', 'profession'], observed=True).size().reset_index(name='n_cells')
    bad = cell_check[cell_check['n_cells'] != expected_cells]
    if not bad.empty:
        raise ValueError(f'Alcune professioni non hanno tutte le {expected_cells} celle attese:\n' + bad.to_string(index=False))
    profession_source = cell_level.groupby(['model_label', 'profession', 'image_source', 'prompt_condition'], as_index=False, observed=True).agg(**{column: (column, 'mean') for column in VAD_COLUMNS}, n_sentiments=('sentiment', 'nunique'))
    save_csv(cell_level, 'rq4_cell_level.csv')
    save_csv(profession_source, 'rq4_profession_source_values.csv')
    return profession_source

def build_absolute_summary(profession_source):
    rows = []
    for dimension in DIMENSIONS:
        for statistic in STATISTICS:
            column = f'vad_{dimension}_{statistic}'
            summary = profession_source.groupby(['model_label', 'image_source', 'prompt_condition'], observed=True).agg(value_mean=(column, 'mean'), value_sd_between_professions=(column, lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0), n_professions=('profession', 'nunique')).reset_index()
            summary['dimension'] = dimension
            summary['statistic'] = statistic
            rows.append(summary)
    result = pd.concat(rows, ignore_index=True)
    save_csv(result, 'rq4_absolute_summary.csv')
    return result

def build_difference_summary(profession_source):
    rows = []
    for dimension in DIMENSIONS:
        column = f'vad_{dimension}_mean'
        wide = profession_source.pivot_table(index=['model_label', 'profession', 'prompt_condition'], columns='image_source', values=column).reset_index()
        if 'AI' not in wide.columns or 'REAL' not in wide.columns:
            raise ValueError('Impossibile calcolare AI - REAL: manca una provenienza.')
        wide['difference'] = wide['AI'] - wide['REAL']
        summary = wide.groupby(['model_label', 'prompt_condition'], observed=True).agg(mean_difference=('difference', 'mean'), median_difference=('difference', 'median'), std_difference=('difference', lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0), min_difference=('difference', 'min'), max_difference=('difference', 'max'), n_professions=('profession', 'nunique')).reset_index()
        summary['dimension'] = dimension
        rows.append(summary)
    result = pd.concat(rows, ignore_index=True)
    save_csv(result, 'rq4_difference_summary.csv')
    return result

def plot_absolute_values(summary):
    plot_df = summary[summary['statistic'] == 'mean'].copy()
    models = ['Qwen Standard', 'Qwen Obliterated']
    x = np.arange(len(PROMPT_ORDER))
    n_bars = len(models) * len(SOURCE_ORDER)
    bar_width = 0.17
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 9.6), sharey=True)
    for ax, dimension in zip(axes, DIMENSIONS):
        dim_df = plot_df[plot_df['dimension'] == dimension]
        for model_index, model in enumerate(models):
            base_color = MODEL_COLORS[model]
            for source_index, source in enumerate(SOURCE_ORDER):
                values = []
                for condition in PROMPT_ORDER:
                    row = dim_df[(dim_df['model_label'] == model) & (dim_df['image_source'] == source) & (dim_df['prompt_condition'] == condition)]
                    if len(row) != 1:
                        raise ValueError(f'Dato non univoco: {model}, {source}, {condition}, {dimension}')
                    values.append(float(row.iloc[0]['value_mean']))
                bar_number = model_index * len(SOURCE_ORDER) + source_index
                offset = (bar_number - (n_bars - 1) / 2) * bar_width
                style = SOURCE_STYLES[source]
                ax.bar(x + offset, values, width=bar_width, color=shade(base_color, style['level']), edgecolor=base_color, hatch=style['hatch'], linewidth=0.8, zorder=3)
        ax.set_ylim(0, 1.02)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xticks(x)
        ax.set_xticklabels([PROMPT_LABELS[condition] for condition in PROMPT_ORDER], fontsize=10)
        ax.set_title(DIMENSION_LABELS[dimension], fontsize=14, fontweight='bold', pad=8)
        ax.set_ylabel('Punteggio medio NRC-VAD', fontsize=10, labelpad=9)
        ax.grid(axis='y', color='#D8D8D8', linewidth=0.7, alpha=0.55, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#555555')
        ax.spines['bottom'].set_color('#555555')
        ax.tick_params(axis='y', labelsize=9)
    axes[-1].set_xlabel('Condizione di prompt', fontsize=10, labelpad=8)
    model_handles = [Patch(facecolor=MODEL_COLORS[model], edgecolor=MODEL_COLORS[model], label=model) for model in models]
    model_legend = fig.legend(handles=model_handles, title='Modello', loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=2, frameon=False, fontsize=9, title_fontsize=9, columnspacing=2.5)
    fig.add_artist(model_legend)
    neutral = '#666666'
    source_handles = [Patch(facecolor=shade(neutral, SOURCE_STYLES[source]['level']), edgecolor=neutral, hatch=SOURCE_STYLES[source]['hatch'], label=SOURCE_LABELS[source]) for source in SOURCE_ORDER]
    fig.legend(handles=source_handles, title='Provenienza', loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False, fontsize=9, title_fontsize=9, columnspacing=2.2)
    fig.suptitle('Provenienza delle immagini e punteggi NRC-VAD', fontsize=16, fontweight='bold', y=0.995)
    fig.text(0.5, 0.966, 'Confronto REAL/AI controllato per professione e condizione visiva; senza prompt di sistema', ha='center', va='center', fontsize=9.5, color='#555555')
    fig.subplots_adjust(left=0.1, right=0.985, top=0.925, bottom=0.16, hspace=0.4)
    return save_figure(fig, 'rq4_source_absolute_values')

def write_difference_table(summary):
    path = OUTPUT_DIR / 'rq4_source_difference_table.tex'
    lines = ['\\begin{table}[H]', '\\vspace{0.2cm}', '\\centering', '\\small', '\\caption{Differenze medie \\texttt{AI-REAL} nei punteggi NRC-VAD per condizione di prompt. Il confronto è calcolato entro professione dopo aver assegnato lo stesso peso alle tre condizioni visive e viene successivamente mediato tra le professioni considerate.}', '\\label{tab:rq4-source}', '\\begin{tabular}{llcc}', '\\hline', '\\textbf{Dimensione} & \\textbf{Condizione} & \\textbf{Standard} & \\textbf{Obliterated} \\\\', '\\hline']
    for dimension in DIMENSIONS:
        first = True
        for condition in PROMPT_ORDER:
            row_df = summary[(summary['dimension'] == dimension) & (summary['prompt_condition'] == condition)]
            values = {row['model_label']: float(row['mean_difference']) for _, row in row_df.iterrows()}
            expected = {'Qwen Standard', 'Qwen Obliterated'}
            if set(values) != expected:
                raise ValueError(f'Mancano valori per {dimension}, {condition}.')
            dimension_cell = DIMENSION_LABELS[dimension] if first else ''
            lines.append(f"{dimension_cell} & {PROMPT_LABELS[condition]} & {format_signed(values['Qwen Standard'])} & {format_signed(values['Qwen Obliterated'])} \\\\")
            first = False
        lines.append('\\hline')
    lines.extend(['\\end{tabular}', '\\vspace{0.3cm}', '\\end{table}'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path

def print_difference_preview(summary):
    print()
    print('Differenze medie AI - REAL:')
    for model in ['Qwen Standard', 'Qwen Obliterated']:
        print()
        print(f'  {model}')
        model_df = summary[summary['model_label'] == model]
        for condition in PROMPT_ORDER:
            subset = model_df[model_df['prompt_condition'] == condition]
            values = {row['dimension']: float(row['mean_difference']) for _, row in subset.iterrows()}
            print(f"    {PROMPT_LABELS[condition]}: V={values.get('valence', np.nan):+.4f} | A={values.get('arousal', np.nan):+.4f} | D={values.get('dominance', np.nan):+.4f}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print()
    print('======================================')
    print(' RQ4 - PROVENIENZA DELLE IMMAGINI')
    print('======================================')
    df = load_data()
    print()
    print(f'Righe NO_SYS nelle quattro condizioni RQ4: {len(df):,}')
    df, completeness = select_complete_images(df)
    print()
    print('Completezza delle immagini:')
    for _, row in completeness.iterrows():
        print(f"  - {row['model_label']}: {row['images_complete']:,} immagini complete su {row['images_before']:,}")
    image_level = build_image_level(df)
    eligible, _ = select_matched_professions(image_level)
    print()
    print('Professioni con REAL e AI in tutte le tre condizioni visive:')
    counts = eligible.groupby('model_label')['profession'].nunique()
    for model in ['Qwen Standard', 'Qwen Obliterated']:
        print(f'  - {model}: {int(counts.get(model, 0))}')
    profession_source = build_profession_source_values(image_level, eligible)
    absolute_summary = build_absolute_summary(profession_source)
    difference_summary = build_difference_summary(profession_source)
    table_path = write_difference_table(difference_summary)
    figure_png, figure_pdf = plot_absolute_values(absolute_summary)
    print_difference_preview(difference_summary)
    print()
    print('Output finali:')
    print(f'  - {table_path.name}')
    print(f'  - {figure_png.name}')
    print(f'  - {figure_pdf.name}')
    print()
    print(f'Cartella output: {OUTPUT_DIR}')
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print()
        print('ERRORE')
        print(exc)
        sys.exit(1)

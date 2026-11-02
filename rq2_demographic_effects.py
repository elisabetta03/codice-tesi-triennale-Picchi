"""RQ2: compare VAD values by perceived gender and perceived racial group, controlling for profession."""
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
OUTPUT_DIR = ROOT / 'analysis_outputs' / 'rq2'
SYSTEM_PROMPT_FILTER = 'NO_SYS'
EXPECTED_PARAPHRASES = 5
PROMPT_ORDER = ['occupation_neutral', 'positive_modifier', 'negative_modifier']
PROMPT_LABELS = {'occupation_neutral': 'Occupational', 'positive_modifier': 'Positive', 'negative_modifier': 'Negative'}
DIMENSIONS = ['valence', 'arousal', 'dominance']
DIMENSION_LABELS = {'valence': 'Valence', 'arousal': 'Arousal', 'dominance': 'Dominance'}
VAD_COLUMNS = {dimension: f'vad_{dimension}_mean' for dimension in DIMENSIONS}
MODEL_COLORS = {'Qwen Standard': '#3978B8', 'Qwen Obliterated': '#D66545'}

def model_name(value):
    if 'OBLITERATED' in str(value).upper():
        return 'Qwen Obliterated'
    return 'Qwen Standard'

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

def shade(color, level):
    rgb = np.array(to_rgb(color))
    return tuple(1 - level * (1 - rgb))

def format_signed(value, digits=4):
    value = round(float(value), digits)
    if value == 0:
        return f'{0:.{digits}f}'.replace('.', ',')
    return f'{value:+.{digits}f}'.replace('.', ',')

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f'Non trovo il dataset LONG:\n{INPUT_FILE}')
    columns = ['image', 'profession', 'model_variant', 'system_prompt_id', 'prompt_id', 'prompt_condition', 'gender_perceived', 'race_perceived', *VAD_COLUMNS.values()]
    df = pd.read_csv(INPUT_FILE, sep=';', usecols=columns, low_memory=False, encoding='utf-8-sig')
    df = df[(df['system_prompt_id'] == SYSTEM_PROMPT_FILTER) & df['prompt_condition'].isin(PROMPT_ORDER)].copy()
    df['model_label'] = df['model_variant'].map(model_name)
    for column in ['gender_perceived', 'race_perceived']:
        df[column] = df[column].astype(str).str.strip().str.upper()
    unexpected_gender = set(df['gender_perceived'].unique()) - {'M', 'F'}
    unexpected_race = set(df['race_perceived'].unique()) - {'W', 'NW'}
    if unexpected_gender:
        raise ValueError(f'Valori inattesi in gender_perceived: {unexpected_gender}')
    if unexpected_race:
        raise ValueError(f'Valori inattesi in race_perceived: {unexpected_race}')
    for column in VAD_COLUMNS.values():
        df[column] = pd.to_numeric(df[column], errors='coerce')
        if df[column].isna().any():
            raise ValueError(f'{column} contiene valori mancanti o non numerici.')
        if df[column].min() < 0 or df[column].max() > 1:
            raise ValueError(f'{column} contiene valori fuori dalla scala 0-1.')
    return df

def select_complete_images(df):
    counts = df.groupby(['model_label', 'image', 'prompt_condition'], observed=True).agg(n_rows=('prompt_id', 'size'), n_ids=('prompt_id', 'nunique')).reset_index()
    counts['condition_complete'] = (counts['n_rows'] == EXPECTED_PARAPHRASES) & (counts['n_ids'] == EXPECTED_PARAPHRASES)
    image_check = counts.groupby(['model_label', 'image'], observed=True).agg(n_conditions=('prompt_condition', 'nunique'), n_complete=('condition_complete', 'sum')).reset_index()
    complete = image_check[(image_check['n_conditions'] == len(PROMPT_ORDER)) & (image_check['n_complete'] == len(PROMPT_ORDER))][['model_label', 'image']]
    filtered = df.merge(complete, on=['model_label', 'image'], how='inner')
    report_rows = []
    for model in ['Qwen Standard', 'Qwen Obliterated']:
        before = df.loc[df['model_label'] == model, 'image'].nunique()
        after = filtered.loc[filtered['model_label'] == model, 'image'].nunique()
        report_rows.append({'model_label': model, 'images_before': int(before), 'images_complete': int(after), 'images_excluded': int(before - after)})
    report = pd.DataFrame(report_rows)
    save_csv(report, 'rq2_completeness_report.csv')
    return (filtered, report)

def build_image_level(df):
    image_level = df.groupby(['model_label', 'image', 'profession', 'gender_perceived', 'race_perceived', 'prompt_condition'], as_index=False, observed=True).agg(**{column: (column, 'mean') for column in VAD_COLUMNS.values()})
    save_csv(image_level, 'rq2_image_level.csv')
    return image_level

def eligible_professions(image_level, demographic, groups):
    base = image_level[image_level['prompt_condition'] == 'occupation_neutral']
    counts = base.groupby(['model_label', 'profession', demographic], observed=True)['image'].nunique().reset_index(name='n_images')
    presence = counts[counts[demographic].isin(groups)].assign(present=lambda x: x['n_images'] > 0).pivot_table(index=['model_label', 'profession'], columns=demographic, values='present', aggfunc='max', fill_value=False).reset_index()
    for group in groups:
        if group not in presence.columns:
            presence[group] = False
    eligible = presence[presence[groups].all(axis=1)][['model_label', 'profession']]
    return eligible

def build_absolute_summary(image_level, eligible, demographic, groups, output_prefix):
    data = image_level.merge(eligible, on=['model_label', 'profession'], how='inner')
    rows = []
    for dimension in DIMENSIONS:
        column = VAD_COLUMNS[dimension]
        profession_values = data.groupby(['model_label', 'profession', demographic, 'prompt_condition'], observed=True)[column].mean().reset_index(name='value')
        summary = profession_values.groupby(['model_label', demographic, 'prompt_condition'], observed=True).agg(mean_value=('value', 'mean'), n_professions=('profession', 'nunique')).reset_index()
        summary['dimension'] = dimension
        rows.append(summary)
    result = pd.concat(rows, ignore_index=True)
    save_csv(result, f'rq2_{output_prefix}_absolute_summary.csv')
    return result

def build_difference_summary(image_level, eligible, demographic, groups, output_prefix):
    data = image_level.merge(eligible, on=['model_label', 'profession'], how='inner')
    rows = []
    for dimension in DIMENSIONS:
        column = VAD_COLUMNS[dimension]
        profession_group = data.groupby(['model_label', 'profession', demographic, 'prompt_condition'], observed=True)[column].mean().reset_index(name='value')
        condition_wide = profession_group.pivot_table(index=['model_label', 'profession', demographic], columns='prompt_condition', values='value').reset_index()
        condition_wide['occupational'] = condition_wide['occupation_neutral']
        condition_wide['delta_positive'] = condition_wide['positive_modifier'] - condition_wide['occupation_neutral']
        condition_wide['delta_negative'] = condition_wide['negative_modifier'] - condition_wide['occupation_neutral']
        long = condition_wide.melt(id_vars=['model_label', 'profession', demographic], value_vars=['occupational', 'delta_positive', 'delta_negative'], var_name='metric', value_name='value')
        group_wide = long.pivot_table(index=['model_label', 'profession', 'metric'], columns=demographic, values='value').reset_index()
        group_wide['difference'] = group_wide[groups[0]] - group_wide[groups[1]]
        summary = group_wide.groupby(['model_label', 'metric'], observed=True).agg(mean_difference=('difference', 'mean'), n_professions=('profession', 'nunique')).reset_index()
        summary['dimension'] = dimension
        rows.append(summary)
    result = pd.concat(rows, ignore_index=True)
    save_csv(result, f'rq2_{output_prefix}_difference_summary.csv')
    return result

def plot_absolute_values(summary, demographic, groups, title, subtitle, stem):
    models = ['Qwen Standard', 'Qwen Obliterated']
    group_styles = {groups[0]: {'level': 1.0, 'hatch': ''}, groups[1]: {'level': 0.58, 'hatch': '//'}}
    x = np.arange(len(PROMPT_ORDER))
    n_bars = len(models) * len(groups)
    bar_width = 0.17
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 9.6), sharey=True)
    for ax, dimension in zip(axes, DIMENSIONS):
        dim_df = summary[summary['dimension'] == dimension]
        for model_index, model in enumerate(models):
            base_color = MODEL_COLORS[model]
            for group_index, group in enumerate(groups):
                values = []
                for condition in PROMPT_ORDER:
                    row = dim_df[(dim_df['model_label'] == model) & (dim_df[demographic] == group) & (dim_df['prompt_condition'] == condition)]
                    if len(row) != 1:
                        raise ValueError(f'Dato non univoco nel grafico: {model}, {group}, {condition}, {dimension}')
                    values.append(float(row.iloc[0]['mean_value']))
                bar_number = model_index * len(groups) + group_index
                offset = (bar_number - (n_bars - 1) / 2) * bar_width
                style = group_styles[group]
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
    group_handles = [Patch(facecolor=shade(neutral, group_styles[group]['level']), edgecolor=neutral, hatch=group_styles[group]['hatch'], label=group) for group in groups]
    fig.legend(handles=group_handles, title='Gruppo', loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False, fontsize=9, title_fontsize=9, columnspacing=2.2)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    fig.text(0.5, 0.966, subtitle, ha='center', va='center', fontsize=9.5, color='#555555')
    fig.subplots_adjust(left=0.1, right=0.985, top=0.925, bottom=0.16, hspace=0.4)
    return save_figure(fig, stem)

def write_difference_table(summary, demographic):
    if demographic == 'gender_perceived':
        filename = 'rq2_gender_table.tex'
        caption = 'Differenze medie \\texttt{M-F} per le tre dimensioni NRC-VAD, calcolate entro professione e successivamente mediate tra le professioni considerate.'
        label = 'tab:rq2-genere'
    else:
        filename = 'rq2_race_table.tex'
        caption = 'Differenze medie \\texttt{W-NW} per le tre dimensioni NRC-VAD, calcolate entro professione e successivamente mediate tra le professioni considerate.'
        label = 'tab:rq2-race'
    path = OUTPUT_DIR / filename
    metric_order = ['occupational', 'delta_positive', 'delta_negative']
    metric_labels = {'occupational': 'Occupational', 'delta_positive': '$\\Delta$ Positive', 'delta_negative': '$\\Delta$ Negative'}
    lines = ['\\begin{table}[H]', '\\vspace{0.2cm}', '\\centering', '\\small', f'\\caption{{{caption}}}', f'\\label{{{label}}}', '\\begin{tabular}{llcc}', '\\hline', '\\textbf{Dimensione} & \\textbf{Misura} & \\textbf{Standard} & \\textbf{Obliterated} \\\\', '\\hline']
    for dimension in DIMENSIONS:
        first = True
        for metric in metric_order:
            row_df = summary[(summary['dimension'] == dimension) & (summary['metric'] == metric)]
            values = {row['model_label']: float(row['mean_difference']) for _, row in row_df.iterrows()}
            dimension_cell = DIMENSION_LABELS[dimension] if first else ''
            lines.append(f"{dimension_cell} & {metric_labels[metric]} & {format_signed(values['Qwen Standard'])} & {format_signed(values['Qwen Obliterated'])} \\\\")
            first = False
        lines.append('\\hline')
    lines.extend(['\\end{tabular}', '\\vspace{0.3cm}', '\\end{table}'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print()
    print('======================================')
    print(' RQ2 - ATTRIBUTI DEMOGRAFICI E VAD')
    print('======================================')
    df = load_data()
    print()
    print(f'Righe NO_SYS nelle tre condizioni RQ2: {len(df):,}')
    df, completeness = select_complete_images(df)
    print()
    print('Completezza:')
    for _, row in completeness.iterrows():
        print(f"  - {row['model_label']}: {row['images_complete']:,} immagini complete su {row['images_before']:,}")
    image_level = build_image_level(df)
    gender_eligible = eligible_professions(image_level, 'gender_perceived', ['M', 'F'])
    gender_absolute = build_absolute_summary(image_level, gender_eligible, 'gender_perceived', ['M', 'F'], 'gender')
    gender_difference = build_difference_summary(image_level, gender_eligible, 'gender_perceived', ['M', 'F'], 'gender')
    race_eligible = eligible_professions(image_level, 'race_perceived', ['W', 'NW'])
    race_absolute = build_absolute_summary(image_level, race_eligible, 'race_perceived', ['W', 'NW'], 'race')
    race_difference = build_difference_summary(image_level, race_eligible, 'race_perceived', ['W', 'NW'], 'race')
    gender_table = write_difference_table(gender_difference, 'gender_perceived')
    race_table = write_difference_table(race_difference, 'race_perceived')
    figures = []
    figures += plot_absolute_values(gender_absolute, 'gender_perceived', ['M', 'F'], 'Condizioni di prompt e genere percepito', 'Valori medi tra professioni; senza prompt di sistema', 'rq2_gender_absolute_values')
    figures += plot_absolute_values(race_absolute, 'race_perceived', ['W', 'NW'], 'Condizioni di prompt e appartenenza razziale percepita', 'Valori medi tra professioni; senza prompt di sistema', 'rq2_race_absolute_values')
    print()
    print('Professioni utilizzabili:')
    for label, eligible in [('M/F', gender_eligible), ('W/NW', race_eligible)]:
        counts = eligible.groupby('model_label')['profession'].nunique()
        for model in ['Qwen Standard', 'Qwen Obliterated']:
            print(f'  - {label} | {model}: {int(counts.get(model, 0))}')
    print()
    print('Tabelle:')
    print(f'  - {gender_table.name}')
    print(f'  - {race_table.name}')
    print()
    print('Grafici:')
    for path in figures:
        print(f'  - {path.name}')
    print()
    print(f'Output: {OUTPUT_DIR}')
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print()
        print('ERRORE')
        print(exc)
        sys.exit(1)

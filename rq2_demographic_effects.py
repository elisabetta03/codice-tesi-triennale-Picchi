"""RQ2: demographic contrasts (M-F and W-NW) in NRC-VAD under NO_SYS.

The analysis follows the procedure described in the thesis:
1. keep occupation_neutral / positive_modifier / negative_modifier under NO_SYS;
2. average the five paraphrases at image level;
3. keep only professions containing both demographic groups;
4. compute, within each profession, the occupation-neutral contrast and the
   differential effect of the positive/negative modifiers;
5. average profession-level contrasts giving each profession equal weight;
6. estimate percentile bootstrap 95% CIs with 10,000 resamples of professions.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / 'merged_outputs' / 'merged_long_qwen_obliterated.csv'
OUTPUT_DIR = ROOT / 'analysis_outputs' / 'rq2'
SYSTEM_PROMPT_FILTER = 'NO_SYS'
EXPECTED_PARAPHRASES = 5
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
MODEL_ORDER = ['Qwen Standard', 'Qwen Obliterated']
MODEL_COLORS = {'Qwen Standard': '#3978B8', 'Qwen Obliterated': '#D66545'}
MODEL_MARKERS = {'Qwen Standard': 'o', 'Qwen Obliterated': 's'}
PROMPT_ORDER = ['occupation_neutral', 'positive_modifier', 'negative_modifier']
DIMENSIONS = ['valence', 'arousal', 'dominance']
DIMENSION_LABELS = {'valence': 'Valence', 'arousal': 'Arousal', 'dominance': 'Dominance'}
VAD_COLUMNS = {d: f'vad_{d}_mean' for d in DIMENSIONS}
METRIC_ORDER = ['occupational', 'delta_positive', 'delta_negative']
METRIC_LABELS = {
    'occupational': 'Occupazione neutra',
    'delta_positive': 'Δ positivo',
    'delta_negative': 'Δ negativo',
}
DEMOGRAPHICS = {
    'gender_perceived': {'groups': ['M', 'F'], 'prefix': 'gender', 'contrast': 'M-F'},
    'race_perceived': {'groups': ['W', 'NW'], 'prefix': 'race', 'contrast': 'W-NW'},
}


def model_name(value):
    return 'Qwen Obliterated' if 'OBLITERATED' in str(value).upper() else 'Qwen Standard'


def save_csv(df, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, sep=';', index=False, encoding='utf-8-sig')
    return path


def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f'{stem}.png'
    pdf = OUTPUT_DIR / f'{stem}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0.12)
    fig.savefig(pdf, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)
    return png, pdf


def format_signed(value, digits=4):
    rounded = round(float(value), digits)
    if rounded == 0:
        return f'{0:.{digits}f}'.replace('.', ',')
    return f'{rounded:+.{digits}f}'.replace('.', ',')


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f'Non trovo il dataset LONG:\n{INPUT_FILE}')
    columns = [
        'image', 'profession', 'model_variant', 'system_prompt_id', 'prompt_id',
        'prompt_condition', 'gender_perceived', 'race_perceived', *VAD_COLUMNS.values()
    ]
    df = pd.read_csv(INPUT_FILE, sep=';', usecols=columns, low_memory=False, encoding='utf-8-sig')
    df = df[(df['system_prompt_id'] == SYSTEM_PROMPT_FILTER) & df['prompt_condition'].isin(PROMPT_ORDER)].copy()
    df['model_label'] = df['model_variant'].map(model_name)
    for column, valid in [('gender_perceived', {'M', 'F'}), ('race_perceived', {'W', 'NW'})]:
        df[column] = df[column].astype(str).str.strip().str.upper()
        unexpected = set(df[column].unique()) - valid
        if unexpected:
            raise ValueError(f'Valori inattesi in {column}: {unexpected}')
    for column in VAD_COLUMNS.values():
        df[column] = pd.to_numeric(df[column], errors='coerce')
        if df[column].isna().any():
            raise ValueError(f'{column} contiene valori mancanti o non numerici.')
        if df[column].min() < 0 or df[column].max() > 1:
            raise ValueError(f'{column} contiene valori fuori dalla scala 0-1.')
    return df


def select_complete_images(df):
    counts = df.groupby(['model_label', 'image', 'prompt_condition'], observed=True).agg(
        n_rows=('prompt_id', 'size'), n_ids=('prompt_id', 'nunique')
    ).reset_index()
    counts['complete_condition'] = (
        (counts['n_rows'] == EXPECTED_PARAPHRASES) &
        (counts['n_ids'] == EXPECTED_PARAPHRASES)
    )
    images = counts.groupby(['model_label', 'image'], observed=True).agg(
        n_conditions=('prompt_condition', 'nunique'),
        n_complete=('complete_condition', 'sum')
    ).reset_index()
    complete = images[
        (images['n_conditions'] == len(PROMPT_ORDER)) &
        (images['n_complete'] == len(PROMPT_ORDER))
    ][['model_label', 'image']]
    filtered = df.merge(complete, on=['model_label', 'image'], how='inner')
    report = []
    for model in MODEL_ORDER:
        before = df.loc[df['model_label'] == model, 'image'].nunique()
        after = filtered.loc[filtered['model_label'] == model, 'image'].nunique()
        report.append({'model_label': model, 'images_before': int(before), 'images_complete': int(after), 'images_excluded': int(before-after)})
    save_csv(pd.DataFrame(report), 'rq2_completeness_report.csv')
    return filtered


def build_image_level(df):
    image_level = df.groupby(
        ['model_label', 'image', 'profession', 'gender_perceived', 'race_perceived', 'prompt_condition'],
        as_index=False, observed=True
    ).agg(**{col: (col, 'mean') for col in VAD_COLUMNS.values()})
    save_csv(image_level, 'rq2_image_level.csv')
    return image_level


def eligible_professions(image_level, demographic, groups):
    base = image_level[image_level['prompt_condition'] == 'occupation_neutral']
    counts = base.groupby(['model_label', 'profession', demographic], observed=True)['image'].nunique().reset_index(name='n_images')
    presence = counts.assign(present=lambda x: x['n_images'] > 0).pivot_table(
        index=['model_label', 'profession'], columns=demographic, values='present', aggfunc='max', fill_value=False
    ).reset_index()
    for group in groups:
        if group not in presence.columns:
            presence[group] = False
    eligible = presence[presence[groups].all(axis=1)][['model_label', 'profession']]
    return eligible


def profession_contrasts(image_level, demographic, groups):
    eligible = eligible_professions(image_level, demographic, groups)
    data = image_level.merge(eligible, on=['model_label', 'profession'], how='inner')
    rows = []
    for dimension in DIMENSIONS:
        column = VAD_COLUMNS[dimension]
        pg = data.groupby(['model_label', 'profession', demographic, 'prompt_condition'], observed=True)[column].mean().reset_index(name='value')
        wide_condition = pg.pivot_table(
            index=['model_label', 'profession', demographic], columns='prompt_condition', values='value'
        ).reset_index()
        wide_condition['occupational'] = wide_condition['occupation_neutral']
        wide_condition['delta_positive'] = wide_condition['positive_modifier'] - wide_condition['occupation_neutral']
        wide_condition['delta_negative'] = wide_condition['negative_modifier'] - wide_condition['occupation_neutral']
        long = wide_condition.melt(
            id_vars=['model_label', 'profession', demographic], value_vars=METRIC_ORDER,
            var_name='metric', value_name='value'
        )
        group_wide = long.pivot_table(
            index=['model_label', 'profession', 'metric'], columns=demographic, values='value'
        ).reset_index()
        group_wide['contrast'] = group_wide[groups[0]] - group_wide[groups[1]]
        group_wide['dimension'] = dimension
        rows.append(group_wide[['model_label', 'profession', 'metric', 'dimension', 'contrast']])
    result = pd.concat(rows, ignore_index=True)
    return result, eligible


def bootstrap_summary(profession_level, prefix):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (model, dimension, metric), group in profession_level.groupby(['model_label', 'dimension', 'metric'], observed=True):
        values = group['contrast'].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        boot = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
        for i in range(BOOTSTRAP_RESAMPLES):
            sample = rng.choice(values, size=len(values), replace=True)
            boot[i] = sample.mean()
        low, high = np.percentile(boot, [2.5, 97.5])
        rows.append({
            'model_label': model,
            'dimension': dimension,
            'metric': metric,
            'mean_difference': float(values.mean()),
            'ci_low': float(low),
            'ci_high': float(high),
            'n_professions': int(len(values)),
            'ci_excludes_zero': bool(low > 0 or high < 0),
        })
    summary = pd.DataFrame(rows)
    save_csv(profession_level, f'rq2_{prefix}_profession_contrasts.csv')
    save_csv(summary, f'rq2_{prefix}_bootstrap_summary.csv')
    return summary


def plot_forest(summary, prefix, contrast_label):
    labels = [(d, m) for d in DIMENSIONS for m in METRIC_ORDER]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    offsets = {'Qwen Standard': 0.12, 'Qwen Obliterated': -0.12}
    for model in MODEL_ORDER:
        xs, lows, highs = [], [], []
        for dimension, metric in labels:
            row = summary[(summary['model_label'] == model) & (summary['dimension'] == dimension) & (summary['metric'] == metric)]
            if len(row) != 1:
                raise ValueError(f'Dato non univoco: {model}, {dimension}, {metric}')
            mean = float(row.iloc[0]['mean_difference'])
            low = float(row.iloc[0]['ci_low'])
            high = float(row.iloc[0]['ci_high'])
            xs.append(mean); lows.append(mean-low); highs.append(high-mean)
        ax.errorbar(
            xs, y + offsets[model], xerr=[lows, highs], fmt=MODEL_MARKERS[model],
            color=MODEL_COLORS[model], ecolor=MODEL_COLORS[model], capsize=2.8,
            markersize=6.5, linewidth=1.4, label=model
        )
    ax.axvline(0, color='#555555', linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{DIMENSION_LABELS[d]}   {METRIC_LABELS[m]}' for d, m in labels], fontsize=10)
    ax.set_xlabel(f'Differenza {contrast_label} (IC 95% bootstrap sulle professioni)', fontsize=11)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.3f}'.replace('.', ',')))
    ax.set_title(f'Differenze {contrast_label} nei punteggi NRC-VAD (NO_SYS)', fontsize=16, fontweight='bold', pad=12)
    ax.grid(axis='x', color='#D8D8D8', linewidth=0.7, alpha=0.55)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.19), ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.28, right=0.98, top=0.9, bottom=0.2)
    return save_figure(fig, f'rq2_{prefix}_forest_plot')


def write_table(summary, prefix, contrast_label):
    path = OUTPUT_DIR / f'rq2_{prefix}_table.tex'
    lines = [
        '\\begin{table}[H]', '\\vspace{0.2cm}', '\\centering', '\\small',
        f'\\caption{{Differenze medie \\texttt{{{contrast_label}}} per le tre dimensioni NRC-VAD, calcolate entro professione e successivamente mediate tra le professioni considerate.}}',
        f'\\label{{tab:rq2-{prefix}}}', '\\begin{tabular}{llcc}', '\\hline',
        '\\textbf{Dimensione} & \\textbf{Misura} & \\textbf{Standard} & \\textbf{Obliterated} \\\\', '\\hline'
    ]
    for dimension in DIMENSIONS:
        first = True
        for metric in METRIC_ORDER:
            values = {}
            for model in MODEL_ORDER:
                row = summary[(summary['model_label'] == model) & (summary['dimension'] == dimension) & (summary['metric'] == metric)]
                values[model] = float(row.iloc[0]['mean_difference'])
            dimension_cell = DIMENSION_LABELS[dimension] if first else ''
            lines.append(f"{dimension_cell} & {METRIC_LABELS[metric]} & {format_signed(values['Qwen Standard'])} & {format_signed(values['Qwen Obliterated'])} \\\\")
            first = False
        lines.append('\\hline')
    lines += ['\\end{tabular}', '\\vspace{0.3cm}', '\\end{table}']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('\n======================================')
    print(' RQ2 - ATTRIBUTI DEMOGRAFICI E VAD')
    print('======================================')
    df = select_complete_images(load_data())
    image_level = build_image_level(df)
    for demographic, config in DEMOGRAPHICS.items():
        profession_level, eligible = profession_contrasts(image_level, demographic, config['groups'])
        summary = bootstrap_summary(profession_level, config['prefix'])
        write_table(summary, config['prefix'], config['contrast'])
        plot_forest(summary, config['prefix'], config['contrast'])
        counts = eligible.groupby('model_label')['profession'].nunique()
        print(f"\n{config['contrast']} - professioni eleggibili:")
        for model in MODEL_ORDER:
            print(f"  - {model}: {int(counts.get(model, 0))}")
    print(f'\nOutput: {OUTPUT_DIR}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('\nERRORE')
        print(exc)
        sys.exit(1)

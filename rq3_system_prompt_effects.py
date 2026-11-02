"""RQ3: quantify system-prompt effects and their interaction with perceived demographic groups."""
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
OUTPUT_DIR = ROOT / 'analysis_outputs' / 'rq3'
USER_PROMPT_CONDITION = 'occupation_neutral'
NO_SYS = 'NO_SYS'
EXPECTED_PARAPHRASES = 5
GEN_PROMPTS = [f'SYS_GEN_{i:03d}' for i in range(1, 8)]
RACE_PROMPTS = [f'SYS_RACE_{i:03d}' for i in range(1, 4)]
ALL_SYSTEMS = [NO_SYS, *GEN_PROMPTS, *RACE_PROMPTS]
GEN_PLOT = [NO_SYS, *GEN_PROMPTS]
RACE_PLOT = [NO_SYS, *RACE_PROMPTS]
PROMPT_LABELS = {'NO_SYS': 'Nessun system prompt', 'SYS_GEN_001': 'Misandrico e misogino', 'SYS_GEN_002': 'Misandrico', 'SYS_GEN_003': 'Misogino', 'SYS_GEN_004': 'Antimisandrico e antimisogino', 'SYS_GEN_005': 'Antimisandrico', 'SYS_GEN_006': 'Antimisogino', 'SYS_GEN_007': 'Neutro sul genere', 'SYS_RACE_001': 'Razzista verso non bianchi', 'SYS_RACE_002': 'Antirazzista verso non bianchi', 'SYS_RACE_003': "Neutro sull'appartenenza razziale"}
SHORT_LABELS = {'NO_SYS': 'NO_SYS', **{f'SYS_GEN_{i:03d}': f'G{i}' for i in range(1, 8)}, **{f'SYS_RACE_{i:03d}': f'R{i}' for i in range(1, 4)}}
DIMENSIONS = ['valence', 'arousal', 'dominance']
DIM_LABELS = {'valence': 'Valence', 'arousal': 'Arousal', 'dominance': 'Dominance'}
STATS = ['mean', 'std', 'median', 'max']
PLOT_STATS = ['mean', 'median', 'max']
STAT_LABELS = {'mean': 'Mean', 'median': 'Median', 'max': 'Max'}
VAD_COLUMNS = [f'vad_{d}_{s}' for d in DIMENSIONS for s in STATS]
MODEL_COLORS = {'Qwen Standard': '#3978B8', 'Qwen Obliterated': '#D66545'}
STAT_LEVELS = {'mean': 1.0, 'median': 0.72, 'max': 0.48}
STAT_HATCHES = {'mean': '', 'median': '//', 'max': '..'}

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
    fig.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0.15)
    fig.savefig(pdf, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    return (png, pdf)

def shade(color, level):
    rgb = np.array(to_rgb(color))
    return tuple(1 - level * (1 - rgb))

def fmt(value, digits=4):
    value = round(float(value), digits)
    if value == 0:
        return f'{0:.{digits}f}'.replace('.', ',')
    return f'{value:+.{digits}f}'.replace('.', ',')

def tex_id(value):
    return str(value).replace('_', '\\_')

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f'Non trovo il dataset LONG:\n{INPUT_FILE}')
    columns = ['image', 'profession', 'model_variant', 'system_prompt_id', 'prompt_id', 'prompt_condition', 'gender_perceived', 'race_perceived', *VAD_COLUMNS]
    df = pd.read_csv(INPUT_FILE, sep=';', usecols=columns, low_memory=False, encoding='utf-8-sig')
    df = df[(df['prompt_condition'] == USER_PROMPT_CONDITION) & df['system_prompt_id'].isin(ALL_SYSTEMS)].copy()
    df['model_label'] = df['model_variant'].map(model_name)
    missing = [s for s in ALL_SYSTEMS if s not in set(df['system_prompt_id'].dropna())]
    if missing:
        raise ValueError('Mancano system prompt: ' + ', '.join(missing))
    for column in ['gender_perceived', 'race_perceived']:
        df[column] = df[column].astype(str).str.strip().str.upper()
    if set(df['gender_perceived'].unique()) - {'M', 'F'}:
        raise ValueError('Valori inattesi in gender_perceived.')
    if set(df['race_perceived'].unique()) - {'W', 'NW'}:
        raise ValueError('Valori inattesi in race_perceived.')
    for column in VAD_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        if df[column].isna().any():
            raise ValueError(f'{column} contiene valori mancanti o non numerici.')
        if df[column].min() < 0 or df[column].max() > 1:
            raise ValueError(f'{column} contiene valori fuori dalla scala 0-1.')
    return df

def select_complete_images(df):
    counts = df.groupby(['model_label', 'image', 'system_prompt_id'], observed=True).agg(n_rows=('prompt_id', 'size'), n_ids=('prompt_id', 'nunique')).reset_index()
    counts['ok'] = (counts['n_rows'] == EXPECTED_PARAPHRASES) & (counts['n_ids'] == EXPECTED_PARAPHRASES)
    images = counts.groupby(['model_label', 'image'], observed=True).agg(n_systems=('system_prompt_id', 'nunique'), n_ok=('ok', 'sum')).reset_index()
    complete = images[(images['n_systems'] == len(ALL_SYSTEMS)) & (images['n_ok'] == len(ALL_SYSTEMS))][['model_label', 'image']]
    filtered = df.merge(complete, on=['model_label', 'image'], how='inner')
    report = []
    for model in ['Qwen Standard', 'Qwen Obliterated']:
        before = df.loc[df['model_label'] == model, 'image'].nunique()
        after = filtered.loc[filtered['model_label'] == model, 'image'].nunique()
        report.append({'model_label': model, 'images_before': int(before), 'images_complete': int(after), 'images_excluded': int(before - after)})
    report = pd.DataFrame(report)
    save_csv(report, 'rq3_completeness_report.csv')
    return (filtered, report)

def build_image_level(df):
    return df.groupby(['model_label', 'image', 'profession', 'gender_perceived', 'race_perceived', 'system_prompt_id'], as_index=False, observed=True).agg(**{column: (column, 'mean') for column in VAD_COLUMNS})

def eligible_professions(image_level, demographic, groups):
    base = image_level[image_level['system_prompt_id'] == NO_SYS]
    counts = base.groupby(['model_label', 'profession', demographic], observed=True)['image'].nunique().reset_index(name='n_images')
    presence = counts[counts[demographic].isin(groups)].assign(present=lambda x: x['n_images'] > 0).pivot_table(index=['model_label', 'profession'], columns=demographic, values='present', aggfunc='max', fill_value=False).reset_index()
    for group in groups:
        if group not in presence.columns:
            presence[group] = False
    return presence[presence[groups].all(axis=1)][['model_label', 'profession']]

def absolute_overall_summary(image_level):
    rows = []
    for dimension in DIMENSIONS:
        for statistic in PLOT_STATS:
            column = f'vad_{dimension}_{statistic}'
            profession = image_level.groupby(['model_label', 'profession', 'system_prompt_id'], observed=True)[column].mean().reset_index(name='value')
            summary = profession.groupby(['model_label', 'system_prompt_id'], observed=True)['value'].mean().reset_index(name='value_mean')
            summary['dimension'] = dimension
            summary['statistic'] = statistic
            rows.append(summary)
    result = pd.concat(rows, ignore_index=True)
    save_csv(result, 'rq3_absolute_overall_summary.csv')
    return result

def absolute_demographic_summary(image_level, eligible, demographic, system_ids, output_prefix):
    data = image_level[image_level['system_prompt_id'].isin(system_ids)].merge(eligible, on=['model_label', 'profession'], how='inner')
    rows = []
    for dimension in DIMENSIONS:
        column = f'vad_{dimension}_mean'
        profession = data.groupby(['model_label', 'profession', demographic, 'system_prompt_id'], observed=True)[column].mean().reset_index(name='value')
        summary = profession.groupby(['model_label', demographic, 'system_prompt_id'], observed=True)['value'].mean().reset_index(name='mean_value')
        summary['dimension'] = dimension
        rows.append(summary)
    result = pd.concat(rows, ignore_index=True)
    save_csv(result, f'rq3_{output_prefix}_absolute_group_summary.csv')
    return result

def build_deltas(image_level):
    mean_cols = [f'vad_{d}_mean' for d in DIMENSIONS]
    baseline = image_level[image_level['system_prompt_id'] == NO_SYS][['model_label', 'image', *mean_cols]].copy()
    baseline = baseline.rename(columns={c: f'{c}_no_sys' for c in mean_cols})
    data = image_level[image_level['system_prompt_id'] != NO_SYS].merge(baseline, on=['model_label', 'image'], how='inner', validate='many_to_one')
    for dimension in DIMENSIONS:
        column = f'vad_{dimension}_mean'
        data[f'{dimension}_delta'] = data[column] - data[f'{column}_no_sys']
    return data

def overall_delta_summary(delta_long):
    rows = []
    for dimension in DIMENSIONS:
        profession = delta_long.groupby(['model_label', 'profession', 'system_prompt_id'], observed=True)[f'{dimension}_delta'].mean().reset_index(name='value')
        summary = profession.groupby(['model_label', 'system_prompt_id'], observed=True)['value'].mean().reset_index(name='mean_delta')
        summary['dimension'] = dimension
        rows.append(summary)
    result = pd.concat(rows, ignore_index=True)
    save_csv(result, 'rq3_overall_summary.csv')
    return result

def demographic_delta_summary(delta_long, eligible, demographic, groups, system_ids, output_prefix):
    data = delta_long[delta_long['system_prompt_id'].isin(system_ids)].merge(eligible, on=['model_label', 'profession'], how='inner')
    rows = []
    for dimension in DIMENSIONS:
        grouped = data.groupby(['model_label', 'profession', demographic, 'system_prompt_id'], observed=True)[f'{dimension}_delta'].mean().reset_index(name='value')
        wide = grouped.pivot_table(index=['model_label', 'profession', 'system_prompt_id'], columns=demographic, values='value').reset_index()
        wide['difference'] = wide[groups[0]] - wide[groups[1]]
        summary = wide.groupby(['model_label', 'system_prompt_id'], observed=True)['difference'].mean().reset_index(name='mean_difference')
        summary['dimension'] = dimension
        rows.append(summary)
    result = pd.concat(rows, ignore_index=True)
    save_csv(result, f'rq3_{output_prefix}_difference_summary.csv')
    return result

def model_legend(fig, y=0.005):
    handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor=MODEL_COLORS[m], label=m) for m in ['Qwen Standard', 'Qwen Obliterated']]
    legend = fig.legend(handles=handles, title='Modello', loc='lower center', bbox_to_anchor=(0.5, y), ncol=2, frameon=False, fontsize=9, title_fontsize=9, columnspacing=2.5)
    fig.add_artist(legend)

def style_axis(ax, dimension, labels):
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_title(DIM_LABELS[dimension], fontsize=14, fontweight='bold', pad=8)
    ax.grid(axis='y', color='#D8D8D8', linewidth=0.7, alpha=0.55, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_overall(summary, system_ids, title, subtitle, stem):
    models = ['Qwen Standard', 'Qwen Obliterated']
    x = np.arange(len(system_ids))
    width = 0.115 if len(system_ids) >= 8 else 0.12
    size = (13.6, 10.6) if len(system_ids) >= 8 else (10.0, 9.6)
    n_bars = len(models) * len(PLOT_STATS)
    fig, axes = plt.subplots(3, 1, figsize=size, sharey=True)
    for ax, dimension in zip(axes, DIMENSIONS):
        dim = summary[summary['dimension'] == dimension]
        for mi, model in enumerate(models):
            for si, statistic in enumerate(PLOT_STATS):
                values = []
                for system_id in system_ids:
                    row = dim[(dim['model_label'] == model) & (dim['system_prompt_id'] == system_id) & (dim['statistic'] == statistic)]
                    if len(row) != 1:
                        raise ValueError(f'Dato non univoco: {model}, {system_id}, {dimension}, {statistic}')
                    values.append(float(row.iloc[0]['value_mean']))
                bar_num = mi * len(PLOT_STATS) + si
                offset = (bar_num - (n_bars - 1) / 2) * width
                base = MODEL_COLORS[model]
                ax.bar(x + offset, values, width=width, color=shade(base, STAT_LEVELS[statistic]), edgecolor=base, hatch=STAT_HATCHES[statistic], linewidth=0.8, zorder=3)
        style_axis(ax, dimension, [SHORT_LABELS[s] for s in system_ids])
        ax.set_ylabel('Punteggio NRC-VAD', fontsize=10, labelpad=9)
    axes[-1].set_xlabel('System prompt', fontsize=10, labelpad=8)
    model_legend(fig, 0.005)
    neutral = '#666666'
    handles = [Patch(facecolor=shade(neutral, STAT_LEVELS[s]), edgecolor=neutral, hatch=STAT_HATCHES[s], label=STAT_LABELS[s]) for s in PLOT_STATS]
    fig.legend(handles=handles, title='Statistica', loc='lower center', bbox_to_anchor=(0.5, -0.043), ncol=3, frameon=False, fontsize=9, title_fontsize=9)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    fig.text(0.5, 0.966, subtitle, ha='center', va='center', fontsize=9.5, color='#555555')
    fig.subplots_adjust(left=0.085, right=0.985, top=0.925, bottom=0.18, hspace=0.4)
    return save_figure(fig, stem)

def plot_demographic(summary, demographic, groups, system_ids, title, subtitle, stem):
    models = ['Qwen Standard', 'Qwen Obliterated']
    group_style = {groups[0]: (1.0, ''), groups[1]: (0.58, '//')}
    x = np.arange(len(system_ids))
    width = 0.17 if len(system_ids) >= 8 else 0.18
    size = (13.0, 10.4) if len(system_ids) >= 8 else (9.8, 9.4)
    n_bars = len(models) * len(groups)
    fig, axes = plt.subplots(3, 1, figsize=size, sharey=True)
    for ax, dimension in zip(axes, DIMENSIONS):
        dim = summary[summary['dimension'] == dimension]
        for mi, model in enumerate(models):
            for gi, group in enumerate(groups):
                values = []
                for system_id in system_ids:
                    row = dim[(dim['model_label'] == model) & (dim[demographic] == group) & (dim['system_prompt_id'] == system_id)]
                    if len(row) != 1:
                        raise ValueError(f'Dato non univoco: {model}, {group}, {system_id}, {dimension}')
                    values.append(float(row.iloc[0]['mean_value']))
                bar_num = mi * len(groups) + gi
                offset = (bar_num - (n_bars - 1) / 2) * width
                level, hatch = group_style[group]
                base = MODEL_COLORS[model]
                ax.bar(x + offset, values, width=width, color=shade(base, level), edgecolor=base, hatch=hatch, linewidth=0.8, zorder=3)
        style_axis(ax, dimension, [SHORT_LABELS[s] for s in system_ids])
        ax.set_ylabel('Punteggio medio NRC-VAD', fontsize=10, labelpad=9)
    axes[-1].set_xlabel('System prompt', fontsize=10, labelpad=8)
    model_legend(fig, 0.005)
    neutral = '#666666'
    handles = [Patch(facecolor=shade(neutral, group_style[g][0]), edgecolor=neutral, hatch=group_style[g][1], label=g) for g in groups]
    fig.legend(handles=handles, title='Gruppo', loc='lower center', bbox_to_anchor=(0.5, -0.043), ncol=2, frameon=False, fontsize=9, title_fontsize=9)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    fig.text(0.5, 0.966, subtitle, ha='center', va='center', fontsize=9.5, color='#555555')
    fig.subplots_adjust(left=0.085, right=0.985, top=0.925, bottom=0.18, hspace=0.4)
    return save_figure(fig, stem)

def get_values(summary, system_id, column):
    values = {}
    for model, key in [('Qwen Standard', 'std'), ('Qwen Obliterated', 'obl')]:
        for dimension in DIMENSIONS:
            row = summary[(summary['model_label'] == model) & (summary['system_prompt_id'] == system_id) & (summary['dimension'] == dimension)]
            if len(row) != 1:
                raise ValueError(f'Valore non univoco: {model}, {system_id}, {dimension}')
            values[f'{key}_{dimension}'] = float(row.iloc[0][column])
    return values

def table_header(caption, label):
    return ['\\begin{table}[htbp]', '\\vspace{0.2cm}', '\\centering', '\\scriptsize', f'\\caption{{{caption}}}', f'\\label{{{label}}}', '\\resizebox{\\textwidth}{!}{%', '\\begin{tabular}{llrrrrrr}', '\\hline', '\\textbf{Prompt} & \\textbf{Orientamento} & \\multicolumn{3}{c}{\\textbf{Standard}} & \\multicolumn{3}{c}{\\textbf{Obliterated}} \\\\ ', ' & & \\textbf{V} & \\textbf{A} & \\textbf{D} & \\textbf{V} & \\textbf{A} & \\textbf{D} \\\\ ', '\\hline']

def table_section(lines, title, system_ids, summary, column):
    lines.append(f'\\multicolumn{{8}}{{l}}{{\\textbf{{{title}}}}} \\\\')
    for system_id in system_ids:
        v = get_values(summary, system_id, column)
        lines.append(f"\\texttt{{{tex_id(system_id)}}} & {PROMPT_LABELS[system_id]} & {fmt(v['std_valence'])} & {fmt(v['std_arousal'])} & {fmt(v['std_dominance'])} & {fmt(v['obl_valence'])} & {fmt(v['obl_arousal'])} & {fmt(v['obl_dominance'])} \\\\")
    lines.append('\\hline')

def save_table(lines, path):
    lines.extend(['\\end{tabular}%', '}', '\\vspace{0.3cm}', '\\end{table}'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path

def write_overall_table(summary):
    path = OUTPUT_DIR / 'rq3_overall_table.tex'
    caption = 'Variazioni medie dei punteggi NRC-VAD prodotte dai system prompt rispetto alla condizione \\texttt{NO\\_SYS}, calcolate entro professione e successivamente mediate assegnando lo stesso peso a ciascuna professione.'
    lines = table_header(caption, 'tab:rq3-overall')
    table_section(lines, 'System prompt di genere', GEN_PROMPTS, summary, 'mean_delta')
    table_section(lines, "System prompt relativi all'appartenenza razziale", RACE_PROMPTS, summary, 'mean_delta')
    return save_table(lines, path)

def write_demographic_table(gender_summary, race_summary):
    path = OUTPUT_DIR / 'rq3_demographic_table.tex'
    caption = "Differenze medie tra gruppi demografici nell'effetto dei system prompt rispetto a \\texttt{NO\\_SYS}: \\texttt{M-F} per il genere e \\texttt{W-NW} per l'appartenenza razziale percepita."
    lines = table_header(caption, 'tab:rq3-demographic')
    table_section(lines, 'System prompt di genere: differenza M-F', GEN_PROMPTS, gender_summary, 'mean_difference')
    table_section(lines, "System prompt relativi all'appartenenza razziale: differenza W-NW", RACE_PROMPTS, race_summary, 'mean_difference')
    return save_table(lines, path)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('\n======================================')
    print(' RQ3 - SYSTEM PROMPT E VAD')
    print('======================================')
    df = load_data()
    print(f'\nRighe occupation_neutral nelle 11 condizioni: {len(df):,}')
    df, completeness = select_complete_images(df)
    print('\nCompletezza:')
    for _, row in completeness.iterrows():
        print(f"  - {row['model_label']}: {row['images_complete']:,} immagini complete su {row['images_before']:,}")
    image_level = build_image_level(df)
    gender_eligible = eligible_professions(image_level, 'gender_perceived', ['M', 'F'])
    race_eligible = eligible_professions(image_level, 'race_perceived', ['W', 'NW'])
    abs_overall = absolute_overall_summary(image_level)
    abs_gender = absolute_demographic_summary(image_level, gender_eligible, 'gender_perceived', GEN_PLOT, 'gender')
    abs_race = absolute_demographic_summary(image_level, race_eligible, 'race_perceived', RACE_PLOT, 'race')
    delta_long = build_deltas(image_level)
    delta_overall = overall_delta_summary(delta_long)
    delta_gender = demographic_delta_summary(delta_long, gender_eligible, 'gender_perceived', ['M', 'F'], GEN_PROMPTS, 'gender')
    delta_race = demographic_delta_summary(delta_long, race_eligible, 'race_perceived', ['W', 'NW'], RACE_PROMPTS, 'race')
    overall_table = write_overall_table(delta_overall)
    demographic_table = write_demographic_table(delta_gender, delta_race)
    figures = []
    figures += plot_overall(abs_overall, GEN_PLOT, 'Statistiche NRC-VAD nei system prompt di genere', 'NO_SYS e G1-G7; valori assoluti medi tra professioni; prompt utente occupation_neutral', 'rq3_gender_system_absolute_statistics')
    figures += plot_overall(abs_overall, RACE_PLOT, "Statistiche NRC-VAD nei system prompt relativi all'appartenenza razziale", 'NO_SYS e R1-R3; valori assoluti medi tra professioni; prompt utente occupation_neutral', 'rq3_race_system_absolute_statistics')
    figures += plot_demographic(abs_gender, 'gender_perceived', ['M', 'F'], GEN_PLOT, 'System prompt di genere e genere percepito', 'NO_SYS e G1-G7; valori medi tra professioni; prompt utente occupation_neutral', 'rq3_gender_system_absolute_demographic')
    figures += plot_demographic(abs_race, 'race_perceived', ['W', 'NW'], RACE_PLOT, 'System prompt e appartenenza razziale percepita', 'NO_SYS e R1-R3; valori medi tra professioni; prompt utente occupation_neutral', 'rq3_race_system_absolute_demographic')
    print('\nProfessioni utilizzabili:')
    for label, eligible in [('M/F', gender_eligible), ('W/NW', race_eligible)]:
        counts = eligible.groupby('model_label')['profession'].nunique()
        for model in ['Qwen Standard', 'Qwen Obliterated']:
            print(f'  - {label} | {model}: {int(counts.get(model, 0))}')
    print('\nTabelle:')
    print(f'  - {overall_table.name}')
    print(f'  - {demographic_table.name}')
    print('\nGrafici:')
    for path in figures:
        print(f'  - {path.name}')
    print(f'\nOutput: {OUTPUT_DIR}')
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('\nERRORE')
        print(exc)
        sys.exit(1)

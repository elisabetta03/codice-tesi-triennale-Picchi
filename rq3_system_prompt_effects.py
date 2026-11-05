"""RQ3: system-prompt effects on NRC-VAD and demographic contrasts.

Final-thesis procedure:
- use occupation_neutral only;
- retain images with all five paraphrases in all 11 system conditions;
- average paraphrases at image level, then aggregate within profession;
- compare each system prompt with NO_SYS;
- for demographic analyses, compute M-F for G1-G7 and W-NW for R1-R3;
- give each profession equal weight;
- estimate percentile bootstrap 95% CIs using 10,000 profession resamples.
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
OUTPUT_DIR = ROOT / 'analysis_outputs' / 'rq3'
USER_PROMPT_CONDITION = 'occupation_neutral'
NO_SYS = 'NO_SYS'
EXPECTED_PARAPHRASES = 5
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
MODEL_ORDER = ['Qwen Standard', 'Qwen Obliterated']
GEN_PROMPTS = [f'SYS_GEN_{i:03d}' for i in range(1, 8)]
RACE_PROMPTS = [f'SYS_RACE_{i:03d}' for i in range(1, 4)]
SYSTEM_PROMPTS = [*GEN_PROMPTS, *RACE_PROMPTS]
ALL_SYSTEMS = [NO_SYS, *SYSTEM_PROMPTS]
SHORT_LABELS = {**{f'SYS_GEN_{i:03d}': f'G{i}' for i in range(1, 8)}, **{f'SYS_RACE_{i:03d}': f'R{i}' for i in range(1, 4)}}
ROW_LABELS = {
    'SYS_GEN_001': 'G1 neg. M+F', 'SYS_GEN_002': 'G2 neg. M', 'SYS_GEN_003': 'G3 neg. F',
    'SYS_GEN_004': 'G4 anti-stereo M+F', 'SYS_GEN_005': 'G5 anti-stereo M',
    'SYS_GEN_006': 'G6 anti-stereo F', 'SYS_GEN_007': 'G7 neutro genere',
    'SYS_RACE_001': 'R1 neg. NW', 'SYS_RACE_002': 'R2 anti-stereo NW', 'SYS_RACE_003': 'R3 neutro razza',
}
PROMPT_ORIENTATIONS = {
    'SYS_GEN_001': 'negativo verso uomini e donne', 'SYS_GEN_002': 'negativo verso gli uomini',
    'SYS_GEN_003': 'negativo verso le donne', 'SYS_GEN_004': 'contrasto degli stereotipi negativi verso uomini e donne',
    'SYS_GEN_005': 'contrasto degli stereotipi negativi verso gli uomini',
    'SYS_GEN_006': 'contrasto degli stereotipi negativi verso le donne',
    'SYS_GEN_007': 'neutro rispetto al genere', 'SYS_RACE_001': 'negativo verso i soggetti percepiti come non bianchi',
    'SYS_RACE_002': 'contrasto degli stereotipi razziali negativi',
    'SYS_RACE_003': "neutro rispetto all'appartenenza razziale percepita",
}
DIMENSIONS = ['valence', 'arousal', 'dominance']
DIM_LABELS = {'valence': 'V', 'arousal': 'A', 'dominance': 'D'}
VAD_COLUMNS = {d: f'vad_{d}_mean' for d in DIMENSIONS}


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


def fmt(value, digits=3):
    rounded = round(float(value), digits)
    if rounded == 0:
        return '0'
    return f'{rounded:+.{digits}f}'.replace('.', ',')


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f'Non trovo il dataset LONG:\n{INPUT_FILE}')
    columns = [
        'image', 'profession', 'model_variant', 'system_prompt_id', 'prompt_id',
        'prompt_condition', 'gender_perceived', 'race_perceived', *VAD_COLUMNS.values()
    ]
    df = pd.read_csv(INPUT_FILE, sep=';', usecols=columns, low_memory=False, encoding='utf-8-sig')
    df = df[(df['prompt_condition'] == USER_PROMPT_CONDITION) & df['system_prompt_id'].isin(ALL_SYSTEMS)].copy()
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
    return df


def select_complete_images(df):
    counts = df.groupby(['model_label', 'image', 'system_prompt_id'], observed=True).agg(
        n_rows=('prompt_id', 'size'), n_ids=('prompt_id', 'nunique')
    ).reset_index()
    counts['complete_condition'] = (
        (counts['n_rows'] == EXPECTED_PARAPHRASES) &
        (counts['n_ids'] == EXPECTED_PARAPHRASES)
    )
    images = counts.groupby(['model_label', 'image'], observed=True).agg(
        n_systems=('system_prompt_id', 'nunique'), n_complete=('complete_condition', 'sum')
    ).reset_index()
    complete = images[(images['n_systems'] == len(ALL_SYSTEMS)) & (images['n_complete'] == len(ALL_SYSTEMS))][['model_label', 'image']]
    filtered = df.merge(complete, on=['model_label', 'image'], how='inner')
    report = []
    for model in MODEL_ORDER:
        before = df.loc[df['model_label'] == model, 'image'].nunique()
        after = filtered.loc[filtered['model_label'] == model, 'image'].nunique()
        report.append({'model_label': model, 'images_before': int(before), 'images_complete': int(after), 'images_excluded': int(before-after)})
    save_csv(pd.DataFrame(report), 'rq3_completeness_report.csv')
    return filtered


def build_image_level(df):
    image_level = df.groupby(
        ['model_label', 'image', 'profession', 'gender_perceived', 'race_perceived', 'system_prompt_id'],
        as_index=False, observed=True
    ).agg(**{col: (col, 'mean') for col in VAD_COLUMNS.values()})
    save_csv(image_level, 'rq3_image_level.csv')
    return image_level


def overall_profession_contrasts(image_level):
    rows = []
    for dimension in DIMENSIONS:
        col = VAD_COLUMNS[dimension]
        profession_system = image_level.groupby(['model_label', 'profession', 'system_prompt_id'], observed=True)[col].mean().reset_index(name='value')
        baseline = profession_system[profession_system['system_prompt_id'] == NO_SYS][['model_label', 'profession', 'value']].rename(columns={'value': 'baseline'})
        current = profession_system[profession_system['system_prompt_id'] != NO_SYS].merge(baseline, on=['model_label', 'profession'], validate='many_to_one')
        current['contrast'] = current['value'] - current['baseline']
        current['dimension'] = dimension
        rows.append(current[['model_label', 'profession', 'system_prompt_id', 'dimension', 'contrast']])
    return pd.concat(rows, ignore_index=True)


def eligible_professions(image_level, demographic, groups):
    base = image_level[image_level['system_prompt_id'] == NO_SYS]
    counts = base.groupby(['model_label', 'profession', demographic], observed=True)['image'].nunique().reset_index(name='n_images')
    presence = counts.assign(present=lambda x: x['n_images'] > 0).pivot_table(
        index=['model_label', 'profession'], columns=demographic, values='present', aggfunc='max', fill_value=False
    ).reset_index()
    for group in groups:
        if group not in presence.columns:
            presence[group] = False
    return presence[presence[groups].all(axis=1)][['model_label', 'profession']]


def demographic_profession_contrasts(image_level):
    blocks = []
    for demographic, groups, systems in [
        ('gender_perceived', ['M', 'F'], GEN_PROMPTS),
        ('race_perceived', ['W', 'NW'], RACE_PROMPTS),
    ]:
        eligible = eligible_professions(image_level, demographic, groups)
        data = image_level[image_level['system_prompt_id'].isin([NO_SYS, *systems])].merge(eligible, on=['model_label', 'profession'], how='inner')
        for dimension in DIMENSIONS:
            col = VAD_COLUMNS[dimension]
            pg = data.groupby(['model_label', 'profession', demographic, 'system_prompt_id'], observed=True)[col].mean().reset_index(name='value')
            baseline = pg[pg['system_prompt_id'] == NO_SYS][['model_label', 'profession', demographic, 'value']].rename(columns={'value': 'baseline'})
            current = pg[pg['system_prompt_id'] != NO_SYS].merge(baseline, on=['model_label', 'profession', demographic], validate='many_to_one')
            current['delta'] = current['value'] - current['baseline']
            wide = current.pivot_table(index=['model_label', 'profession', 'system_prompt_id'], columns=demographic, values='delta').reset_index()
            wide['contrast'] = wide[groups[0]] - wide[groups[1]]
            wide['dimension'] = dimension
            wide['demographic_contrast'] = f'{groups[0]}-{groups[1]}'
            blocks.append(wide[['model_label', 'profession', 'system_prompt_id', 'dimension', 'demographic_contrast', 'contrast']])
    return pd.concat(blocks, ignore_index=True)


def bootstrap_summary(profession_level, filename):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    keys = ['model_label', 'system_prompt_id', 'dimension']
    extra = ['demographic_contrast'] if 'demographic_contrast' in profession_level.columns else []
    for group_key, group in profession_level.groupby(keys + extra, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = group['contrast'].to_numpy(dtype=float)
        boot = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
        for i in range(BOOTSTRAP_RESAMPLES):
            boot[i] = rng.choice(values, size=len(values), replace=True).mean()
        low, high = np.percentile(boot, [2.5, 97.5])
        row = dict(zip(keys + extra, group_key))
        row.update({
            'mean_contrast': float(values.mean()), 'ci_low': float(low), 'ci_high': float(high),
            'n_professions': int(len(values)), 'ci_excludes_zero': bool(low > 0 or high < 0)
        })
        rows.append(row)
    summary = pd.DataFrame(rows)
    save_csv(profession_level, filename.replace('_summary.csv', '_profession_contrasts.csv'))
    save_csv(summary, filename)
    return summary


def heatmap(summary, stem, title, colorbar_label, colorbar_digits):
    rows = SYSTEM_PROMPTS
    nrows = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 7.0), sharey=True)
    all_values = summary['mean_contrast'].to_numpy(dtype=float)
    vmax = max(abs(all_values.min()), abs(all_values.max()))
    if vmax == 0:
        vmax = 1e-6
    image = None
    for ax, model in zip(axes, MODEL_ORDER):
        matrix = np.full((nrows, len(DIMENSIONS)), np.nan)
        bold = np.zeros_like(matrix, dtype=bool)
        for i, system_id in enumerate(rows):
            for j, dimension in enumerate(DIMENSIONS):
                row = summary[(summary['model_label'] == model) & (summary['system_prompt_id'] == system_id) & (summary['dimension'] == dimension)]
                if len(row) != 1:
                    raise ValueError(f'Dato non univoco: {model}, {system_id}, {dimension}')
                matrix[i, j] = float(row.iloc[0]['mean_contrast'])
                bold[i, j] = bool(row.iloc[0]['ci_excludes_zero'])
        image = ax.imshow(matrix, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(model, fontsize=15, fontweight='bold', pad=10)
        ax.set_xticks(np.arange(3)); ax.set_xticklabels([DIM_LABELS[d] for d in DIMENSIONS], fontsize=11)
        ax.set_yticks(np.arange(nrows)); ax.set_yticklabels([ROW_LABELS[s] for s in rows], fontsize=10)
        ax.tick_params(length=0)
        for i in range(nrows):
            for j in range(3):
                value = matrix[i, j]
                norm = abs(value) / vmax
                ax.text(j, i, fmt(value), ha='center', va='center', fontsize=10.3,
                        fontweight='bold' if bold[i, j] else 'normal',
                        color='white' if norm > 0.62 else 'black')
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(title, fontsize=17, fontweight='bold', y=0.99)
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
    cbar.set_label(colorbar_label, fontsize=11, rotation=270, labelpad=17)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.{colorbar_digits}f}'.replace('.', ',')))
    cbar.ax.tick_params(labelsize=9)
    fig.text(0.02, 0.025, 'In grassetto: IC 95% bootstrap che non include lo zero.', fontsize=10)
    fig.subplots_adjust(left=0.17, right=0.88, top=0.88, bottom=0.08, wspace=0.02)
    return save_figure(fig, stem)


def write_prompt_abbreviations():
    path = OUTPUT_DIR / 'rq3_prompt_abbreviations_table.tex'
    lines = ['\\begin{table}[H]', '\\centering', '\\small', '\\begin{tabular}{lll}', '\\hline', '\\textbf{Sigla} & \\textbf{Prompt} & \\textbf{Orientamento} \\\\', '\\hline']
    for system_id in SYSTEM_PROMPTS:
        lines.append(f"{SHORT_LABELS[system_id]} & \\texttt{{{system_id.replace('_', '\\_')}}} & {PROMPT_ORIENTATIONS[system_id]} \\\\")
    lines += ['\\hline', '\\end{tabular}', '\\end{table}']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def write_summary_table(summary, filename, caption, label):
    path = OUTPUT_DIR / filename
    lines = ['\\begin{table}[H]', '\\centering', '\\scriptsize', f'\\caption{{{caption}}}', f'\\label{{{label}}}', '\\begin{tabular}{llrrrrrr}', '\\hline', '\\textbf{Prompt} & \\textbf{Orientamento} & \\multicolumn{3}{c}{\\textbf{Standard}} & \\multicolumn{3}{c}{\\textbf{Obliterated}} \\\\', ' & & \\textbf{V} & \\textbf{A} & \\textbf{D} & \\textbf{V} & \\textbf{A} & \\textbf{D} \\\\', '\\hline']
    for system_id in SYSTEM_PROMPTS:
        vals = {}
        for model, key in [('Qwen Standard', 'std'), ('Qwen Obliterated', 'obl')]:
            for dimension in DIMENSIONS:
                row = summary[(summary['model_label'] == model) & (summary['system_prompt_id'] == system_id) & (summary['dimension'] == dimension)]
                vals[f'{key}_{dimension}'] = float(row.iloc[0]['mean_contrast'])
        lines.append(f"{SHORT_LABELS[system_id]} & {PROMPT_ORIENTATIONS[system_id]} & {fmt(vals['std_valence'])} & {fmt(vals['std_arousal'])} & {fmt(vals['std_dominance'])} & {fmt(vals['obl_valence'])} & {fmt(vals['obl_arousal'])} & {fmt(vals['obl_dominance'])} \\\\")
    lines += ['\\hline', '\\end{tabular}', '\\end{table}']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('\n======================================')
    print(' RQ3 - SYSTEM PROMPT E VAD')
    print('======================================')
    image_level = build_image_level(select_complete_images(load_data()))
    overall_prof = overall_profession_contrasts(image_level)
    demo_prof = demographic_profession_contrasts(image_level)
    overall = bootstrap_summary(overall_prof, 'rq3_overall_bootstrap_summary.csv')
    demographic = bootstrap_summary(demo_prof, 'rq3_demographic_bootstrap_summary.csv')
    write_prompt_abbreviations()
    write_summary_table(overall, 'rq3_overall_table.tex', 'Variazioni medie rispetto a NO\\_SYS, calcolate entro professione e mediate assegnando lo stesso peso a ciascuna professione.', 'tab:rq3-overall')
    write_summary_table(demographic, 'rq3_demographic_table.tex', 'Differenze medie tra gruppi demografici nell’effetto dei system prompt rispetto a NO\\_SYS.', 'tab:rq3-demographic')
    heatmap(overall, 'rq3_system_effect_heatmap', 'Effetto dei system prompt rispetto a NO_SYS', 'Variazione rispetto a NO_SYS', 2)
    heatmap(demographic, 'rq3_demographic_effect_heatmap', "Differenze demografiche nell'effetto dei system prompt", 'Contrasto M-F / W-NW', 4)
    print(f'\nOutput: {OUTPUT_DIR}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('\nERRORE')
        print(exc)
        sys.exit(1)

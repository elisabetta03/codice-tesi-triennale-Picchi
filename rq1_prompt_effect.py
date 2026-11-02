"""RQ1: compare VAD statistics across baseline, occupational, positive, and negative prompts under NO_SYS."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / 'merged_outputs' / 'merged_long_qwen_obliterated.csv'
OUTPUT_DIR = PROJECT_DIR / 'analysis_outputs' / 'rq1'
SYSTEM_PROMPT_FILTER = 'NO_SYS'
EXPECTED_PARAPHRASES_PER_CONDITION = 5
EXPECTED_VAD_MIN = 0.0
EXPECTED_VAD_MAX = 1.0
PROMPT_ORDER = ['baseline_neutral', 'occupation_neutral', 'positive_modifier', 'negative_modifier']
PROMPT_LABELS = {'baseline_neutral': 'Baseline', 'occupation_neutral': 'Occupational', 'positive_modifier': 'Positive', 'negative_modifier': 'Negative'}
DIMENSIONS = ['valence', 'arousal', 'dominance']
DIMENSION_LABELS = {'valence': 'Valence', 'arousal': 'Arousal', 'dominance': 'Dominance'}
STATISTICS = ['mean', 'std', 'median', 'max']
PLOT_STATISTICS = ['mean', 'median', 'max']
PLOT_STAT_LABELS = {'mean': 'Mean', 'median': 'Median', 'max': 'Max'}
VAD_COLUMNS = [f'vad_{dimension}_{statistic}' for dimension in DIMENSIONS for statistic in STATISTICS]
MODEL_COLORS = {'Qwen Standard': '#3978B8', 'Qwen Obliterated': '#D66545'}
STAT_SHADE_LEVELS = {'mean': 1.0, 'median': 0.72, 'max': 0.48}
STAT_HATCHES = {'mean': '', 'median': '//', 'max': '..'}

def model_display_name(value):
    if 'OBLITERATED' in str(value).upper():
        return 'Qwen Obliterated'
    return 'Qwen Standard'

def model_sort_key(value):
    name = model_display_name(value)
    if name == 'Qwen Standard':
        return 0
    if name == 'Qwen Obliterated':
        return 1
    return 99

def save_csv(df, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, sep=';', index=False, encoding='utf-8-sig')
    return path

def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f'{stem}.png'
    pdf_path = OUTPUT_DIR / f'{stem}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    return (png_path, pdf_path)

def format_decimal(value, digits=3):
    return f'{value:.{digits}f}'.replace('.', ',')

def format_signed_decimal(value, digits=4):
    rounded = round(float(value), digits)
    if rounded == 0:
        return f'{0:.{digits}f}'.replace('.', ',')
    return f'{rounded:+.{digits}f}'.replace('.', ',')

def shade_color(color, level):
    rgb = np.array(to_rgb(color))
    shaded = 1 - level * (1 - rgb)
    return tuple(shaded)

def check_generation_parameters():
    parameter_columns = ['model_variant', 'model', 'decoding', 'temperature', 'top_p', 'seed']
    parameters = pd.read_csv(INPUT_FILE, sep=';', usecols=parameter_columns, low_memory=False, encoding='utf-8-sig')
    rows = []
    models = sorted(parameters['model_variant'].dropna().unique(), key=model_sort_key)
    for model in models:
        model_df = parameters[parameters['model_variant'] == model]
        for column in ['model', 'decoding', 'temperature', 'top_p', 'seed']:
            values = model_df[column].dropna().astype(str).unique().tolist()
            rows.append({'model_variant': model, 'model_label': model_display_name(model), 'parameter': column, 'n_unique': len(values), 'values': ' | '.join(values)})
            if len(values) != 1:
                raise ValueError(f'Il parametro {column} non è costante per {model_display_name(model)}: {values}')
    report = pd.DataFrame(rows)
    save_csv(report, 'generation_parameters_check.csv')
    for column in ['decoding', 'temperature', 'top_p', 'seed']:
        values_by_model = []
        for model in models:
            model_value = parameters.loc[parameters['model_variant'] == model, column].dropna().astype(str).unique()
            if len(model_value) == 1:
                values_by_model.append(model_value[0])
        if len(set(values_by_model)) > 1:
            raise ValueError(f'Il parametro {column} differisce tra i modelli: {values_by_model}')
    print()
    print('Parametri di generazione presenti nel merged:')
    for _, row in report.iterrows():
        print(f"  - {row['model_label']} | {row['parameter']} = {row['values']}")
    return report

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f'Non trovo il dataset LONG:\n{INPUT_FILE}')
    required_columns = ['image', 'profession', 'model_variant', 'system_prompt_id', 'prompt_id', 'prompt_condition', *VAD_COLUMNS]
    df = pd.read_csv(INPUT_FILE, sep=';', usecols=required_columns, low_memory=False, encoding='utf-8-sig')
    df = df[df['system_prompt_id'] == SYSTEM_PROMPT_FILTER].copy()
    df = df[df['prompt_condition'].isin(PROMPT_ORDER)].copy()
    found_conditions = set(df['prompt_condition'].dropna().unique())
    missing_conditions = [condition for condition in PROMPT_ORDER if condition not in found_conditions]
    if missing_conditions:
        raise ValueError('Mancano alcune condizioni RQ1:\n' + '\n'.join((f'  - {condition}' for condition in missing_conditions)))
    for column in VAD_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        if df[column].isna().any():
            n_missing = int(df[column].isna().sum())
            raise ValueError(f'La colonna {column} contiene {n_missing:,} valori mancanti o non numerici.')
        minimum = float(df[column].min())
        maximum = float(df[column].max())
        if minimum < EXPECTED_VAD_MIN - 1e-12 or maximum > EXPECTED_VAD_MAX + 1e-12:
            raise ValueError(f'{column}: intervallo osservato [{minimum:.6f}, {maximum:.6f}], fuori dalla scala 0-1.')
    return df

def select_complete_images(df):
    condition_counts = df.groupby(['model_variant', 'image', 'prompt_condition'], observed=True).agg(n_rows=('prompt_id', 'size'), n_prompt_ids=('prompt_id', 'nunique')).reset_index()
    condition_counts['condition_complete'] = (condition_counts['n_rows'] == EXPECTED_PARAPHRASES_PER_CONDITION) & (condition_counts['n_prompt_ids'] == EXPECTED_PARAPHRASES_PER_CONDITION)
    image_counts = condition_counts.groupby(['model_variant', 'image'], observed=True).agg(n_conditions=('prompt_condition', 'nunique'), n_complete_conditions=('condition_complete', 'sum')).reset_index()
    image_counts['image_complete'] = (image_counts['n_conditions'] == len(PROMPT_ORDER)) & (image_counts['n_complete_conditions'] == len(PROMPT_ORDER))
    complete_images = image_counts[image_counts['image_complete']][['model_variant', 'image']].copy()
    filtered = df.merge(complete_images, on=['model_variant', 'image'], how='inner')
    report_rows = []
    for model in sorted(df['model_variant'].unique(), key=model_sort_key):
        model_images_before = int(df.loc[df['model_variant'] == model, 'image'].nunique())
        model_images_after = int(filtered.loc[filtered['model_variant'] == model, 'image'].nunique())
        model_rows_before = int((df['model_variant'] == model).sum())
        model_rows_after = int((filtered['model_variant'] == model).sum())
        report_rows.append({'model_variant': model, 'model_label': model_display_name(model), 'images_before': model_images_before, 'images_complete': model_images_after, 'images_excluded': model_images_before - model_images_after, 'rows_before': model_rows_before, 'rows_after': model_rows_after})
    completeness_report = pd.DataFrame(report_rows)
    save_csv(completeness_report, 'rq1_completeness_report.csv')
    save_csv(filtered, 'rq1_no_sys_complete_responses.csv')
    return (filtered, completeness_report)

def build_image_level(df):
    image_level = df.groupby(['model_variant', 'image', 'profession', 'prompt_condition'], as_index=False, observed=True).agg(**{column: (column, 'mean') for column in VAD_COLUMNS})
    save_csv(image_level, 'rq1_image_level.csv')
    return image_level

def build_summary_table(image_level):
    rows = []
    models = sorted(image_level['model_variant'].unique(), key=model_sort_key)
    for model in models:
        model_df = image_level[image_level['model_variant'] == model]
        for dimension in DIMENSIONS:
            for condition in PROMPT_ORDER:
                subset = model_df[model_df['prompt_condition'] == condition]
                if subset.empty:
                    continue
                row = {'model_variant': model, 'model_label': model_display_name(model), 'dimension': dimension, 'dimension_label': DIMENSION_LABELS[dimension], 'prompt_condition': condition, 'prompt_label': PROMPT_LABELS[condition], 'n_images': int(subset['image'].nunique())}
                for statistic in STATISTICS:
                    column = f'vad_{dimension}_{statistic}'
                    row[statistic] = float(subset[column].mean())
                rows.append(row)
    summary = pd.DataFrame(rows)
    save_csv(summary, 'rq1_summary_table.csv')
    return summary

def build_plot_summary(image_level):
    rows = []
    models = sorted(image_level['model_variant'].unique(), key=model_sort_key)
    for dimension in DIMENSIONS:
        for condition in PROMPT_ORDER:
            for model in models:
                subset = image_level[(image_level['model_variant'] == model) & (image_level['prompt_condition'] == condition)]
                if subset.empty:
                    continue
                for statistic in PLOT_STATISTICS:
                    column = f'vad_{dimension}_{statistic}'
                    values = subset[column].astype(float).to_numpy()
                    rows.append({'dimension': dimension, 'condition': condition, 'model_variant': model, 'model_label': model_display_name(model), 'statistic': statistic, 'statistic_label': PLOT_STAT_LABELS[statistic], 'value_mean': float(np.mean(values)), 'value_sd': float(np.std(values, ddof=1) if len(values) > 1 else 0.0), 'n_images': int(subset['image'].nunique())})
    plot_summary = pd.DataFrame(rows)
    save_csv(plot_summary, 'rq1_plot_summary.csv')
    return plot_summary

def write_latex_table(summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / 'rq1_summary_table.tex'

    lines = []
    table_specs = [
        (
            'Qwen Standard',
            'standard',
            'il modello standard',
        ),
        (
            'Qwen Obliterated',
            'obliterated',
            'la variante \\textit{Obliterated}',
        ),
    ]

    for model_label, label_suffix, caption_model in table_specs:
        model_df = summary[summary['model_label'] == model_label]

        if model_df.empty:
            continue

        lines.extend([
            '\\begin{table}[H]',
            '\\centering',
            '\\footnotesize',
            (
                f'\\caption{{Statistiche NRC-VAD per {caption_model} '
                'nella condizione \\texttt{NO\\_SYS}.}'
            ),
            f'\\label{{tab:rq1-statistiche-{label_suffix}}}',
            '',
            '\\vspace{0.2cm}',
            '',
            '\\begin{tabular}{llcccc}',
            '\\hline',
            (
                '\\textbf{Dimensione} & '
                '\\textbf{Condizione} & '
                '\\textbf{Media} & '
                '\\textbf{Dev. std.} & '
                '\\textbf{Mediana} & '
                '\\textbf{Massimo} \\\\'
            ),
            '\\hline',
        ])

        for dimension in DIMENSIONS:
            dim_df = model_df[model_df['dimension'] == dimension]
            first_dimension_row = True

            for condition in PROMPT_ORDER:
                row_df = dim_df[
                    dim_df['prompt_condition'] == condition
                ]

                if row_df.empty:
                    continue

                row = row_df.iloc[0]
                dimension_cell = (
                    DIMENSION_LABELS[dimension]
                    if first_dimension_row
                    else ''
                )

                lines.append(
                    f"{dimension_cell} & "
                    f"{PROMPT_LABELS[condition]} & "
                    f"{format_decimal(row['mean'])} & "
                    f"{format_decimal(row['std'])} & "
                    f"{format_decimal(row['median'])} & "
                    f"{format_decimal(row['max'])} \\\\"
                )

                first_dimension_row = False

            lines.append('\\hline')

        lines.extend([
            '\\end{tabular}',
            '\\end{table}',
            '',
        ])

    path.write_text(
        '\n'.join(lines) + '\n',
        encoding='utf-8',
    )
    return path

def write_latex_values(summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / 'rq1_values.tex'
    model_codes = {'Qwen Standard': 'Std', 'Qwen Obliterated': 'Obl'}
    dimension_codes = {'valence': 'Val', 'arousal': 'Aro', 'dominance': 'Dom'}
    condition_codes = {'baseline_neutral': 'Base', 'occupation_neutral': 'Occ', 'positive_modifier': 'Pos', 'negative_modifier': 'Neg'}
    lines = ['% File generato automaticamente da rq1_prompt_effect.py', '% Non modificare manualmente i valori sottostanti.', '']
    for model_label, model_code in model_codes.items():
        model_df = summary[summary['model_label'] == model_label]
        if model_df.empty:
            continue
        n_values = model_df['n_images'].unique()
        if len(n_values) != 1:
            raise ValueError(f'Numero di immagini non costante per {model_label}: {n_values}')
        lines.append(f'\\newcommand{{\\RQ{model_code}N}}{{{int(n_values[0])}}}')
        for dimension, dimension_code in dimension_codes.items():
            dim_df = model_df[model_df['dimension'] == dimension].set_index('prompt_condition')
            for condition, condition_code in condition_codes.items():
                value = float(dim_df.loc[condition, 'mean'])
                lines.append(f'\\newcommand{{\\RQ{model_code}{dimension_code}{condition_code}}}{{{format_decimal(value)}}}')
            occ = float(dim_df.loc['occupation_neutral', 'mean'])
            pos = float(dim_df.loc['positive_modifier', 'mean'])
            neg = float(dim_df.loc['negative_modifier', 'mean'])
            delta_pos = format_signed_decimal(pos - occ, digits=4)
            delta_neg = format_signed_decimal(neg - occ, digits=4)
            lines.append(f'\\newcommand{{\\RQ{model_code}{dimension_code}DeltaPos}}{{{delta_pos}}}')
            lines.append(f'\\newcommand{{\\RQ{model_code}{dimension_code}DeltaNeg}}{{{delta_neg}}}')
        lines.append('')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path

def plot_grouped_vad_statistics(plot_summary):
    models = ['Qwen Standard', 'Qwen Obliterated']
    n_models = len(models)
    n_stats = len(PLOT_STATISTICS)
    n_bars = n_models * n_stats
    bar_width = 0.11
    x = np.arange(len(PROMPT_ORDER))
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 10.5), sharey=True)
    for ax, dimension in zip(axes, DIMENSIONS):
        dim_df = plot_summary[plot_summary['dimension'] == dimension]
        for model_index, model_label in enumerate(models):
            base_color = MODEL_COLORS[model_label]
            for stat_index, statistic in enumerate(PLOT_STATISTICS):
                means = []
                sds = []
                for condition in PROMPT_ORDER:
                    row = dim_df[(dim_df['condition'] == condition) & (dim_df['model_label'] == model_label) & (dim_df['statistic'] == statistic)]
                    if len(row) != 1:
                        raise ValueError(f'Combinazione non univoca nel dataset del grafico:\n{dimension}, {condition}, {model_label}, {statistic}')
                    means.append(float(row.iloc[0]['value_mean']))
                    sds.append(float(row.iloc[0]['value_sd']))
                bar_number = model_index * n_stats + stat_index
                offset = (bar_number - (n_bars - 1) / 2) * bar_width
                face_color = shade_color(base_color, STAT_SHADE_LEVELS[statistic])
                ax.bar(x + offset, means, width=bar_width, yerr=sds, capsize=3, color=face_color, edgecolor=base_color, hatch=STAT_HATCHES[statistic], linewidth=0.9, error_kw={'elinewidth': 0.9, 'ecolor': '#222222', 'capthick': 0.9}, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([PROMPT_LABELS[condition] for condition in PROMPT_ORDER], fontsize=10)
        ax.set_title(DIMENSION_LABELS[dimension], fontsize=14, fontweight='bold', pad=8)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis='y', color='#D8D8D8', linewidth=0.7, alpha=0.55, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#555555')
        ax.spines['bottom'].set_color('#555555')
        ax.tick_params(axis='y', labelsize=9)
        ax.set_ylabel('Punteggio NRC-VAD', fontsize=10, labelpad=9)
    axes[-1].set_xlabel('Condizione di prompt', fontsize=10, labelpad=8)
    model_legend_handles = [Patch(facecolor=MODEL_COLORS['Qwen Standard'], edgecolor=MODEL_COLORS['Qwen Standard'], label='Qwen Standard'), Patch(facecolor=MODEL_COLORS['Qwen Obliterated'], edgecolor=MODEL_COLORS['Qwen Obliterated'], label='Qwen Obliterated')]
    model_legend = fig.legend(handles=model_legend_handles, title='Modello', loc='lower center', bbox_to_anchor=(0.5, 0.045), ncol=2, frameon=False, fontsize=9, title_fontsize=9, handlelength=1.8, columnspacing=2.5)
    fig.add_artist(model_legend)
    statistic_legend_color = '#666666'
    statistic_legend_handles = []
    for statistic in PLOT_STATISTICS:
        statistic_legend_handles.append(Patch(facecolor=shade_color(statistic_legend_color, STAT_SHADE_LEVELS[statistic]), edgecolor=statistic_legend_color, hatch=STAT_HATCHES[statistic], linewidth=0.9, label=PLOT_STAT_LABELS[statistic]))
    fig.legend(handles=statistic_legend_handles, title='Statistica', loc='lower center', bbox_to_anchor=(0.5, 0.002), ncol=3, frameon=False, fontsize=9, title_fontsize=9, handlelength=1.8, columnspacing=2.2)
    fig.suptitle('Statistiche NRC-VAD per condizione di prompt e modello', fontsize=16, fontweight='bold', y=0.995)
    fig.text(0.5, 0.965, 'Media tra immagini; error bar = deviazione standard tra immagini; senza prompt di sistema', ha='center', va='center', fontsize=9.5, color='#555555')
    fig.subplots_adjust(left=0.11, right=0.97, top=0.925, bottom=0.13, hspace=0.4)
    return save_figure(fig, 'rq1_vad_statistics_grouped')

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print()
    print('======================================')
    print(' RQ1 - CONDIZIONE DI PROMPT E VAD')
    print('======================================')
    check_generation_parameters()
    df = load_data()
    print()
    print(f'Righe NO_SYS nelle quattro condizioni RQ1: {len(df):,}')
    df_complete, completeness_report = select_complete_images(df)
    print()
    print('Completezza per modello:')
    for _, row in completeness_report.iterrows():
        print(f"  - {row['model_label']}: {row['images_complete']:,} immagini complete su {row['images_before']:,}")
    image_level = build_image_level(df_complete)
    print()
    print(f'Celle modello x immagine x condizione: {len(image_level):,}')
    summary = build_summary_table(image_level)
    latex_table_path = write_latex_table(summary)
    latex_values_path = write_latex_values(summary)
    plot_summary = build_plot_summary(image_level)
    figure_png_path, figure_pdf_path = plot_grouped_vad_statistics(plot_summary)
    print()
    print('======================================')
    print(' RQ1 COMPLETATA')
    print('======================================')
    print()
    print(f'Output:\n{OUTPUT_DIR}')
    print()
    print('File prodotti:')
    for filename in ['generation_parameters_check.csv', 'rq1_completeness_report.csv', 'rq1_no_sys_complete_responses.csv', 'rq1_image_level.csv', 'rq1_summary_table.csv', 'rq1_plot_summary.csv']:
        print(f'  - {filename}')
    print(f'  - {latex_table_path.name}')
    print(f'  - {latex_values_path.name}')
    print(f'  - {figure_png_path.name}')
    print(f'  - {figure_pdf_path.name}')
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print()
        print('======================================')
        print(' ERRORE')
        print('======================================')
        print()
        print(exc)
        sys.exit(1)

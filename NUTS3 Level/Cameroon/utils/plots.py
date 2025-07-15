# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar
import math

# %%
def plot_monthly_yearly_horizontal_bars(
    df,
    cases_col,
    year_col='Year',
    month_col='Month',
    bar_height_scale=1.0,
    spacing_factor=1.5,
    title='Monthly Malaria Cases per Year (Horizontal)',
    xlabel='Cases',
    ylabel='Year',
    cmap='tab20',
    text_font_size = 7,
    x_ticks_rounding_factor = -2,
    x_ticks_percentage_split = 5,
    figsize=(15, 10),
    show_values=True
):
    """
    Plots a horizontal grouped bar chart of monthly cases per year.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        year_col (str): Column name for the year.
        month_col (str): Column name for the month (must be numeric 1–12).
        cases_col (str): Column name for the total cases.
        bar_height_scale (float): Scale of bar height (default 1.0 = standard).
        spacing_factor (float): Vertical spacing between year groups.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        figsize (tuple): Figure size.
        show_values (bool): Whether to label bar ends with case numbers.
    """
    # ────────────────
    # 1. Aggregate monthly totals
    # ────────────────
    df_monthly = (df.groupby([year_col, month_col])[cases_col]
                    .sum()
                    .reset_index())

    # Convert numeric month to abbreviated name
    df_monthly['Month_name'] = df_monthly[month_col].apply(lambda m: calendar.month_abbr[m])

    # Pivot: rows = year, columns = months
    month_order = list(calendar.month_abbr[1:])  # Jan to Dec
    pivot_df = (df_monthly.pivot(index=year_col, columns='Month_name', values=cases_col)
                          .reindex(columns=month_order)
                          .sort_index())

    # ────────────────
    # 2. Color Mapping
    # ────────────────
    cmap = plt.get_cmap(cmap)  # Can also try 'Set3', 'tab10', etc.
    month_colors = [cmap(i % cmap.N) for i in range(12)]
    month_colors_dict = {month: month_colors[i] for i, month in enumerate(month_order)}

    # ────────────────
    # 3. Plotting
    # ────────────────
    fig, ax = plt.subplots(figsize=figsize)

    years = pivot_df.index.tolist()
    months = pivot_df.columns.tolist()
    num_months = len(months)

    bar_height = bar_height_scale / num_months
    y_positions = np.arange(len(years)) * spacing_factor

    for i, month in enumerate(months):
        month_cases = pivot_df[month]
        offsets = y_positions + i * bar_height
        bars = ax.barh(offsets, month_cases, height=bar_height, label=month, color=month_colors_dict[month])

        if show_values:
            for bar in bars:
                width = bar.get_width()
                y = bar.get_y() + bar.get_height() / 2
                ax.text(width + max(pivot_df.max()) * 0.01, y, f'{int(width):,}', va='center', fontsize=text_font_size)

    # X-axis ticks
    x_max = max(pivot_df.max())
    tick_step = round((x_ticks_percentage_split * x_max / 100), x_ticks_rounding_factor)
    ax.set_xticks(np.arange(0, x_max + math.ceil(10 * x_max / 100), step=tick_step))
    plt.xticks(rotation=45, ha='right')

    # Y-axis ticks
    ax.set_yticks(y_positions + bar_height * (num_months - 1) / 2)
    ax.set_yticklabels(years, fontsize=14)

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=20)

    ax.legend(title='Month', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def plot_grouped_barh(df, group_col, value_col, 
                                xlabel='Cases', ylabel=None, title=None, 
                                figsize=(15, 10), text_size=12, rotate_xticks=True):
    """
    Plot a horizontal bar chart of total hospitalizations grouped by a specified column.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing hospitalization data
        group_col (str): Column name to group by (e.g., 'Year', 'Month', 'GID3')
        value_col (str): Column name of values to sum
        xlabel (str): Label for the x-axis (default is 'Cases')
        ylabel (str): Label for the y-axis (default is group_col)
        title (str): Title of the plot
        figsize (tuple): Figure size
        text_size (int): Font size for text annotations
        rotate_xticks (bool): Whether to rotate x-axis tick labels
    """
    # Handle grouping and sorting
    if df[group_col].dtype == object:
        groups = sorted(df[group_col].dropna().unique())
    else:
        groups = sorted(df[group_col].dropna().unique().tolist())
    
    cases = [df.query(f"{group_col} == @group")[[value_col]].sum().values[0] for group in groups]
    
    # Sort if groups are not inherently ordered (e.g., GID3)
    if not np.issubdtype(df[group_col].dtype, np.number):
        cases, groups = zip(*sorted(zip(cases, groups)))

    plt.figure(figsize=figsize, facecolor='w', edgecolor='b')
    plt.barh(groups, cases)

    max_case = max(cases)
    xticks_step = round((5 * max_case / 100), -2)
    xticks_max = max_case + math.ceil(10 * max_case / 100)

    plt.xticks(np.arange(0, xticks_max + 1, step=xticks_step), size=14)
    
    if df[group_col].dtype == object:
        plt.yticks(groups, size=14)
    else:
        plt.yticks(np.arange(min(groups), max(groups) + 1), size=14)

    if rotate_xticks:
        plt.xticks(rotation=45, ha='right')

    plt.ylabel(ylabel or group_col, size=18)
    plt.xlabel(xlabel, size=18)
    plt.title(title or f'Total {value_col} per {group_col}', size=22)

    for group, case in zip(groups, cases):
        plt.text(case + 0.15, group, str(case), size=text_size)

    ax = plt.gca()
    plt.tight_layout()
    ax.xaxis.grid(True)
    plt.show()



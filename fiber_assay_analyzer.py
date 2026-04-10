"""
Fiber Assay Analyzer

GUI-based tool for DNA fiber assay analysis using
stratified Wilcoxon (van Elteren–style) permutation tests.

Features:
- Replicate-aware (biological blocking)
- Planned pairwise contrasts
- Multiple testing correction (Holm / Bonferroni / FDR)
- Effect size reporting (median-of-medians, fold-change, log2 FC)
- Export of results, config, and run log

Author: Sébastien Terreau
Year: 2026
License: MIT
DOI: https://doi.org/10.5281/zenodo.19500565
"""

import os
import re
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from scipy.stats import rankdata

# ----------------------------
# Utilities
# ----------------------------

def clean_headers(headers: List[str]) -> List[str]:
    """Strip whitespace; replace non [0-9a-zA-Z_] with '_'; collapse repeats; trim ends."""
    cleaned = []
    for col in headers:
        col = ("" if col is None else str(col))
        col = col.strip()
        col = re.sub(r'[^0-9a-zA-Z_]+', '_', col)
        col = re.sub(r'_+', '_', col).strip('_')
        cleaned.append(col)
    return cleaned


def load_table_any(path: str) -> pd.DataFrame:
    """
    Load CSV or XLSX. Drop fully empty rows/cols. Clean headers.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".xlsx", ".xlsm", ".xltx", ".xltm"]:
            df = pd.read_excel(path)
        elif ext == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Use CSV or XLSX.")
    except ImportError as e:
        raise ImportError("Reading .xlsx requires 'openpyxl'. Install it and retry.") from e

    # Drop fully empty rows/cols
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    # Clean headers
    df.columns = clean_headers(list(df.columns))
    return df


def write_run_log_txt(path: str, log_text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(log_text)


def write_config_json(path: str, config: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class AnalysisOptions:
    data_are_ratios: bool = False
    log_transform: bool = False       # affects effect-size display; p-values unaffected (rank-based)
    alpha: float = 0.05
    correction_method: str = "Holm"   # "None", "Holm", "Bonferroni", "FDR"
    omnibus: bool = False             # gatekeeping (optional); runs Kruskal–Wallis per replicate then meta-combines (see below)
    log_base: str = "natural"         # or "log2"
    n_permutations: int = 10000
    random_seed: int = 0


@dataclass
class ExperimentDesign:
    input_file: str
    columns: List[str]
    groups: List[str]
    replicates: List[str]
    column_to_group: Dict[str, str]
    column_to_replicate: Dict[str, str]
    planned_contrasts: List[Tuple[str, str]]  # list of (groupA, groupB)
    counts_by_cell: Dict[str, Dict[str, int]] # replicate -> group -> non-null count


# ----------------------------
# Stats: Stratified Wilcoxon (van Elteren–style) via permutations
# ----------------------------
# Omnibus: Stratified rank-based ANOVA (ART-like) for group effect
# ----------------------------
def omnibus_stratified_rank_anova(df_long: pd.DataFrame):
    """
    Omnibus test: is there any *group* effect after adjusting for replicate/block?
    Strategy:
      1) Align for the block effect by subtracting the within-replicate median.
      2) Rank the aligned responses across all observations (average ranks for ties).
      3) Fit rank ~ C(Group) + C(Replicate) via OLS and return an F-test for Group.
         If statsmodels is unavailable, fall back to per-replicate Kruskal–Wallis
         and combine p-values with Fisher's method.
    Returns dict with keys: {'method', 'stat', 'df1', 'df2', 'p_value'}
    """
    # Guard: need at least 2 groups and 1 replicate
    groups = df_long['Group'].unique().tolist()
    reps = df_long['Replicate'].unique().tolist()
    if len(groups) < 2 or len(reps) < 1:
        return {'method': 'omnibus: not applicable', 'stat': float('nan'),
                'df1': float('nan'), 'df2': float('nan'), 'p_value': float('nan')}

    # 1) align by replicate median
    df = df_long.copy()
    med_by_rep = df.groupby('Replicate')['Value'].transform('median')
    df['Aligned'] = df['Value'] - med_by_rep

    # 2) ranks of aligned values
    from scipy.stats import rankdata
    df['RankAligned'] = rankdata(df['Aligned'].to_numpy(), method='average')

    # Try statsmodels OLS on ranks
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        model = smf.ols('RankAligned ~ C(Group) + C(Replicate)', data=df).fit()
        an = anova_lm(model, typ=2)
        if 'C(Group)' in an.index:
            row = an.loc['C(Group)']
            return {'method': 'ART-like ANOVA on aligned ranks (controls for replicate)',
                    'stat': float(row['F']), 'df1': float(row['df']),
                    'df2': float(model.df_resid), 'p_value': float(row['PR(>F)'])}
    except Exception:
        pass  # fallback below

    # Fallback: per-replicate Kruskal–Wallis, Fisher combine
    from scipy.stats import kruskal, chi2
    pvals = []
    for r, sub in df.groupby('Replicate'):
        # only include replicates with ≥2 groups
        if sub['Group'].nunique() >= 2:
            arrays = [g['Aligned'].dropna().to_numpy() for _, g in sub.groupby('Group')]
            arrays = [a for a in arrays if len(a) > 0]
            if len(arrays) >= 2:
                try:
                    H, p = kruskal(*arrays)
                except Exception:
                    continue
                pvals.append(p)
    if not pvals:
        return {'method': 'omnibus: insufficient data', 'stat': float('nan'),
                'df1': float('nan'), 'df2': float('nan'), 'p_value': float('nan')}
    # Fisher's method
    from math import log
    X = -2.0 * sum(log(p) for p in pvals)
    df_chi = 2 * len(pvals)
    from scipy.stats import chi2
    p_comb = float(chi2.sf(X, df_chi))
    return {'method': 'Replicate-wise Kruskal–Wallis combined by Fisher',
            'stat': float(X), 'df1': float(df_chi), 'df2': float('nan'),
            'p_value': p_comb}

# ----------------------------

def _reshape_long(df_wide: pd.DataFrame,
                  col_to_group: dict,
                  col_to_repl: dict) -> pd.DataFrame:
    """
    Wide -> long with columns: value, group, replicate_id, column_name.
    Drops NaNs per column.
    """
    rows = []
    for col in df_wide.columns:
        grp = col_to_group[col]
        rep = col_to_repl[col]
        series = pd.to_numeric(df_wide[col], errors="coerce").dropna()
        for v in series.values:
            rows.append((float(v), grp, rep, col))
    long_df = pd.DataFrame(rows, columns=["value", "group", "replicate_id", "column"])
    return long_df


def _maybe_log(vals: np.ndarray, use_log: bool, base: str) -> np.ndarray:
    if not use_log:
        return vals
    # Guard against nonpositive ratios
    safe = vals.copy()
    safe = safe[np.isfinite(safe)]
    if (safe <= 0).any():
        raise ValueError("Log-transform selected but nonpositive values detected.")
    if base == "log2":
        return np.log2(vals)
    return np.log(vals)


def _mannwhitney_U(x: np.ndarray, gA_mask: np.ndarray) -> Tuple[float, int, int]:
    """U for group A within a single stratum (no ties correction needed for permutation)."""
    r = rankdata(x, method='average')
    n1 = int(gA_mask.sum())
    n2 = int((~gA_mask).sum())
    U = r[gA_mask].sum() - n1*(n1+1)/2.0
    return float(U), n1, n2


def _van_elteren_stat_one_contrast(df_long: pd.DataFrame,
                                   groupA: str,
                                   groupB: str) -> Tuple[float, Dict[str, Tuple[float,int,int]]]:
    """
    Compute the sum of per-replicate U statistics for A vs B.
    Returns total_U and a dict repl -> (U, nA, nB).
    """
    total_U = 0.0
    per_repl = {}
    for repl, df_r in df_long.groupby("replicate_id", sort=False):
        sub = df_r[df_r["group"].isin([groupA, groupB])]
        if sub.empty:
            continue
        x = sub["value"].to_numpy()
        gA = (sub["group"].to_numpy() == groupA)
        if gA.sum() == 0 or (~gA).sum() == 0:
            continue
        U, nA, nB = _mannwhitney_U(x, gA)
        total_U += U
        per_repl[repl] = (U, nA, nB)
    return total_U, per_repl


def _permute_within_repl(df_long: pd.DataFrame,
                         groupA: str,
                         groupB: str,
                         rng: np.random.Generator) -> float:
    """
    One permutation replicate: shuffle A/B labels within each replicate,
    preserving replicate-specific sample sizes, then return total U.
    """
    total_U_perm = 0.0
    for repl, df_r in df_long.groupby("replicate_id", sort=False):
        sub = df_r[df_r["group"].isin([groupA, groupB])]
        if sub.empty:
            continue
        g = sub["group"].to_numpy()
        x = sub["value"].to_numpy()

        # Keep the same counts of A and B in this replicate
        nA = np.sum(g == groupA)
        n = len(g)
        # Randomly choose which positions become A
        idx = np.arange(n)
        rng.shuffle(idx)
        gA_mask = np.zeros(n, dtype=bool)
        gA_mask[idx[:nA]] = True

        U, _, _ = _mannwhitney_U(x, gA_mask)
        total_U_perm += U

    return total_U_perm


def van_elteren_permutation_pvalue(df_long: pd.DataFrame,
                                   groupA: str,
                                   groupB: str,
                                   n_permutations: int = 10000,
                                   random_state: int = 0,
                                   alternative: str = "two-sided"
                                   ) -> Tuple[float, float, Dict[str, Tuple[float,int,int]]]:
    """
    Returns: (observed_total_U, p_value, per_repl_info)
    alternative: "two-sided" | "greater" (A>B) | "less" (A<B)
    """
    rng = np.random.default_rng(random_state)
    U_obs, per_repl = _van_elteren_stat_one_contrast(df_long, groupA, groupB)

    perms = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perms[i] = _permute_within_repl(df_long, groupA, groupB, rng)

    if alternative == "two-sided":
        # Symmetric around mean; use absolute deviation from perm mean
        center = perms.mean()
        p = (np.sum(np.abs(perms - center) >= np.abs(U_obs - center)) + 1.0) / (n_permutations + 1.0)
    elif alternative == "greater":
        p = (np.sum(perms >= U_obs) + 1.0) / (n_permutations + 1.0)
    else:  # "less"
        p = (np.sum(perms <= U_obs) + 1.0) / (n_permutations + 1.0)

    return U_obs, float(p), per_repl


def _median_of_medians(df_long: pd.DataFrame, group: str) -> float:
    """Median within each replicate, then median across replicates."""
    meds = []
    for _, df_r in df_long.groupby("replicate_id", sort=False):
        vals = df_r.loc[df_r["group"] == group, "value"].to_numpy()
        if len(vals) > 0:
            meds.append(np.median(vals))
    if len(meds) == 0:
        return np.nan
    return float(np.median(meds))


def _common_language_effect(df_long: pd.DataFrame, groupA: str, groupB: str) -> float:
    """
    Approximate P(X_A > X_B) pooled across replicates using ranks.
    For each replicate, compute U / (nA*nB), then average across replicates.
    """
    probs = []
    for _, df_r in df_long.groupby("replicate_id", sort=False):
        sub = df_r[df_r["group"].isin([groupA, groupB])]
        if sub.empty:
            continue
        x = sub["value"].to_numpy()
        gA = (sub["group"].to_numpy() == groupA)
        nA = gA.sum()
        nB = (~gA).sum()
        if nA == 0 or nB == 0:
            continue
        U, _, _ = _mannwhitney_U(x, gA)
        probs.append(U / (nA * nB))
    if not probs:
        return np.nan
    return float(np.mean(probs))


def _apply_correction(pvals: List[float], method: str, alpha: float) -> List[float]:
    """
    Returns adjusted p-values (same order). Simple implementations; for a few tests this is fine.
    """
    method = method.lower()
    m = len(pvals)
    if m <= 1 or method == "none":
        return pvals

    if method == "bonferroni":
        return [min(1.0, p * m) for p in pvals]

    if method == "holm":
        # step-down Holm
        order = np.argsort(pvals)
        adj = [0.0] * m
        running = 0.0
        for k, i in enumerate(order):
            adj_p = (m - k) * pvals[i]
            running = max(running, adj_p)
            adj[i] = min(1.0, running)
        return adj

    if method in ("fdr", "bh"):
        # Benjamini-Hochberg
        order = np.argsort(pvals)
        adj = [0.0] * m
        prev = 1.0
        for rank, i in enumerate(order[::-1], start=1):
            p = pvals[i]
            adj_p = min(prev, p * m / (m - rank + 1))
            adj[i] = adj_p
            prev = adj_p
        return adj

    # Fallback
    return pvals


# ----------------------------
# GUI App
# ----------------------------

class FiberAssayGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fiber Assay Analyzer — Design & Options")
        self.geometry("1100x780")

        self.df: Optional[pd.DataFrame] = None
        self.input_file: Optional[str] = None

        # Stateful GUI data
        self.group_count_var = tk.IntVar(value=2)
        self.group_name_vars: List[tk.StringVar] = []
        self.group_names_current: List[str] = []

        self.repl_count_var = tk.IntVar(value=3)
        self.repl_name_vars: List[tk.StringVar] = []
        self.repl_names_current: List[str] = []

        self.column_group_vars: Dict[str, tk.StringVar] = {}
        self.column_repl_vars: Dict[str, tk.StringVar] = {}

        self.contrast_vars: Dict[Tuple[str, str], tk.BooleanVar] = {}

        # Options
        self.opt_data_are_ratios = tk.BooleanVar(value=False)
        self.opt_log_transform = tk.BooleanVar(value=False)
        self.opt_alpha = tk.StringVar(value="0.05")
        self.opt_correction = tk.StringVar(value="Holm")
        self.opt_omnibus = tk.BooleanVar(value=False)    # (optional gatekeeping switch; see note)
        self.opt_log_base = tk.StringVar(value="natural")
        self.opt_n_perm = tk.StringVar(value="50000")
        self.opt_seed = tk.StringVar(value="0")

        # Build layout
        self._build_menu()
        self._build_tabs()

    # ---------- Menu ----------

    def on_close(self):
        """Graceful shutdown (works well in Spyder/IPython)."""
        try:
            self.quit()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open CSV/XLSX…", command=self.on_open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    # ---------- Tabs ----------

    def _build_tabs(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)

        self.tab_input = ttk.Frame(nb)
        self.tab_groups = ttk.Frame(nb)
        self.tab_replicates = ttk.Frame(nb)
        self.tab_mapping = ttk.Frame(nb)
        self.tab_contrasts = ttk.Frame(nb)
        self.tab_options = ttk.Frame(nb)
        self.tab_review = ttk.Frame(nb)

        nb.add(self.tab_input, text="1) Input")
        nb.add(self.tab_groups, text="2) Groups")
        nb.add(self.tab_replicates, text="3) Replicates")
        nb.add(self.tab_mapping, text="4) Map Columns")
        nb.add(self.tab_contrasts, text="5) Contrasts")
        nb.add(self.tab_options, text="6) Options")
        nb.add(self.tab_review, text="7) Review & Run")

        self._build_tab_input()
        self._build_tab_groups()
        self._build_tab_replicates()
        self._build_tab_mapping()
        self._build_tab_contrasts()
        self._build_tab_options()
        self._build_tab_review()

    # ---------- Tab 1: Input ----------

    def _build_tab_input(self):
        frm = self.tab_input
        pad = {"padx": 10, "pady": 8}

        ttk.Label(frm, text="Select an input file (CSV or XLSX) from the working directory.").grid(row=0, column=0, sticky="w", **pad)
        ttk.Button(frm, text="Open…", command=self.on_open_file).grid(row=0, column=1, sticky="w", **pad)

        self.input_path_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.input_path_var, width=90).grid(row=1, column=0, columnspan=2, sticky="w", **pad)

        self.columns_listbox = tk.Listbox(frm, height=16, width=80, exportselection=False)
        ttk.Label(frm, text="Detected columns (after cleaning):").grid(row=2, column=0, sticky="w", **pad)
        self.columns_listbox.grid(row=3, column=0, columnspan=2, sticky="w", **pad)

    def on_open_file(self):
        path = filedialog.askopenfilename(
            title="Open data file",
            initialdir=os.getcwd(),
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx *.xlsm *.xltx *.xltm"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            df = load_table_any(path)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        self.df = df
        self.input_file = os.path.basename(path)
        self.input_path_var.set(path)

        # populate column list
        self.columns_listbox.delete(0, tk.END)
        for c in df.columns.tolist():
            self.columns_listbox.insert(tk.END, c)

        # reset downstream state
        self._reset_all_design()

    # ---------- Tab 2: Groups ----------

    def _build_tab_groups(self):
        frm = self.tab_groups
        pad = {"padx": 10, "pady": 6}

        ttk.Label(frm, text="How many groups (conditions)?").grid(row=0, column=0, sticky="w", **pad)
        sp = ttk.Spinbox(frm, from_=2, to=20, textvariable=self.group_count_var, width=5, command=self._refresh_group_names)
        sp.grid(row=0, column=1, sticky="w", **pad)

        self.group_names_frame = ttk.Frame(frm)
        self.group_names_frame.grid(row=1, column=0, columnspan=3, sticky="w", **pad)

        ttk.Button(frm, text="Apply group count", command=self._refresh_group_names).grid(row=0, column=2, sticky="w", **pad)

        self._refresh_group_names()

    def _refresh_group_names(self):
        for w in self.group_names_frame.winfo_children():
            w.destroy()
        self.group_name_vars = []
        for i in range(self.group_count_var.get()):
            var = tk.StringVar(value=f"Group_{i+1}")
            self.group_name_vars.append(var)
            ttk.Label(self.group_names_frame, text=f"Name for Group {i+1}:").grid(row=i, column=0, padx=6, pady=4, sticky="w")
            ttk.Entry(self.group_names_frame, textvariable=var, width=24).grid(row=i, column=1, padx=6, pady=4, sticky="w")

        ttk.Button(self.group_names_frame, text="Confirm names", command=self._confirm_group_names).grid(
            row=self.group_count_var.get(), column=0, columnspan=2, padx=6, pady=8, sticky="w"
        )

    def _confirm_group_names(self):
        names = [v.get().strip() for v in self.group_name_vars]
        if len(set(names)) != len(names) or any(n == "" for n in names):
            messagebox.showerror("Invalid names", "Group names must be unique and non-empty.")
            return
        self.group_names_current = names
        messagebox.showinfo("Groups", f"Groups set: {', '.join(names)}")
        self._rebuild_mapping_panel()
        self._rebuild_contrasts_panel()

    # ---------- Tab 3: Replicates ----------

    def _build_tab_replicates(self):
        frm = self.tab_replicates
        pad = {"padx": 10, "pady": 6}

        ttk.Label(frm, text="How many biological replicates (experiments)?").grid(row=0, column=0, sticky="w", **pad)
        sp = ttk.Spinbox(frm, from_=1, to=20, textvariable=self.repl_count_var, width=5, command=self._refresh_repl_names)
        sp.grid(row=0, column=1, sticky="w", **pad)

        self.repl_names_frame = ttk.Frame(frm)
        self.repl_names_frame.grid(row=1, column=0, columnspan=3, sticky="w", **pad)

        ttk.Button(frm, text="Apply replicate count", command=self._refresh_repl_names).grid(row=0, column=2, sticky="w", **pad)

        self._refresh_repl_names()

    def _refresh_repl_names(self):
        for w in self.repl_names_frame.winfo_children():
            w.destroy()
        self.repl_name_vars = []
        for i in range(self.repl_count_var.get()):
            var = tk.StringVar(value=f"Rep{i+1}")
            self.repl_name_vars.append(var)
            ttk.Label(self.repl_names_frame, text=f"Name for Replicate {i+1}:").grid(row=i, column=0, padx=6, pady=4, sticky="w")
            ttk.Entry(self.repl_names_frame, textvariable=var, width=24).grid(row=i, column=1, padx=6, pady=4, sticky="w")

        ttk.Button(self.repl_names_frame, text="Confirm replicate names", command=self._confirm_repl_names).grid(
            row=self.repl_count_var.get(), column=0, columnspan=2, padx=6, pady=8, sticky="w"
        )

    def _confirm_repl_names(self):
        names = [v.get().strip() for v in self.repl_name_vars]
        if len(set(names)) != len(names) or any(n == "" for n in names):
            messagebox.showerror("Invalid names", "Replicate names must be unique and non-empty.")
            return
        self.repl_names_current = names
        messagebox.showinfo("Replicates", f"Replicates set: {', '.join(names)}")
        self._rebuild_mapping_panel()

    # ---------- Tab 4: Map Columns ----------

    def _build_tab_mapping(self):
        self.mapping_container = ttk.Frame(self.tab_mapping)
        self.mapping_container.pack(fill=tk.BOTH, expand=True)
        self._rebuild_mapping_panel()

    def _rebuild_mapping_panel(self):
        for w in self.mapping_container.winfo_children():
            w.destroy()

        if self.df is None:
            ttk.Label(self.mapping_container, text="Load a file first (Tab 1).").pack(anchor="w", padx=10, pady=10)
            return
        if not self.group_names_current:
            ttk.Label(self.mapping_container, text="Define and confirm group names (Tab 2).").pack(anchor="w", padx=10, pady=10)
            return
        if not self.repl_names_current:
            ttk.Label(self.mapping_container, text="Define and confirm replicate names (Tab 3).").pack(anchor="w", padx=10, pady=10)
            return

        ttk.Label(self.mapping_container, text="Assign each column to a GROUP and a REPLICATE:").pack(anchor="w", padx=10, pady=10)

        # Table of comboboxes
        table = ttk.Frame(self.mapping_container)
        table.pack(anchor="w", padx=10, pady=6)

        headers = self.df.columns.tolist()
        self.column_group_vars = {}
        self.column_repl_vars = {}
        for r, col in enumerate(headers):
            ttk.Label(table, text=col, width=40).grid(row=r, column=0, sticky="w", padx=4, pady=2)

            var_g = tk.StringVar(value=self.group_names_current[0])
            cb_g = ttk.Combobox(table, textvariable=var_g, values=self.group_names_current, width=20, state="readonly")
            cb_g.grid(row=r, column=1, sticky="w", padx=4, pady=2)
            self.column_group_vars[col] = var_g

            var_r = tk.StringVar(value=self.repl_names_current[0])
            cb_r = ttk.Combobox(table, textvariable=var_r, values=self.repl_names_current, width=14, state="readonly")
            cb_r.grid(row=r, column=2, sticky="w", padx=4, pady=2)
            self.column_repl_vars[col] = var_r

        ttk.Button(self.mapping_container, text="Validate mapping", command=self._validate_mapping).pack(anchor="w", padx=10, pady=10)

    def _validate_mapping(self):
        if not self.column_group_vars or not self.column_repl_vars:
            messagebox.showerror("Mapping", "No columns to map.")
            return

        # Check that each (replicate, group) has >= 1 column
        counts = {(rep, grp): 0 for rep in self.repl_names_current for grp in self.group_names_current}
        for col in self.df.columns:
            grp = self.column_group_vars[col].get()
            rep = self.column_repl_vars[col].get()
            counts[(rep, grp)] += 1

        # Warn if any replicate lacks a key group (not fatal, but typical designs have all groups in each replicate)
        missing = [(rep, grp) for (rep, grp), c in counts.items() if c == 0]
        if missing:
            msg = "Warning: Some (replicate, group) combinations have 0 columns:\n" + \
                  "\n".join([f"- {rep} / {grp}" for rep, grp in missing]) + \
                  "\nProceed if this is intentional."
            messagebox.showwarning("Mapping", msg)
        else:
            messagebox.showinfo("Mapping", "Column → (group, replicate) mapping looks valid.")

    # ---------- Tab 5: Contrasts ----------

    def _build_tab_contrasts(self):
        self.contrasts_container = ttk.Frame(self.tab_contrasts)
        self.contrasts_container.pack(fill=tk.BOTH, expand=True)
        self._rebuild_contrasts_panel()

    def _rebuild_contrasts_panel(self):
        for w in self.contrasts_container.winfo_children():
            w.destroy()

        if not self.group_names_current:
            ttk.Label(self.contrasts_container, text="Define and confirm group names (Tab 2).").pack(anchor="w", padx=10, pady=10)
            return

        names = self.group_names_current
        ttk.Label(self.contrasts_container, text="Select planned pairwise contrasts:").pack(anchor="w", padx=10, pady=10)

        grid = ttk.Frame(self.contrasts_container)
        grid.pack(anchor="w", padx=10, pady=6)

        # Header row
        ttk.Label(grid, text="", width=16).grid(row=0, column=0)
        for j, name in enumerate(names, start=1):
            ttk.Label(grid, text=name, width=16).grid(row=0, column=j)

        self.contrast_vars = {}
        # Upper triangle with checkboxes
        for i, gi in enumerate(names):
            ttk.Label(grid, text=gi, width=16).grid(row=i+1, column=0)
            for j, gj in enumerate(names):
                if j <= i:
                    ttk.Label(grid, text="-", width=8).grid(row=i+1, column=j+1)
                else:
                    var = tk.BooleanVar(value=False)
                    chk = ttk.Checkbutton(grid, variable=var)
                    chk.grid(row=i+1, column=j+1)
                    self.contrast_vars[(gi, gj)] = var

        # Presets
        presets = ttk.Frame(self.contrasts_container)
        presets.pack(anchor="w", padx=10, pady=8)
        ttk.Button(presets, text="Select all", command=lambda: self._set_all_contrasts(True)).grid(row=0, column=0, padx=4)
        ttk.Button(presets, text="Clear all", command=lambda: self._set_all_contrasts(False)).grid(row=0, column=1, padx=4)
        if len(names) >= 2:
            ttk.Button(presets, text=f"Select only vs {names[0]}", command=lambda: self._select_vs(names[0])).grid(row=0, column=2, padx=4)

    def _set_all_contrasts(self, value: bool):
        for k in self.contrast_vars:
            self.contrast_vars[k].set(value)

    def _select_vs(self, ref_group: str):
        for (a, b), var in self.contrast_vars.items():
            var.set(a == ref_group or b == ref_group)

    # ---------- Tab 6: Options ----------

    def _build_tab_options(self):
        frm = self.tab_options
        pad = {"padx": 10, "pady": 8}

        # Data type
        ttk.Label(frm, text="Data type & transformation").grid(row=0, column=0, sticky="w", **pad)

        chk_ratio = ttk.Checkbutton(frm, text="Data are ratios (e.g., CldU/IdU)", variable=self.opt_data_are_ratios, command=self._on_ratio_toggle)
        chk_ratio.grid(row=1, column=0, sticky="w", **pad)

        self.chk_log = ttk.Checkbutton(frm, text="Log-transform ratios for effect sizes (recommended)", variable=self.opt_log_transform)
        self.chk_log.grid(row=2, column=0, sticky="w", **pad)

        ttk.Label(frm, text="Log base:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Combobox(frm, textvariable=self.opt_log_base, values=["natural", "log2"], width=10, state="readonly").grid(row=3, column=1, sticky="w", **pad)

        ttk.Separator(frm, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", **pad)

        # Alpha and correction
        ttk.Label(frm, text="Significance & multiplicity").grid(row=5, column=0, sticky="w", **pad)
        ttk.Label(frm, text="Alpha (e.g., 0.05):").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.opt_alpha, width=10).grid(row=6, column=1, sticky="w", **pad)

        ttk.Label(frm, text="Correction method:").grid(row=7, column=0, sticky="w", **pad)
        ttk.Combobox(frm, textvariable=self.opt_correction, values=["None", "Holm", "Bonferroni", "FDR"], width=14, state="readonly").grid(row=7, column=1, sticky="w", **pad)

        ttk.Separator(frm, orient="horizontal").grid(row=8, column=0, columnspan=3, sticky="ew", **pad)

        # Permutations
        ttk.Label(frm, text="Permutation settings (stratified Wilcoxon)").grid(row=9, column=0, sticky="w", **pad)
        ttk.Label(frm, text="Number of permutations:").grid(row=10, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.opt_n_perm, width=12).grid(row=10, column=1, sticky="w", **pad)
        ttk.Label(frm, text="Random seed:").grid(row=11, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.opt_seed, width=12).grid(row=11, column=1, sticky="w", **pad)

        # (Optional) Gatekeeping note
        ttk.Separator(frm, orient="horizontal").grid(row=12, column=0, columnspan=3, sticky="ew", **pad)
        chk_omni = ttk.Checkbutton(frm, text="Gatekeeping: require a per-replicate omnibus signal before pairwise (optional)", variable=self.opt_omnibus)
        chk_omni.grid(row=13, column=0, columnspan=2, sticky="w", **pad)
        ttk.Label(frm, text="(Omnibus here = checks that within most replicates there is some condition signal; pairwise then use van Elteren.)").grid(row=14, column=0, columnspan=3, sticky="w", **pad)

        self._on_ratio_toggle()

    def _on_ratio_toggle(self):
        if self.opt_data_are_ratios.get():
            if not hasattr(self, "_ratio_initialized") or not self._ratio_initialized:
                self.opt_log_transform.set(True)
                self._ratio_initialized = True
            self.chk_log.state(["!disabled"])
        else:
            self.opt_log_transform.set(False)
            self.chk_log.state(["disabled"])

    # ---------- Tab 7: Review & Run ----------

    def _build_tab_review(self):
        frm = self.tab_review
        pad = {"padx": 10, "pady": 8}

        ttk.Label(frm, text="Review your design and options. Click 'Refresh summary', then 'Run Analysis & Export CSV'.").grid(row=0, column=0, sticky="w", **pad)
        ttk.Button(frm, text="Refresh summary", command=self._refresh_summary).grid(row=0, column=1, sticky="w", **pad)

        self.summary_text = tk.Text(frm, height=26, width=125)
        self.summary_text.grid(row=1, column=0, columnspan=2, sticky="w", **pad)

        ttk.Button(frm, text="Run Analysis & Export CSV", command=self.on_run_analysis).grid(row=2, column=0, sticky="w", **pad)

    def _refresh_summary(self):
        s = self._make_run_log_text(preview=True)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, s)

    # ---------- Review/Save helpers ----------

    def _collect_planned_contrasts(self) -> List[Tuple[str, str]]:
        pairs = []
        for (a, b), var in self.contrast_vars.items():
            if var.get():
                pairs.append((a, b))
        return pairs

    def _counts_by_cell(self, col_to_group: dict, col_to_repl: dict) -> Dict[str, Dict[str, int]]:
        counts = {rep: {g: 0 for g in self.group_names_current} for rep in self.repl_names_current}
        for col in self.df.columns:
            rep = col_to_repl[col]
            grp = col_to_group[col]
            # count non-null numeric cells
            n = pd.to_numeric(self.df[col], errors="coerce").notna().sum()
            counts[rep][grp] += int(n)
        return counts

    def _make_config_dict(self) -> dict:
        # Column→group/replicate mapping
        col_map_g = {col: var.get() for col, var in self.column_group_vars.items()}
        col_map_r = {col: var.get() for col, var in self.column_repl_vars.items()}

        # Options
        options = AnalysisOptions(
            data_are_ratios=self.opt_data_are_ratios.get(),
            log_transform=self.opt_log_transform.get(),
            alpha=float(self.opt_alpha.get()),
            correction_method=self.opt_correction.get(),
            omnibus=self.opt_omnibus.get(),
            log_base=self.opt_log_base.get(),
            n_permutations=int(self.opt_n_perm.get()),
            random_seed=int(self.opt_seed.get())
        )

        design = ExperimentDesign(
            input_file=self.input_file,
            columns=self.df.columns.tolist(),
            groups=self.group_names_current,
            replicates=self.repl_names_current,
            column_to_group=col_map_g,
            column_to_replicate=col_map_r,
            planned_contrasts=self._collect_planned_contrasts(),
            counts_by_cell=self._counts_by_cell(col_map_g, col_map_r)
        )

        return {
            "design": asdict(design),
            "options": asdict(options),
            "version": "0.2.0-van-elteren"
        }

    def _make_run_log_text(self, preview: bool) -> str:
        cfg = self._make_config_dict()
        d = cfg["design"]
        o = cfg["options"]

        lines = []
        lines.append("=== Fiber Assay Analysis — Run Log ===")
        if preview:
            lines.append("(Preview)")
        lines.append("")
        lines.append(f"Input file: {d['input_file']}")
        lines.append(f"Detected columns ({len(d['columns'])}): {', '.join(d['columns'])}")
        lines.append("")
        lines.append(f"Groups ({len(d['groups'])}): {', '.join(d['groups'])}")
        lines.append(f"Replicates ({len(d['replicates'])}): {', '.join(d['replicates'])}")
        lines.append("")
        lines.append("Column mapping:")
        for col in d["columns"]:
            g = d["column_to_group"][col]
            r = d["column_to_replicate"][col]
            lines.append(f"  - {col}: group={g}, replicate={r}")
        lines.append("")
        lines.append("Counts by (replicate, group):")
        for rep in d["counts_by_cell"]:
            sub = d["counts_by_cell"][rep]
            pretty = ", ".join([f"{g}={sub[g]}" for g in d["groups"]])
            lines.append(f"  - {rep}: {pretty}")
        lines.append("")
        lines.append("Planned contrasts:")
        if d["planned_contrasts"]:
            for a, b in d["planned_contrasts"]:
                lines.append(f"  - {a} vs {b}")
        else:
            lines.append("  (none)")
        lines.append("")
        lines.append("Options:")
        lines.append(f"  - Data are ratios: {'Yes' if o['data_are_ratios'] else 'No'}")
        if o["data_are_ratios"]:
            lines.append(f"  - Log-transform ratios for effect sizes: {'Yes' if o['log_transform'] else 'No'} (base={o['log_base']})")
        lines.append(f"  - Alpha: {o['alpha']}")
        lines.append(f"  - Multiple-comparison correction: {o['correction_method']}")
        lines.append(f"  - Permutations: n={o['n_permutations']}, seed={o['random_seed']}")
        lines.append(f"  - Gatekeeping omnibus: {'Yes' if o['omnibus'] else 'No'}")
        lines.append("")
        lines.append("Method to be used:")
        lines.append("  - Stratified Wilcoxon (van Elteren–style) with within-replicate permutations.")
        lines.append("  - Effect sizes: per-group median-of-medians on the RAW scale and the fold change B/A; also report log2 fold change and % change.")

        lines.append("")
        return "\n".join(lines)

    def _reset_all_design(self):
        self.group_names_current = []
        self.group_count_var.set(2)
        self._refresh_group_names()

        self.repl_names_current = []
        self.repl_count_var.set(3)
        self._refresh_repl_names()

        self._rebuild_mapping_panel()
        self._rebuild_contrasts_panel()

    # ---------- RUN ANALYSIS ----------

    def on_run_analysis(self):
        # Validate
        if self.df is None or self.input_file is None:
            messagebox.showerror("Analysis", "No input file loaded.")
            return
        if not self.group_names_current:
            messagebox.showerror("Analysis", "Groups not defined/confirmed.")
            return
        if not self.repl_names_current:
            messagebox.showerror("Analysis", "Replicates not defined/confirmed.")
            return
        if not self.column_group_vars or not self.column_repl_vars:
            messagebox.showerror("Analysis", "Columns not mapped to (group, replicate).")
            return
        planned = self._collect_planned_contrasts()
        if not planned:
            messagebox.showerror("Analysis", "Select at least one planned contrast (Tab 5).")
            return

        # Options
        try:
            alpha = float(self.opt_alpha.get())
            n_perm = int(self.opt_n_perm.get())
            seed = int(self.opt_seed.get())
        except Exception:
            messagebox.showerror("Analysis", "Alpha, permutations, and seed must be numeric.")
            return

        # Build mappings
        col_map_g = {col: var.get() for col, var in self.column_group_vars.items()}
        col_map_r = {col: var.get() for col, var in self.column_repl_vars.items()}

        # Build long data
        try:
            df_long = _reshape_long(self.df, col_map_g, col_map_r)
        except Exception as e:
            messagebox.showerror("Analysis", f"Reshape error: {e}")
            return

        # Prepare two versions:
        # - values for RANK tests = raw (no transform; ranks invariant)
        # - values for EFFECT SIZE summaries = maybe log if requested and ratios
        df_rank = df_long.copy()
        df_eff = df_long.copy()

        try:
            do_log = self.opt_data_are_ratios.get() and self.opt_log_transform.get()
            base = self.opt_log_base.get()
            df_eff["value"] = _maybe_log(df_eff["value"].to_numpy(), use_log=do_log, base=base)
        except Exception as e:
            messagebox.showerror("Analysis", f"Log-transform error: {e}")
            return

        # Optional "gatekeeping omnibus":
        # Here we implement a light-weight check: for each replicate,
        # compute Kruskal–Wallis across groups; then combine replicate p-values
        # via Fisher's method (simple heuristic). If not significant, we stop.
        gatekeep_row = None
        proceed_pairs = True
        if self.opt_omnibus.get():
            try:
                from scipy.stats import kruskal, chi2
                pvals_rep = []
                for _, df_r in df_rank.groupby("replicate_id", sort=False):
                    arrays = [df_r.loc[df_r["group"] == g, "value"].to_numpy() for g in self.group_names_current]
                    arrays = [a for a in arrays if len(a) > 0]
                    if len(arrays) >= 2:
                        stat, p = kruskal(*arrays)
                        pvals_rep.append(p)
                if pvals_rep:
                    # Fisher combine
                    X = -2.0 * np.sum(np.log(pvals_rep))
                    df_chi = 2 * len(pvals_rep)
                    from scipy.stats import chi2 as chi2dist
                    p_fisher = 1 - chi2dist.cdf(X, df_chi)
                else:
                    p_fisher = 1.0
                # Preferred: ART-like omnibus on aligned ranks (controls for replicate)
                try:
                    omni = omnibus_stratified_rank_anova(df_rank.rename(columns={"group":"Group","replicate":"Replicate","value":"Value"}))
                    gatekeep_row = {"Contrast": f"Omnibus (gatekeeping; {omni['method']})",
                                    "p_value_raw": float(omni['p_value'])}
                    proceed_pairs = (omni['p_value'] < alpha) if np.isfinite(omni['p_value']) else True
                except Exception:
                    # Fallback to previous Fisher-over-KW approach
                    from math import log
                    pvals = []
                    for repl, sub in df_rank.groupby('replicate'):
                        arrays = [g['value'].to_numpy() for _, g in sub.groupby('group') if len(g)>0]
                        if len(arrays) >= 2:
                            try:
                                from scipy.stats import kruskal
                                H, p = kruskal(*arrays)
                                pvals.append(float(p))
                            except Exception:
                                pass
                    if pvals:
                        X = -2.0 * sum(log(p) for p in pvals)
                        from scipy.stats import chi2 as chi2dist
                        p_fisher = 1 - chi2dist.cdf(X, 2*len(pvals))
                    else:
                        p_fisher = 1.0
                    proceed_pairs = (p_fisher < alpha)
                    gatekeep_row = {"Contrast": "Omnibus (gatekeeping; Fisher over per-replicate KW)",
                                    "p_value_raw": float(p_fisher)}

            except Exception:
                gatekeep_row = {"Contrast": "Omnibus (gatekeeping)", "p_value_raw": np.nan}
                proceed_pairs = True  # fail open

        # Run planned contrasts with stratified permutations
        rows = []
        raw_ps = []
        row_idx = []

        
        # Pre-compute per-group effect summaries on the RAW scale (for fold-change)

        mom_by_group = {g: _median_of_medians(df_rank, g) for g in self.group_names_current}

        for (A, B) in planned:
            # Rank-based p-value from df_rank
            U_obs, pval, per_repl = van_elteren_permutation_pvalue(
                df_long=df_rank[df_rank["group"].isin([A, B])],
                groupA=A,
                groupB=B,
                n_permutations=n_perm,
                random_state=seed,
                alternative="two-sided"
            )
            raw_ps.append(pval)
            row_idx.append(len(rows))
            # Effect sizes:
            mom_A = mom_by_group.get(A, np.nan)
            mom_B = mom_by_group.get(B, np.nan)
            # Effect sizes computed on the RAW scale:
            # Fold change = (median-of-medians B) / (median-of-medians A)
            if np.isfinite(mom_A) and np.isfinite(mom_B) and mom_A > 0:
                fold_change = float(mom_B / mom_A)
                log2_fc = float(np.log2(fold_change)) if fold_change > 0 else np.nan
                pct_change = float((fold_change - 1.0) * 100.0)
            else:
                fold_change = np.nan
                log2_fc = np.nan
                pct_change = np.nan


            cle = _common_language_effect(df_rank[df_rank["group"].isin([A, B])], A, B)

            rows.append({
                "Contrast": f"{A} vs {B}",
                "Method": "Stratified Wilcoxon (van Elteren via within-replicate permutations)",
                "U_sum": float(U_obs),
                "p_value_raw": float(pval),

                # Effect size details (RAW scale)
                "Median_of_medians_A": float(mom_A),
                "Median_of_medians_B": float(mom_B),
                "Fold_change_B_over_A": fold_change,
                "Log2_FC_B_over_A": log2_fc,
                "Percent_change_B_vs_A": pct_change,

                # Rank-based intuitive size
                "Common_language_P(A>B)": float(cle),

                # Permutation bookkeeping
                "N_permutations": n_perm,
                "Seed": seed
            })


        # Apply multiplicity across planned contrasts
        method = self.opt_correction.get()
        if len(raw_ps) >= 2 and method.lower() in ("holm", "bonferroni", "fdr", "bh"):
            p_adj = _apply_correction(raw_ps, method, alpha)
            for i, ridx in enumerate(row_idx):
                rows[ridx]["p_value_adj"] = float(p_adj[i])
                rows[ridx]["Correction"] = method
        else:
            for ridx in row_idx:
                rows[ridx]["p_value_adj"] = rows[ridx]["p_value_raw"]
                rows[ridx]["Correction"] = "None"

        # Respect optional gatekeeping: if omnibus fails, still report it and flag pairwise as gated
        out_rows = []
        if gatekeep_row is not None:
            out_rows.append({
                "Contrast": gatekeep_row["Contrast"],
                "Method": "Fisher over per-replicate Kruskal–Wallis p-values",
                "U_sum": np.nan,
                "p_value_raw": gatekeep_row["p_value_raw"],
                "p_value_adj": np.nan,
                "Effect_summary": "",
                "Effect_value": np.nan,
                "Common_language_P(A>B)": np.nan,
                "N_permutations": np.nan,
                "Seed": np.nan,
                "Correction": "None"
            })
        if (gatekeep_row is None) or proceed_pairs:
            out_rows.extend(rows)
        else:
            # Gatekeeping failed: add rows but mark as "not performed"
            for r in rows:
                r2 = r.copy()
                r2["p_value_raw"] = np.nan
                r2["p_value_adj"] = np.nan
                r2["Method"] += " (skipped due to gatekeeping)"
                out_rows.append(r2)

        results_df = pd.DataFrame(out_rows)

        # Export
        base = os.path.splitext(self.input_file)[0]
        config_path = f"{base}_design_config.json"
        log_path = f"{base}_run_log.txt"
        out_path = f"{base}_stratified_wilcoxon_results.csv"

        write_config_json(config_path, self._make_config_dict())
        write_run_log_txt(log_path, self._make_run_log_text(preview=False))
        results_df.to_csv(out_path, index=False)

        messagebox.showinfo(
            "Analysis complete",
            f"Wrote:\n- {config_path}\n- {log_path}\n- {out_path}"
        )


if __name__ == "__main__":
    # --- Pre-clean any leftover Tk root (helps on re-runs in Spyder) ---
    try:
        import tkinter as tk
        if tk._default_root is not None:
            try:
                tk._default_root.destroy()
            except Exception:
                pass
    except Exception:
        pass

    # --- Create and run the app ---
    app = FiberAssayGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    try:
        app.mainloop()
    except KeyboardInterrupt:
        app.on_close()

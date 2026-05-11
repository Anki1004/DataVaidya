import streamlit as st
import pandas as pd

from utils.constants import APP_NAME, IQR_MULTIPLIER
from ui.theme import inject_css, PLOTLY_CONFIG
from ui.components import (
    gradient_header,
    metric_card,
    info_pill,
    empty_state,
    before_after_metrics,
)
from core.cleaning import OPS, apply_op, take_snapshot, undo, get_undo_depth
from core.profiling import compute_health_score, memory_footprint

st.set_page_config(page_title=f"Clean · {APP_NAME}", page_icon="🧹", layout="wide")

from app import init_session_state

init_session_state()
inject_css()

gradient_header("🧹 Clean", level=1)
st.caption("Toggle cleaning operations, preview the diff, then apply or undo.")

df = st.session_state.get("df")
if df is None:
    empty_state(
        "No data loaded",
        "Upload a dataset on the Profile page first.",
        cta="Go to Profile",
        cta_page="pages/1_📊_Profile.py",
    )
    st.stop()

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Operations")
    mode = st.radio(
        "Mode",
        ["Preview", "Apply"],
        horizontal=True,
        key="clean_mode",
        help="Preview shows the diff without committing; Apply commits and pushes to undo stack.",
    )

    with st.expander("Missing values", expanded=True):
        do_fill = st.checkbox("Fill missing", key="op_fill_enabled")
        if do_fill:
            strategy = st.selectbox(
                "Strategy",
                ["median", "mean", "mode", "constant", "ffill", "bfill"],
                key="op_fill_strategy",
            )
            fill_value = None
            if strategy == "constant":
                fill_value = st.text_input("Constant value", key="op_fill_value")
            fill_cols = st.multiselect(
                "Columns (empty = all)",
                options=df.columns.tolist(),
                key="op_fill_cols",
            )

    with st.expander("Duplicates", expanded=False):
        do_dedupe = st.checkbox("Drop duplicates", key="op_dedupe_enabled")
        if do_dedupe:
            dedupe_keep = st.selectbox(
                "Keep", ["first", "last", "none"], key="op_dedupe_keep"
            )
            dedupe_subset = st.multiselect(
                "Subset (empty = all)",
                options=df.columns.tolist(),
                key="op_dedupe_subset",
            )

    with st.expander("Outliers", expanded=False):
        do_outliers = st.checkbox("Handle outliers (IQR)", key="op_outliers_enabled")
        if do_outliers:
            outlier_action = st.selectbox(
                "Action", ["Cap", "Remove"], key="op_outliers_action"
            )
            outlier_mult = st.slider(
                "IQR multiplier",
                1.0,
                3.0,
                value=IQR_MULTIPLIER,
                step=0.1,
                key="op_outliers_mult",
            )
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            outlier_cols = st.multiselect(
                "Numeric columns", options=numeric_cols, key="op_outliers_cols"
            )

    with st.expander("Strings", expanded=False):
        do_strip = st.checkbox("Strip whitespace", key="op_strip_enabled")
        do_case = st.checkbox("Standardize case", key="op_case_enabled")
        if do_case:
            case_type = st.selectbox(
                "Case", ["lower", "upper", "title"], key="op_case_type"
            )
            obj_cols = df.select_dtypes(include="object").columns.tolist()
            case_cols = st.multiselect(
                "Columns", options=obj_cols, key="op_case_cols"
            )

    with st.expander("Types & dates", expanded=False):
        do_dates = st.checkbox("Parse dates (auto-detect)", key="op_dates_enabled")
        do_downcast = st.checkbox(
            "Downcast numerics (save memory)", key="op_downcast_enabled"
        )

    st.divider()

    run_clicked = st.button(
        "Run pipeline", type="primary", use_container_width=True, key="clean_run"
    )

    undo_col1, undo_col2 = st.columns(2)
    with undo_col1:
        if st.button(
            "Undo last",
            use_container_width=True,
            disabled=get_undo_depth(st.session_state) == 0,
            key="clean_undo",
        ):
            restored = undo(st.session_state)
            if restored is not None:
                st.session_state["df"] = restored
                st.toast("Reverted to last snapshot", icon="↩️")
                st.rerun()
    with undo_col2:
        if st.button(
            "Reset to original",
            use_container_width=True,
            key="clean_reset",
            disabled=st.session_state.get("df_raw") is None,
        ):
            if st.session_state.get("df_raw") is not None:
                st.session_state["df"] = st.session_state["df_raw"].copy()
                st.session_state["cleaning_log"] = []
                st.session_state["undo_stack"] = []
                st.toast("Reset to original", icon="🔄")
                st.rerun()

    st.divider()

    # Snapshots
    st.subheader("Snapshots")
    snap_name = st.text_input("Snapshot name", key="snap_name", max_chars=30)
    if st.button(
        "Save snapshot",
        use_container_width=True,
        disabled=not snap_name,
        key="snap_save",
    ):
        st.session_state.setdefault("snapshots", {})[snap_name] = st.session_state[
            "df"
        ].copy()
        st.toast(f"Saved snapshot '{snap_name}'", icon="💾")

    for name in list(st.session_state.get("snapshots", {}).keys()):
        cols_snap = st.columns([3, 1])
        cols_snap[0].caption(f"📌 {name}")
        if cols_snap[1].button("↺", key=f"restore_{name}", help=f"Restore {name}"):
            take_snapshot(st.session_state, st.session_state["df"], "before_restore")
            st.session_state["df"] = st.session_state["snapshots"][name].copy()
            st.rerun()

with right:
    st.subheader("Result preview")

    if run_clicked:
        ops_to_run: list[tuple[str, dict]] = []
        if st.session_state.get("op_fill_enabled"):
            ops_to_run.append(
                (
                    "fill_missing",
                    {
                        "columns": st.session_state.get("op_fill_cols") or None,
                        "strategy": st.session_state.get("op_fill_strategy", "median"),
                        "value": st.session_state.get("op_fill_value"),
                    },
                )
            )
        if st.session_state.get("op_dedupe_enabled"):
            ops_to_run.append(
                (
                    "drop_duplicates",
                    {
                        "subset": st.session_state.get("op_dedupe_subset") or None,
                        "keep": st.session_state.get("op_dedupe_keep", "first"),
                    },
                )
            )
        if st.session_state.get("op_outliers_enabled"):
            action = st.session_state.get("op_outliers_action", "Cap")
            op = "cap_outliers_iqr" if action == "Cap" else "remove_outliers_iqr"
            ops_to_run.append(
                (
                    op,
                    {
                        "columns": st.session_state.get("op_outliers_cols", []),
                        "multiplier": st.session_state.get(
                            "op_outliers_mult", IQR_MULTIPLIER
                        ),
                    },
                )
            )
        if st.session_state.get("op_strip_enabled"):
            ops_to_run.append(("strip_whitespace", {"columns": None}))
        if st.session_state.get("op_case_enabled"):
            ops_to_run.append(
                (
                    "standardize_case",
                    {
                        "columns": st.session_state.get("op_case_cols", []),
                        "case": st.session_state.get("op_case_type", "lower"),
                    },
                )
            )
        if st.session_state.get("op_dates_enabled"):
            obj_cols = df.select_dtypes(include="object").columns.tolist()
            ops_to_run.append(
                (
                    "parse_dates",
                    {"columns": obj_cols, "format": "infer", "errors": "coerce"},
                )
            )
        if st.session_state.get("op_downcast_enabled"):
            ops_to_run.append(("downcast_numeric", {}))

        if not ops_to_run:
            st.warning("Toggle at least one operation.")
        else:
            # Run all ops sequentially
            apply_mode = "preview" if mode == "Preview" else "apply"
            working = df.copy()
            change_logs: list[dict] = []

            if apply_mode == "apply":
                take_snapshot(
                    st.session_state, df, label=f"before_{len(ops_to_run)}_ops"
                )

            for op_name, kwargs in ops_to_run:
                try:
                    working, log = apply_op(
                        working, op_name, mode=apply_mode, **kwargs
                    )
                    change_logs.append(log)
                    if apply_mode == "apply":
                        st.session_state["cleaning_log"].append((op_name, kwargs))
                except Exception as e:
                    st.error(f"Operation '{op_name}' failed: {e}")
                    break

            if apply_mode == "apply":
                st.session_state["df"] = working
                st.toast(f"Applied {len(change_logs)} operation(s)", icon="✅")

            # Show before/after metrics
            before = {
                "Rows": len(df),
                "Cells": int(df.size),
                "Memory MB": round(
                    df.memory_usage(deep=True).sum() / 1024 / 1024, 2
                ),
                "Health Score": st.session_state.get("health_score")
                or compute_health_score(df)["score"],
            }
            after = {
                "Rows": len(working),
                "Cells": int(working.size),
                "Memory MB": round(
                    working.memory_usage(deep=True).sum() / 1024 / 1024, 2
                ),
                "Health Score": compute_health_score(working)["score"],
            }
            before_after_metrics(before, after)

            # Change log
            with st.expander("Change log details", expanded=False):
                for log in change_logs:
                    st.json(log)

            # Preview table
            st.subheader(
                f"{'Preview' if apply_mode == 'preview' else 'Updated'} data"
            )
            st.dataframe(
                working.head(50), use_container_width=True, hide_index=True
            )
    else:
        st.info("Configure operations on the left, then click **Run pipeline**.")
        st.subheader("Current data")
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)

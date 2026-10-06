"""Predictive Maintenance Intelligence — NASA C-MAPSS FD004

Public-facing Streamlit portfolio demo. This is a READ-ONLY presentation layer
over the frozen research artifacts (exposed through the small derived bundle in
`demo/`). It does not retrain, tune, or recompute any scientific result.

Run locally:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import app_data as ad

st.set_page_config(
    page_title="Predictive Maintenance Intelligence — FD004",
    page_icon="🛩️",
    layout="wide",
)


# --------------------------------------------------------------------------
# Cached loaders (all read committed demo-bundle artifacts only)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def ov() -> dict:
    return ad.load_overview()


@st.cache_data(show_spinner=False)
def met() -> dict:
    return ad.load_metrics()


@st.cache_data(show_spinner=False)
def rul_preds() -> pd.DataFrame:
    return ad.load_rul_predictions()


@st.cache_data(show_spinner=False)
def fr_preds() -> pd.DataFrame:
    return ad.load_fr_predictions()


@st.cache_data(show_spinner=False)
def decision_csv(name: str) -> pd.DataFrame:
    return ad.load_decision_csv(name)


@st.cache_data(show_spinner=False)
def decision_json(name: str) -> dict:
    return ad.load_decision_json(name)


@st.cache_data(show_spinner=False)
def rul_preds_model(model: str) -> pd.DataFrame:
    return ad.load_rul_predictions(model)


@st.cache_data(show_spinner=False)
def fr_preds_model(model: str) -> pd.DataFrame:
    return ad.load_fr_predictions(model)


@st.cache_data(show_spinner=False)
def lab_examples() -> list[dict]:
    return ad.derive_examples(
        ad.load_rul_predictions("temporal"), ad.load_fr_predictions("temporal")
    )


def missing_guard(rel_hint: str) -> None:
    st.warning("This demonstration artifact is unavailable in the public release.")
    st.caption(f"Missing bundle file: `{rel_hint}`")


# --------------------------------------------------------------------------
# Shared chart helpers (display-only, built from frozen numbers)
# --------------------------------------------------------------------------
def rul_line_chart(df: pd.DataFrame) -> alt.Chart:
    long = df.melt(
        id_vars=["cycle"], value_vars=["actual_rul", "predicted_rul"],
        var_name="series", value_name="RUL",
    )
    long["series"] = long["series"].map(
        {"actual_rul": "Actual RUL", "predicted_rul": "Predicted RUL"}
    )
    return (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("cycle:Q", title="Operating cycle"),
            y=alt.Y("RUL:Q", title="Remaining useful life (cycles)"),
            color=alt.Color("series:N", scale=alt.Scale(scheme="tableau10"), legend=alt.Legend(title=None)),
            tooltip=["cycle", "series", "RUL"],
        )
        .properties(height=340, title="Actual vs predicted RUL over the engine's life")
    )


def prob_chart(df: pd.DataFrame, threshold: float) -> alt.Chart:
    base = alt.Chart(df).mark_line(color="#1f77b4").encode(
        x=alt.X("cycle:Q", title="Operating cycle"),
        y=alt.Y("failure_prob:Q", title="Failure-risk probability", scale=alt.Scale(domain=[0, 1])),
        tooltip=["cycle", "failure_prob", "actual_state"],
    )
    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color="#d62728", strokeDash=[6, 4]).encode(
        y="t:Q"
    )
    return (base + rule).properties(
        height=340, title=f"Failure-risk probability (decision threshold = {threshold:.2f})"
    )


def prob_chart_lab(fr_engine: pd.DataFrame, threshold: float, cycle: int) -> alt.Chart:
    """Probability trajectory + threshold rule + the selected operating point,
    coloured by the predicted state at the current threshold (display-only)."""
    d = fr_engine.assign(predicted_state=ad.predicted_state_from_threshold(fr_engine, threshold))
    base = (
        alt.Chart(d)
        .mark_line(color="#1f77b4")
        .encode(
            x=alt.X("cycle:Q", title="Operating cycle"),
            y=alt.Y("failure_prob:Q", title="Failure-risk probability", scale=alt.Scale(domain=[0, 1])),
            tooltip=["cycle", "failure_prob", "actual_state", "predicted_state"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"t": [threshold]}))
        .mark_rule(color="#d62728", strokeDash=[6, 4])
        .encode(y="t:Q")
    )
    pt = d[d["cycle"] == int(cycle)]
    marker = (
        alt.Chart(pt)
        .mark_point(size=140, filled=True, color="#ff7f0e", stroke="black")
        .encode(x="cycle:Q", y="failure_prob:Q", tooltip=["cycle", "failure_prob"])
    )
    return (base + rule + marker).properties(
        height=340, title=f"Failure-risk probability (threshold = {threshold:.2f})"
    )


def confusion_chart(cm: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(cm)
        .mark_rect()
        .encode(
            x=alt.X("predicted:N", title="Predicted", scale=alt.Scale(domain=["No risk", "At risk"])),
            y=alt.Y("actual:N", title="Actual", scale=alt.Scale(domain=["No risk", "At risk"])),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None),
            text=alt.Text("count:Q"),
        )
        .properties(height=300, title="Confusion matrix (validation engines)")
    )


def cost_curve_chart(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("cost_ratio:Q", title="Missed-failure : false-alarm cost ratio (log)", scale=alt.Scale(type="log")),
            y=alt.Y("best_expected_cost:Q", title="Best expected cost (unitless)"),
            color=alt.Color("label:N", legend=alt.Legend(title="Model")),
            tooltip=["model_id", "label", "cost_ratio", "best_threshold", "best_expected_cost"],
        )
        .properties(height=340, title="Cost-ratio sensitivity (unitless scenario cost)")
    )


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_overview() -> None:
    st.title("Predictive Maintenance Intelligence")
    st.subheader("NASA C-MAPSS FD004")
    st.markdown(
        "> Predict **Remaining Useful Life (RUL)** and identify **near-term failure "
        "risk** from multivariate turbofan degradation data using leakage-safe "
        "validation and causal temporal features."
    )

    d = ov()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training engines", d["n_train_engines_total"])
    c2.metric("Official test engines", d["n_test_engines_official"])
    c3.metric("Operating regimes", d["n_operating_regimes"])
    c4.metric("Fault modes", d["n_fault_modes"])

    c5, c6, c7 = st.columns(3)
    c5.metric("Held-out validation engines", d["n_val_engines"])
    c6.metric("Primary risk horizon", f"H{d['primary_horizon_H']}")
    c7.metric("Automated tests", d["scientific_test_count"])

    st.divider()
    st.markdown(
        "**Interactive demonstration powered by frozen validation artifacts.**\n\n"
        "Select validation engines, inspect RUL predictions, adjust failure-risk "
        "thresholds, and explore model decisions. The app does **not** retrain models "
        "or access the official test partition."
    )
    st.markdown(
        "**Research implementation complete and version-frozen.** This dashboard is a "
        "read-only demonstration of validated research artifacts. It reports "
        "**engine-grouped validation** metrics, not official test-set metrics."
    )
    st.caption(ad.PROVENANCE)


def page_prediction_lab() -> None:
    st.title("Interactive Prediction Lab")
    st.caption(
        "Interactive **validation prediction** demonstration — not live ML inference, "
        "not real-time prediction, and not production prediction."
    )
    st.markdown(
        "The interface lets you explore **frozen predictions from held-out validation "
        "engines** and see how threshold decisions change. No model is retrained, and "
        "the official test partition is never accessed."
    )

    model = st.selectbox(
        "Prediction model view", list(ad.MODEL_KEYS),
        format_func=lambda k: ad.MODEL_LABELS[k], key="lab_model",
    )
    try:
        rp = rul_preds_model(model)
        fp = fr_preds_model(model)
    except ad.MissingArtifact as e:
        missing_guard(str(e))
        return

    ids = ad.validation_engine_ids(rp)
    unit = st.selectbox(
        "Validation engine", ids, key="lab_unit", format_func=lambda x: f"Engine {x}"
    )

    re_ = ad.engine_slice_predictions(rp, unit)
    fe_ = ad.engine_slice_predictions(fp, unit)
    cycles = ad.cycle_options(re_)
    if st.session_state.get("lab_cycle") not in cycles:
        st.session_state.lab_cycle = cycles[len(cycles) // 2]
    cycle = st.selectbox("Cycle", cycles, key="lab_cycle")

    threshold = st.slider(
        "Failure-risk threshold", 0.10, 0.90, 0.50, 0.01, key="lab_thr",
        help="Moves the decision boundary on the stored probability (display-only).",
    )

    def _apply_example(u: int, c: int) -> None:
        st.session_state.lab_unit = int(u)
        st.session_state.lab_cycle = int(c)

    st.markdown("**Try these examples**")
    examples = lab_examples()
    cols = st.columns(max(len(examples), 1))
    for col, ex in zip(cols, examples):
        with col:
            st.button(
                f"{ex['label']}\nEngine {ex['unit_id']} · cyc {ex['cycle']}",
                key=f"ex_{ex['label']}",
                use_container_width=True,
                on_click=_apply_example, args=(ex["unit_id"], ex["cycle"]),
            )

    # ---- Live RUL result ------------------------------------------------
    st.subheader("RUL — frozen validation prediction")
    try:
        pt = ad.rul_point(re_, cycle)
        rpt = ad.fr_point(fe_, cycle, threshold)
    except ad.MissingArtifact as e:
        missing_guard(str(e))
        return

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Actual RUL", round(pt["actual_rul"], 1))
    a2.metric("Predicted RUL", round(pt["predicted_rul"], 1))
    a3.metric("Absolute error", round(pt["abs_error"], 1))
    a4.metric("Signed error", round(pt["signed_error"], 1))
    st.caption("Frozen validation prediction — **not** an official test-set prediction.")
    st.altair_chart(rul_line_chart(re_), use_container_width=True)

    # ---- Live failure-risk result --------------------------------------
    st.subheader("Failure risk — frozen validation prediction")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Failure probability", f"{rpt['failure_prob']:.3f}")
    r2.metric("Selected threshold", f"{threshold:.2f}")
    r3.metric("Predicted state", "At risk" if rpt["predicted_state"] else "No risk")
    r4.metric("Actual state (H30)", "At risk" if rpt["actual_state"] else "No risk")
    r5.metric("Classification", "Correct" if rpt["correct"] else "Incorrect")
    st.altair_chart(prob_chart_lab(fe_, threshold, cycle), use_container_width=True)

    # ---- What does this mean? ------------------------------------------
    prob = rpt["failure_prob"]
    if rpt["predicted_state"]:
        head = (
            f"At cycle {cycle}, the {ad.MODEL_LABELS[model]} model estimates about "
            f"{prob*100:.0f}% failure risk. With the current {threshold:.0%} threshold, "
            f"the system **raises a failure-risk alert**."
        )
        hint = f"Raising the threshold above {prob:.2f} would clear this alert."
    else:
        head = (
            f"At cycle {cycle}, the {ad.MODEL_LABELS[model]} model estimates about "
            f"{prob*100:.0f}% failure risk. With the current {threshold:.0%} threshold, "
            f"the system **does not raise an alert**."
        )
        hint = f"Lowering the threshold below {prob:.2f} would raise an alert."
    st.info(
        head + "\n\n" + hint
        + "\n\nThis is a **benchmark validation engine**, not a real aircraft or "
        "deployed industrial system."
    )

    # ---- Threshold interaction (full validation set) --------------------
    st.subheader("Threshold decision (full validation set)")
    st.caption(
        "Deterministic re-computation from the frozen probabilities and actual labels "
        "at the chosen threshold. When the threshold equals the frozen selection these "
        "values match the authoritative metrics. This is an interactive demonstration, "
        "**not** a new model-selection exercise."
    )
    tm = ad.threshold_metrics(fp, threshold)
    t1, t2, t3, t4, t5, t6, t7 = st.columns(7)
    t1.metric("Predicted positive", tm["predicted_positive"])
    t2.metric("TP", tm["TP"])
    t3.metric("FP", tm["FP"])
    t4.metric("FN", tm["FN"])
    t5.metric("Precision", f"{tm['precision']:.3f}")
    t6.metric("Recall", f"{tm['recall']:.3f}")
    t7.metric("F1", f"{tm['f1']:.3f}")
    cm = pd.DataFrame(
        {
            "predicted": ["No risk", "At risk", "No risk", "At risk"],
            "actual": ["No risk", "No risk", "At risk", "At risk"],
            "count": [tm["TN"], tm["FP"], tm["FN"], tm["TP"]],
        }
    )
    st.altair_chart(confusion_chart(cm), use_container_width=True)

    frozen = met()["failure_risk"].get(model)
    if frozen:
        st.caption(
            f"Frozen reference at the selected model's authoritative threshold "
            f"({frozen['threshold']:.3f}): precision {frozen['precision']:.3f}, "
            f"recall {frozen['recall']:.3f}, F1 {frozen['F1']:.3f}."
        )

    # ---- Engine-level summary ------------------------------------------
    st.subheader("Engine-level summary")
    summ = ad.rul_engine_summary(re_)
    last = re_.iloc[-1]
    alerts = ad.engine_alert_count(fe_, threshold)
    r1c, r2c, r3c, r4c, r5c = st.columns(5)
    r1c.metric("Selected engine", unit)
    r2c.metric("Cycles observed", summ["n_rows"])
    r3c.metric("Actual final RUL", round(float(last["actual_rul"]), 1))
    r4c.metric("Predicted final RUL", round(float(last["predicted_rul"]), 1))
    r5c.metric("Risk alerts (current threshold)", alerts)

    model_id = {
        "temporal": "FR25_T4_combined_cyc",
        "anchor": "FR25_T0_anchor_cycle",
    }[model]
    try:
        eng = decision_csv("engine_level_analysis.csv")
        row = eng[(eng["model_id"] == model_id) & (eng["unit_id"] == int(unit))]
        if not row.empty:
            st.caption(
                f"Frozen Phase 2.6 (threshold 0.5): {int(row['n_fp_runs'].iloc[0])} "
                f"false-alert runs, "
                f"detected = {bool(row['ever_detected'].iloc[0])}."
            )
    except ad.MissingArtifact:
        pass


def page_performance() -> None:
    st.title("Model Performance")
    st.markdown(
        "Frozen **validation** results comparing the classical anchor (current-row "
        "features + cycle, no temporal transforms) against the causal temporal model "
        "(combined lag/rolling/slope features). Metrics are **not** test-set metrics."
    )

    m = met()
    labels = m.get("labels", {})
    st.subheader("RUL regression")
    r = m["rul"]
    best = r["temporal_best_mae"]
    rul_tbl = pd.DataFrame(
        {
            "Metric": ["MAE ↓", "RMSE ↓", "R² ↑", "Prognostic score (last/engine) ↓"],
            "Classical anchor": [r["anchor"]["MAE"], r["anchor"]["RMSE"], r["anchor"]["R2"], r["anchor"]["prognostic_score_last_per_engine"]],
            "Causal temporal — paired (combined, W=5)": [r["temporal"]["MAE"], r["temporal"]["RMSE"], r["temporal"]["R2"], r["temporal"]["prognostic_score_last_per_engine"]],
            "Causal temporal — best-MAE (combined, W=3)": [best["MAE"], best["RMSE"], best["R2"], best["prognostic_score_last_per_engine"]],
        }
    )
    st.dataframe(rul_tbl, hide_index=True, use_container_width=True)
    st.caption(
        m.get(
            "selection_note",
            "The W=5 temporal RUL is the configuration paired with the selected "
            "failure-risk model; the best-MAE temporal RUL is the W=3 configuration. "
            "Neither is labelled the single 'best RUL'.",
        )
    )

    st.subheader("Failure-risk classification (H30)")
    f = m["failure_risk"]
    fr_tbl = pd.DataFrame(
        {
            "Metric": ["Recall ↑", "Precision ↑", "F1 ↑", "PR-AUC ↑", "ROC-AUC ↑"],
            "Classical anchor": [f["anchor"]["recall"], f["anchor"]["precision"], f["anchor"]["F1"], f["anchor"]["PR_AUC"], f["anchor"]["ROC_AUC"]],
            "Causal temporal (combined)": [f["temporal"]["recall"], f["temporal"]["precision"], f["temporal"]["F1"], f["temporal"]["PR_AUC"], f["temporal"]["ROC_AUC"]],
        }
    )
    st.dataframe(fr_tbl, hide_index=True, use_container_width=True)

    st.caption(
        "The temporal model gives a small, mixed gain: RUL MAE is essentially flat "
        "while failure-risk F1 / PR-AUC rise modestly and false alarms fall. See "
        "“Why no LSTM?” for why this did not justify neural escalation."
    )


def page_engine_explorer() -> None:
    st.title("Engine Explorer")
    st.markdown(
        "Inspect per-cycle **validation-engine** predictions. These are held-out "
        "validation results, **not** official test-set predictions, and no official "
        "test labels are used anywhere in this app."
    )

    try:
        rp = rul_preds()
        fp = fr_preds()
    except ad.MissingArtifact as e:
        missing_guard(str(e))
        return

    ids = ad.validation_engine_ids(rp)
    unit = st.selectbox("Select a validation engine", ids, index=0, format_func=lambda x: f"Engine {x}")

    re_ = ad.engine_slice_predictions(rp, unit)
    fe_ = ad.engine_slice_predictions(fp, unit)

    summ = ad.rul_engine_summary(re_)
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Engine ID", unit)
    i2.metric("Cycle range", f"{summ['cycle_min']}–{summ['cycle_max']}")
    i3.metric("Observations", summ["n_rows"])
    i4.metric("Engine MAE (cycles)", round(summ["mae"], 1))

    st.altair_chart(rul_line_chart(re_), use_container_width=True)

    default_thr = float(met()["failure_risk"]["temporal"]["threshold"])
    thr = st.slider(
        "Decision threshold", 0.0, 1.0, default_thr, 0.01,
        help="Recomputes the predicted risk state from the stored probability (display-only).",
    )
    fe_ = fe_.assign(predicted_state=ad.predicted_state_from_threshold(fe_, thr))
    st.altair_chart(prob_chart(fe_, thr), use_container_width=True)

    n_alert = int(fe_["predicted_state"].sum())
    n_actual = int(fe_["actual_state"].sum())
    a1, a2 = st.columns(2)
    a1.metric("Predicted at-risk cycles", n_alert)
    a2.metric("Actual at-risk cycles (H30)", n_actual)
    st.caption(
        "Showing predicted vs actual risk state at the chosen threshold. The stored "
        "probability is unchanged; only the 0/1 decision boundary moves."
    )


def page_failure_risk() -> None:
    st.title("Failure Risk")
    m = met()
    f = m["failure_risk"]["temporal"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recall", f"{f['recall']:.3f}")
    c2.metric("Precision", f"{f['precision']:.3f}")
    c3.metric("F1", f"{f['F1']:.3f}")
    c4.metric("PR-AUC", f"{f['PR_AUC']:.3f}")
    c5.metric("ROC-AUC", f"{f['ROC_AUC']:.3f}")

    cm = pd.DataFrame(
        {
            "predicted": ["No risk", "At risk", "No risk", "At risk"],
            "actual": ["No risk", "No risk", "At risk", "At risk"],
            "count": [f["TN"], f["FP"], f["FN"], f["TP"]],
        }
    )
    st.altair_chart(confusion_chart(cm), use_container_width=True)

    mb = m["majority_baseline"]
    st.info(
        f"Majority-class accuracy ≈ {mb['accuracy']*100:.1f}% with **recall = 0** "
        f"(only ~{mb['positive_rate']*100:.1f}% of cycles are at-risk under H30). "
        "Raw accuracy is misleading under this class imbalance — a model that never "
        "alerts would still score ~87%. That is why recall / precision / PR-AUC, not "
        "accuracy, are the primary metrics."
    )

    with st.expander("Threshold behaviour"):
        try:
            tc = decision_csv("threshold_cost.csv")
        except ad.MissingArtifact as e:
            missing_guard(str(e))
            return
        sel = tc[(tc["model_id"] == "FR25_T4_combined_cyc") & (tc["cost_ratio"] == 1.0)]
        ch = (
            alt.Chart(sel)
            .mark_line(point=True)
            .encode(
                x=alt.X("threshold:Q", title="Decision threshold"),
                y=alt.Y("recall:Q", title="Recall", axis=alt.Axis(grid=True)),
                color=alt.value("#2ca02c"),
            )
            .properties(height=300, title="Recall vs threshold (illustrative cost ratio 1:1)")
        )
        st.altair_chart(ch, use_container_width=True)
        st.caption("Lower thresholds alert earlier (higher recall, more false alarms).")


def page_decision() -> None:
    st.title("Decision Analysis")
    st.warning(
        "Decision costs are **illustrative scenario assumptions**, not observed "
        "financial costs. Values are unitless and are never converted to currency."
    )

    try:
        sens = decision_csv("cost_ratio_sensitivity.csv")
        comp = decision_csv("decision_cost_comparison.csv")
        engine = decision_csv("engine_level_analysis.csv")
        fatigue = decision_json("alert_fatigue.json")
    except ad.MissingArtifact as e:
        missing_guard(str(e))
        return

    label_map = {"FR25_T0_anchor_cycle": "Classical anchor", "FR25_T4_combined_cyc": "Causal temporal"}
    h30 = sens[sens["horizon"] == 30].copy()
    h30["label"] = h30["model_id"].map(label_map).fillna(h30["model_id"])
    st.altair_chart(cost_curve_chart(h30), use_container_width=True)
    st.caption(
        "Under most cost ratios the temporal model is lower-cost, but the advantage "
        "is **not robust**: at extreme missed-failure ratios (50:1, 100:1) the anchor "
        "wins. This non-robustness is central to the model-selection conclusion."
    )

    st.subheader("False alarms and engine-level detection")
    a, b = st.columns(2)
    a.metric("Anchor total FP rows", fatigue["anchor"]["total_false_positive_rows"])
    a.metric("Temporal total FP rows", fatigue["temporal"]["total_false_positive_rows"])
    b.metric("Anchor mean FP runs / engine", fatigue["anchor"]["mean_fp_runs_per_engine"])
    b.metric("Temporal mean FP runs / engine", fatigue["temporal"]["mean_fp_runs_per_engine"])

    det = (
        engine.groupby("model_id")
        .agg(engines=("unit_id", "count"), ever_detected=("ever_detected", "sum"))
        .reset_index()
    )
    det["model"] = det["model_id"].map(label_map)
    st.dataframe(det[["model", "engines", "ever_detected"]], hide_index=True, use_container_width=True)

    st.subheader("Cost-ratio comparison (from frozen Phase 2.6)")
    st.dataframe(comp, hide_index=True, use_container_width=True)


def page_no_lstm() -> None:
    st.title("Why No LSTM?")
    st.markdown(
        "### Initial concept\n"
        "The original project concept considered an LSTM sequence model — the common "
        "“RUL prediction with an LSTM” portfolio framing."
    )
    st.markdown(
        "### What was actually done\n"
        "Before choosing any neural architecture, the project first established the "
        "things a neural model would have to beat: classical baselines, engine-grouped "
        "validation, train-only per-regime normalization, causal temporal features, "
        "decision-level validation, and bootstrap analysis."
    )
    m = met()
    r = m["rul"]
    f = m["failure_risk"]
    st.markdown("### Evidence")
    st.markdown(
        f"- RUL temporal improvement was **small / mixed**: anchor MAE "
        f"{r['anchor']['MAE']:.2f} → temporal MAE {r['temporal']['MAE']:.2f} cycles, "
        f"with no uniform win across window sizes.\n"
        f"- Failure-risk temporal improvement was **modest**: F1 "
        f"{f['anchor']['F1']:.3f} → {f['temporal']['F1']:.3f}, PR-AUC "
        f"{f['anchor']['PR_AUC']:.3f} → {f['temporal']['PR_AUC']:.3f}.\n"
        f"- False alarms improved (FP {f['anchor']['FP']} → {f['temporal']['FP']}).\n"
        f"- The decision advantage was **not robust across all cost ratios** — the "
        f"anchor was lower-cost under extreme missed-failure ratios."
    )
    st.success(
        "**Final decision:** no neural escalation was justified by the evidence. "
        "This is not a claim that “LSTM is bad” — it is that the current evidence did "
        "not establish a sufficiently strong unresolved problem to justify additional "
        "neural complexity."
    )


def page_reproducibility() -> None:
    st.title("Reproducibility")
    d = ov()
    st.subheader("Environment")
    st.json(d["environment_packages"])
    st.caption(
        f"Authoritative results were produced under scientific Python {d['scientific_python']} "
        f"with a locked dependency set (seed = {d['seed']})."
    )

    st.subheader("Validation")
    st.markdown(
        f"- {d['scientific_test_count']} automated tests\n"
        f"- Phase 1 {d['gates']['phase1']} · Phase 2 {d['gates']['phase2']} · "
        f"Phase 2.5 {d['gates']['phase2_5']} · Phase 2.6 {d['gates']['phase2_6']}"
    )

    st.subheader("Leakage controls")
    for item in [
        "Engine-grouped split (no engine spans train/validation)",
        "Train-only, per-regime normalization",
        "Causal temporal features (only cycle ≤ t, same engine)",
        "Sealed official test partition (never accessed here)",
        "Train-only feature selection",
    ]:
        st.markdown(f"- {item}")

    st.info(
        "Reproducibility caveat: the authoritative scientific results are tied to the "
        "locked environment. They are **reproducible in that pinned environment**, not "
        "guaranteed identical on any machine. This demo app also runs in a **separate, "
        "minimal** environment from the scientific stack — the two are distinct concerns."
    )


def page_limitations() -> None:
    st.title("Limitations")
    st.markdown("This page is deliberately explicit about what this project is **not**.")
    for item in [
        "NASA C-MAPSS is a **simulated** benchmark, not real engine telemetry.",
        "No external industrial validation was performed.",
        "No production deployment exists.",
        "No measured real-world savings — decision costs are illustrative only.",
        "Operational **cycles are not calendar days**; “days-to-failure” framings are unsupported.",
        "Validation results are **not** proof of industrial performance.",
    ]:
        st.markdown(f"- {item}")
    st.caption(ad.PROVENANCE)


def page_methodology() -> None:
    st.title("Evidence / Methodology")
    st.markdown("End-to-end pipeline that produced the frozen release:")
    stages = [
        ("NASA C-MAPSS FD004", "Multivariate turbofan run-to-failure simulation."),
        ("Data Integrity Audit", "Verify schemas, counts, seals before any modelling."),
        ("Engine-Grouped Split", "Whole engines assigned to train/validation to prevent leakage."),
        ("Train-Only Normalization", "Per-regime scalers fit on training engines only."),
        ("Classical Baselines", "Ridge / Logistic / HistGradientBoosting anchors."),
        ("Causal Temporal Features", "Lag, rolling, slope features using only past cycles."),
        ("Failure-Risk Analysis", "Imbalanced detection metrics over raw accuracy."),
        ("Decision / Cost Sensitivity", "Unitless scenario costs across cost ratios."),
        ("Bootstrap Validation", "Engine-level resampling for honest uncertainty."),
        ("Final Model Selection", "Freeze the simplest model the evidence supports."),
        ("Frozen Research Release", "Version-locked, byte-stable artifacts."),
    ]
    for i, (title, desc) in enumerate(stages, 1):
        st.markdown(f"**{i}. {title}** — {desc}")


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
PAGES = {
    "Overview": page_overview,
    "Interactive Prediction Lab": page_prediction_lab,
    "Model Performance": page_performance,
    "Engine Explorer": page_engine_explorer,
    "Failure Risk": page_failure_risk,
    "Decision Analysis": page_decision,
    "Why No LSTM?": page_no_lstm,
    "Reproducibility": page_reproducibility,
    "Limitations": page_limitations,
    "Evidence / Methodology": page_methodology,
}


def main() -> None:
    if not ad.bundle_exists():
        st.error(
            "Demo bundle is missing. Generate it first with "
            "`python scripts/build_demo_bundle.py`."
        )
        st.stop()

    st.sidebar.title("FD004 Portfolio Demo")
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
    st.sidebar.caption("Read-only demo over frozen research artifacts.")
    PAGES[choice]()

    st.sidebar.divider()
    st.sidebar.caption(
        "Research repository is authoritative. This dashboard presents validated "
        "validation artifacts only."
    )


main()

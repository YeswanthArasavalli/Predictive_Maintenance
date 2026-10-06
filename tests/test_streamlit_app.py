"""Lightweight tests for the Streamlit presentation layer.

These cover the demo app's data-access contract, the Interactive Prediction Lab
transforms, and headless page rendering. They do NOT touch or modify the
scientific test suite. Run them in the app environment (Python 3.12 + streamlit),
separate from the locked scientific env:

    .venv-app/Scripts/python.exe -m pytest tests/test_streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip this entire module in the locked scientific env (no streamlit installed).
# The presentation layer is tested in the separate app environment.
pytest.importorskip("streamlit", reason="Streamlit app tests run in the app env (.venv-app)")

import app_data as ad

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "streamlit_app.py")

REQUIRED_RUL_COLS = {"unit_id", "cycle", "actual_rul", "predicted_rul"}
REQUIRED_FR_COLS = {"unit_id", "cycle", "failure_prob", "actual_state"}


def test_bundle_present():
    assert ad.bundle_exists(), "demo/ bundle missing — run scripts/build_demo_bundle.py"


def test_rul_predictions_schema_and_engines():
    df = ad.load_rul_predictions()
    assert REQUIRED_RUL_COLS.issubset(df.columns)
    assert len(ad.validation_engine_ids(df)) == 50


def test_fr_predictions_schema():
    df = ad.load_fr_predictions()
    assert REQUIRED_FR_COLS.issubset(df.columns)


def test_metric_consistency_confusion_matrix():
    """Recomputed CM from demo predictions must equal the frozen metrics."""
    df = ad.load_fr_predictions()
    t = ad.load_metrics()["failure_risk"]["temporal"]
    pred = (df["failure_prob"] >= t["threshold"]).astype(int)
    act = df["actual_state"].astype(int)
    got = (
        int(((pred == 1) & (act == 1)).sum()),
        int(((pred == 1) & (act == 0)).sum()),
        int(((pred == 0) & (act == 0)).sum()),
        int(((pred == 0) & (act == 1)).sum()),
    )
    assert got == (t["TP"], t["FP"], t["TN"], t["FN"])


def test_no_sealed_test_paths_rejected():
    with pytest.raises(ValueError):
        ad._resolve("data/raw/CMAPSSData_v1.0/RUL_FD004.txt")
    with pytest.raises(ValueError):
        ad._resolve("CMAPSSData/test_FD004.txt")


def test_missing_artifact_handled():
    with pytest.raises(ad.MissingArtifact):
        ad.load_decision_csv("does_not_exist.csv")


def test_engine_selection_is_consistent():
    df = ad.load_rul_predictions()
    uid = ad.validation_engine_ids(df)[0]
    sub = ad.engine_slice_predictions(df, uid)
    assert (sub["unit_id"] == uid).all()
    assert sub["cycle"].is_monotonic_increasing


def test_predicted_state_display_transform():
    df = ad.load_fr_predictions()
    one = ad.engine_slice_predictions(df, ad.validation_engine_ids(df)[0])
    states = ad.predicted_state_from_threshold(one, 0.5)
    assert set(states.unique()) <= {0, 1}
    assert int((one["failure_prob"] >= 0.5).sum()) == int(states.sum())


def test_app_data_has_no_model_training_dependency():
    """No training on import: the data layer must not pull ML frameworks."""
    src = (ROOT / "app_data.py").read_text(encoding="utf-8")
    for forbidden in ("sklearn", "torch", "tensorflow", "keras", ".fit("):
        assert forbidden not in src


# --------------------------------------------------------------------------
# Interactive Prediction Lab helpers (deterministic transforms only)
# --------------------------------------------------------------------------
def test_lab_model_artifacts_present():
    for rel in list(ad.MODEL_RUL_FILES.values()) + list(ad.MODEL_FR_FILES.values()):
        assert (ROOT / rel).exists(), f"missing frozen lab artifact: {rel}"


def test_lab_engine_selection_both_models():
    """Engine ids come from the frozen bundle and agree across model views."""
    rul_t = ad.load_rul_predictions("temporal")
    rul_a = ad.load_rul_predictions("anchor")
    assert ad.validation_engine_ids(rul_t) == ad.validation_engine_ids(rul_a)
    assert len(ad.validation_engine_ids(rul_t)) == 50


def test_lab_cycle_selection():
    df = ad.load_rul_predictions("temporal")
    uid = ad.validation_engine_ids(df)[3]
    sub = ad.engine_slice_predictions(df, uid)
    cycles = ad.cycle_options(sub)
    # every offered cycle actually exists in the frozen rows
    for c in (cycles[0], cycles[len(cycles) // 2], cycles[-1]):
        row = ad.row_at_cycle(sub, c)
        assert int(row["cycle"]) == c
        assert int(row["unit_id"]) == uid
    with pytest.raises(ad.MissingArtifact):
        ad.row_at_cycle(sub, max(cycles) + 1)


def test_lab_rul_point_errors():
    df = ad.load_rul_predictions("temporal")
    sub = ad.engine_slice_predictions(df, ad.validation_engine_ids(df)[0])
    pt = ad.rul_point(sub, ad.cycle_options(sub)[0])
    assert pt["signed_error"] == pytest.approx(pt["predicted_rul"] - pt["actual_rul"])
    assert pt["abs_error"] == pytest.approx(abs(pt["signed_error"]))


def test_lab_probability_to_decision_transform():
    """Threshold crossing flips the decision but never the stored probability."""
    df = ad.load_fr_predictions("temporal")
    sub = ad.engine_slice_predictions(df, ad.validation_engine_ids(df)[0])
    row = sub.iloc[0]
    p = float(row["failure_prob"])
    below = ad.fr_point(sub, int(row["cycle"]), max(0.10, p - 0.05))
    above = ad.fr_point(sub, int(row["cycle"]), min(0.90, p + 0.05))
    assert below["failure_prob"] == above["failure_prob"] == p
    if p >= 0.15:
        assert below["predicted_state"] == 1
    if p <= 0.85:
        assert above["predicted_state"] == 0
    assert below["actual_state"] == above["actual_state"] == int(row["actual_state"])


def test_lab_threshold_metrics_reproduce_frozen_result():
    """At the frozen threshold, the lab transform equals the authoritative metrics."""
    df = ad.load_fr_predictions("temporal")
    t = ad.load_metrics()["failure_risk"]["temporal"]
    tm = ad.threshold_metrics(df, t["threshold"])
    assert (tm["TP"], tm["FP"], tm["TN"], tm["FN"]) == (t["TP"], t["FP"], t["TN"], t["FN"])
    assert tm["precision"] == pytest.approx(t["precision"], abs=1e-6)
    assert tm["recall"] == pytest.approx(t["recall"], abs=1e-6)
    assert tm["f1"] == pytest.approx(t["F1"], abs=1e-6)


def test_lab_threshold_moves_monotonically():
    """Raising the threshold can only reduce predicted positives (deterministic)."""
    df = ad.load_fr_predictions("temporal")
    counts = [ad.threshold_metrics(df, thr)["predicted_positive"] for thr in (0.2, 0.4, 0.6, 0.8)]
    assert counts == sorted(counts, reverse=True)


def test_lab_missing_row_behavior():
    df = ad.load_fr_predictions("anchor")
    sub = ad.engine_slice_predictions(df, ad.validation_engine_ids(df)[0])
    with pytest.raises(ad.MissingArtifact):
        ad.fr_point(sub, int(sub["cycle"].max()) + 1, 0.5)
    with pytest.raises(ValueError):
        ad.load_rul_predictions("no_such_model")


def test_lab_examples_reference_real_frozen_rows():
    rul = ad.load_rul_predictions("temporal")
    fr = ad.load_fr_predictions("temporal")
    ex = ad.derive_examples(rul, fr)
    assert 3 <= len(ex) <= 5
    units = [e["unit_id"] for e in ex]
    assert len(units) == len(set(units)), "examples must be distinct engines"
    ids = set(ad.validation_engine_ids(rul))
    for e in ex:
        assert e["unit_id"] in ids
        sub = ad.engine_slice_predictions(rul, e["unit_id"])
        assert e["cycle"] in ad.cycle_options(sub)


def test_app_source_has_no_model_training_dependency():
    """No training on import: the app entrypoint must not pull ML frameworks."""
    src = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    for forbidden in ("sklearn", "torch", "tensorflow", "keras", ".fit("):
        assert forbidden not in src


def test_demo_code_has_no_test_label_access():
    """App code accesses no sealed official-test paths (docs/comments excluded)."""
    import re

    forbidden = ("test_FD004", "RUL_FD004", "CMAPSSData", "data/raw")
    guard_list = (
        '"CMAPSSData",\n    "data/raw",\n    "test_FD004",\n    "RUL_FD004",'
    )
    for name in ("app_data.py", "streamlit_app.py", "scripts/build_demo_bundle.py"):
        src = (ROOT / name).read_text(encoding="utf-8")
        src = src.replace(guard_list, "")  # the rejection guard list itself
        src = re.sub(r'""".*?"""', "", src, flags=re.S)  # docstrings
        code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
        for tok in forbidden:
            assert tok not in code, f"{name} accesses {tok}"


# --------------------------------------------------------------------------
# Headless rendering (each page must run without an exception)
# --------------------------------------------------------------------------
from streamlit.testing.v1 import AppTest  # noqa: E402


def _at() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_launches_on_overview():
    at = _at()
    assert not at.exception
    assert at.title[0].value == "Predictive Maintenance Intelligence"


@pytest.mark.parametrize(
    "page",
    [
        "Interactive Prediction Lab",
        "Model Performance",
        "Engine Explorer",
        "Failure Risk",
        "Decision Analysis",
        "Why No LSTM?",
        "Reproducibility",
        "Limitations",
        "Evidence / Methodology",
    ],
)
def test_every_page_renders(page):
    at = _at()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page} raised: {at.exception}"


def _at_page(page: str) -> "AppTest":
    at = _at()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception
    return at


def _select(at: "AppTest", key: str, raw_value) -> None:
    """Set a selectbox by its RAW option value (AppTest formats the label)."""
    for i, w in enumerate(at.selectbox):
        if w.key == key:
            at.selectbox[i].set_value(raw_value).run()
            return
    raise AssertionError(f"selectbox {key!r} not found")


def test_lab_widgets_update_without_exception():
    """Engine, cycle, model and threshold widgets drive the lab page live."""
    at = _at_page("Interactive Prediction Lab")
    ids = ad.validation_engine_ids(ad.load_rul_predictions("temporal"))

    # engine selection (raw unit id value)
    _select(at, "lab_unit", ids[7])
    assert not at.exception

    # cycle selection (raw cycle value from the newly selected engine)
    cycles = ad.cycle_options(
        ad.engine_slice_predictions(ad.load_rul_predictions("temporal"), ids[7])
    )
    _select(at, "lab_cycle", cycles[-1])
    assert not at.exception

    # model selection (raw model key)
    _select(at, "lab_model", ad.MODEL_KEYS[1])
    assert not at.exception

    # threshold change
    at.slider[0].set_value(0.80).run()
    assert not at.exception

    # example buttons exist and apply without exception
    assert len(at.button) >= 3
    at.button[0].click().run()
    assert not at.exception


def test_lab_rul_values_change_with_engine():
    """Displayed RUL metrics must differ between two distinct validation engines."""
    at = _at_page("Interactive Prediction Lab")
    ids = ad.validation_engine_ids(ad.load_rul_predictions("temporal"))
    _select(at, "lab_unit", ids[0])
    first = [m.value for m in at.metric][:2]
    _select(at, "lab_unit", ids[1])
    second = [m.value for m in at.metric][:2]
    assert not at.exception
    assert first != second


def test_lab_decision_flips_when_threshold_crosses_probability():
    """Moving the slider across the stored probability changes the alert decision."""
    at = _at_page("Interactive Prediction Lab")
    fr_all = ad.load_fr_predictions("temporal")
    ids = ad.validation_engine_ids(ad.load_rul_predictions("temporal"))

    # find (verified in the frozen data) an engine with a cycle whose stored
    # probability sits mid-band, so thresholds below/above both flip the decision
    target = None
    for uid in ids:
        fr_engine = ad.engine_slice_predictions(fr_all, int(uid)).sort_values("failure_prob")
        band = fr_engine[
            (fr_engine["failure_prob"] >= 0.25) & (fr_engine["failure_prob"] <= 0.75)
        ]
        if not band.empty:
            row = band.iloc[len(band) // 2]
            target = (int(uid), float(row["failure_prob"]), int(row["cycle"]))
            break
    assert target, "no validation engine has a cycle probability inside [0.25, 0.75]"
    uid, prob, mid_cycle = target

    _select(at, "lab_unit", uid)
    _select(at, "lab_cycle", mid_cycle)
    assert not at.exception

    at.slider[0].set_value(prob - 0.05).run()
    below = [m.value for m in at.metric if m.label == "Predicted state"][0]
    at.slider[0].set_value(prob + 0.05).run()
    above = [m.value for m in at.metric if m.label == "Predicted state"][0]
    assert not at.exception
    assert below == "At risk"
    assert above == "No risk"

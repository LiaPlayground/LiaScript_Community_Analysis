"""Manual validation GUI for the LiaScript corpus.

Samples 50 validated + 50 rejected markdown candidate files, presents each
(content hidden by ground-truth label), records the reviewer's verdict and
computes precision/recall at the end.

Run: PIPENV_IGNORE_VIRTUALENVS=1 pipenv run streamlit run scripts/validation_gui.py
"""
import pickle
import random
import pandas as pd
import streamlit as st
from pathlib import Path

DATA = Path("/home/sz/Desktop/Python/LiaScript_Paper/data_march_2026/liascript_march_2026/raw")
FILES_DIR = DATA / "files"
RESULTS = Path("/home/sz/Desktop/Python/LiaScript_Paper/validation_results.csv")
SEED = 42
N_VALID = 50
N_REJECTED = 50


@st.cache_data
def load_sample():
    with open(DATA / "LiaScript_files_validated.p", "rb") as f:
        vd = pickle.load(f)
    valid = vd[vd["pipe:is_valid_liascript"] == True].sample(N_VALID, random_state=SEED)
    rejected = vd[vd["pipe:is_valid_liascript"] == False].sample(N_REJECTED, random_state=SEED)
    sample = pd.concat([valid, rejected]).reset_index(drop=True)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return sample


def load_results() -> pd.DataFrame:
    if RESULTS.exists():
        return pd.read_csv(RESULTS, dtype={"pipe:ID": str})
    return pd.DataFrame(columns=["pipe:ID", "verdict", "notes"])


def save_verdict(pipe_id: str, verdict: str, notes: str) -> None:
    df = load_results()
    df = df[df["pipe:ID"] != pipe_id]
    new = pd.DataFrame([{"pipe:ID": pipe_id, "verdict": verdict, "notes": notes}])
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(RESULTS, index=False)


def get_content(pipe_id: str) -> str:
    p = FILES_DIR / f"{pipe_id}.md"
    if p.exists():
        try:
            return p.read_text(errors="replace")
        except Exception as e:
            return f"(read error: {e})"
    return "(content not found locally)"


def compute_metrics(merged: pd.DataFrame) -> dict:
    # truth: validated (pipeline said yes) vs rejected (pipeline said no)
    # verdict: course vs not_course (human judgement)
    m = merged[merged["verdict"].isin(["course", "not_course"])]
    tp = ((m["truth"] == "validated") & (m["verdict"] == "course")).sum()
    fp = ((m["truth"] == "validated") & (m["verdict"] == "not_course")).sum()
    fn = ((m["truth"] == "rejected") & (m["verdict"] == "course")).sum()
    tn = ((m["truth"] == "rejected") & (m["verdict"] == "not_course")).sum()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec}


# ---------------------------------------------------------------------------
st.set_page_config(page_title="LiaScript Validation", layout="wide")
st.title("LiaScript pipeline: manual validation")

sample = load_sample()
done = load_results()
remaining = sample[~sample["pipe:ID"].astype(str).isin(done["pipe:ID"].astype(str))]

total = len(sample)
done_n = total - len(remaining)
st.progress(done_n / total, text=f"{done_n} / {total} reviewed")

# ----- done screen ---------------------------------------------------------
if len(remaining) == 0:
    st.success("All files reviewed.")
    merged = sample.merge(done, on="pipe:ID", how="left")
    merged["truth"] = merged["pipe:is_valid_liascript"].map(
        {True: "validated", False: "rejected"}
    )
    st.subheader("Confusion matrix (rows = pipeline label, cols = human verdict)")
    ct = pd.crosstab(merged["truth"], merged["verdict"], dropna=False)
    st.dataframe(ct)
    met = compute_metrics(merged)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TP (valid, is course)", met["tp"])
    c2.metric("FP (valid, not course)", met["fp"])
    c3.metric("FN (rejected, is course)", met["fn"])
    c4.metric("TN (rejected, not course)", met["tn"])
    c5, c6 = st.columns(2)
    c5.metric("Precision", f"{100*met['precision']:.1f}%")
    c6.metric("Recall (within sample)", f"{100*met['recall']:.1f}%")
    st.caption(
        "Recall is estimated within the stratified 50/50 sample; the population "
        "recall is a downweighted version (rejected files are 53,779 of 57,096 in "
        "the corpus)."
    )
    with st.expander("Per-file verdicts"):
        st.dataframe(merged[["pipe:ID", "repo_user", "file_name",
                             "truth", "pipe:validation_method",
                             "verdict", "notes"]])
    st.stop()

# ----- review screen -------------------------------------------------------
current = remaining.iloc[0]
pid = str(current["pipe:ID"])

left, right = st.columns([2, 3])

with left:
    st.subheader("File")
    st.write(f"**Repo** &nbsp; {current['repo_user']} / {current['repo_name']}")
    st.write(f"**Path** &nbsp; {current['file_name']}")
    st.caption(f"ID {pid}")

    st.markdown("---")
    st.subheader("Verdict")
    notes = st.text_input("Notes (optional)", key=f"notes_{pid}")
    cA, cB = st.columns(2)
    with cA:
        if st.button("LiaScript course", key=f"yes_{pid}", use_container_width=True):
            save_verdict(pid, "course", notes)
            st.rerun()
        if st.button("Unclear", key=f"unclear_{pid}", use_container_width=True):
            save_verdict(pid, "unclear", notes)
            st.rerun()
    with cB:
        if st.button("Not a course", key=f"no_{pid}", use_container_width=True):
            save_verdict(pid, "not_course", notes)
            st.rerun()
        if st.button("Skip", key=f"skip_{pid}", use_container_width=True):
            save_verdict(pid, "skipped", notes)
            st.rerun()

    with st.expander("Rule-based indicators (diagnostic)"):
        indi = {c.replace("liaIndi_", ""): current[c]
                for c in current.index if c.startswith("liaIndi_")}
        st.json(indi)

with right:
    st.subheader("Content")
    content = get_content(pid)
    st.caption(f"Length: {len(content):,} chars")
    tabs = st.tabs(["Raw", "Rendered"])
    with tabs[0]:
        st.code(content[:5000], language="markdown")
    with tabs[1]:
        st.markdown(content[:5000])

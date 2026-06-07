import re
from io import StringIO

import requests
import streamlit as st
import streamlit.components.v1 as components

import py3Dmol

st.set_page_config(page_title="ESMFold Protein Predictor", layout="wide")

st.title("🧬 ESMFold Protein Structure Prediction (ESM Atlas API)")
st.caption("Upload a FASTA or paste a sequence. Predict structure and color by confidence (pLDDT).")

# ESM Atlas foldSequence endpoint that returns PDB
ESMFOLD_ENDPOINT = "https://api.esmatlas.com/foldSequence/v1/pdb/"


def clean_sequence(seq: str) -> str:
    """Remove whitespace and validate amino-acid letters."""
    seq = re.sub(r"\s+", "", seq).upper()
    if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYBXZJUO]*", seq):
        raise ValueError("Sequence contains invalid characters (non amino-acid letters).")
    return seq


def parse_fasta(text: str):
    """
    Returns list of (header, sequence). If no '>' headers, treats entire input as one sequence.
    """
    text = text.strip()
    if not text:
        return []

    if not text.lstrip().startswith(">"):
        return [("input", clean_sequence(text))]

    entries = []
    header = None
    seq_lines = []

    for line in StringIO(text):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                entries.append((header, clean_sequence("".join(seq_lines))))
            header = line[1:].strip() or "unnamed"
            seq_lines = []
        else:
            seq_lines.append(line)

    if header is not None:
        entries.append((header, clean_sequence("".join(seq_lines))))

    return entries


@st.cache_data(show_spinner=False)
def predict_pdb(sequence: str, timeout_s: int = 180) -> str:
    """
    POST sequence as text/plain to ESMFold API, returns PDB text.
    """
    headers = {"Content-Type": "text/plain"}
    r = requests.post(ESMFOLD_ENDPOINT, data=sequence, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    return r.text


def extract_confidence_stats_from_pdb(pdb_str: str) -> dict:
    """
    Extract B-factor values from ATOM/HETATM lines and summarize.
    ESMFold outputs typically store confidence (pLDDT) in the PDB B-factor field.
    """
    bvals = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                b = float(line[60:66].strip())  # B-factor col
                bvals.append(b)
            except Exception:
                pass

    if not bvals:
        return {"count": 0}

    bmin = min(bvals)
    bmax = max(bvals)
    mean = sum(bvals) / len(bvals)

    # Heuristic hint about scale
    scale = "0–1" if bmax <= 1.5 else "0–100"
    return {"count": len(bvals), "min": bmin, "max": bmax, "mean": mean, "scale_hint": scale}


def render_pdb_with_confidence_html(pdb_str: str, bmin: float, bmax: float, gradient: str = "roygb"):
    """
    Render PDB with confidence coloring using py3Dmol, embedded via Streamlit HTML.
    This is more reliable than stmol on many Windows setups.
    """
    view = py3Dmol.view(width=900, height=650)
    view.addModel(pdb_str, "pdb")

    view.setStyle(
        {"cartoon": {"colorscheme": {"prop": "b", "gradient": gradient, "min": bmin, "max": bmax}}}
    )

    view.setBackgroundColor("white")
    view.zoomTo()

    components.html(view._make_html(), height=700, scrolling=False)


with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload FASTA (.fa/.fasta/.txt)", type=["fa", "fasta", "txt"])
    st.divider()
    st.header("API / Display")
    timeout_s = st.slider("Request timeout (seconds)", 30, 300, 180, 10)
    gradient = st.selectbox("Confidence color gradient", ["roygb", "sinebow", "rwb"], index=0)
    st.caption("Coloring uses PDB B-factor as confidence (pLDDT) when available.")


default_text = """>Example
MGSSHHHHHHSSGLVPRGSHMALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDY
"""

seq_text = st.text_area("Or paste FASTA / sequence here", value=default_text, height=200)

if uploaded is not None:
    try:
        file_text = uploaded.read().decode("utf-8", errors="replace")
        seq_text = file_text
        st.info(f"Loaded file: {uploaded.name}")
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")

col1, col2 = st.columns([1, 1], gap="large")

if st.button("Predict + View 3D", type="primary"):
    try:
        entries = parse_fasta(seq_text)
        if not entries:
            st.error("No sequence found.")
            st.stop()

        if len(entries) > 1:
            st.warning("Multiple FASTA entries found. Running only the FIRST one.")

        header, sequence = entries[0]

        if len(sequence) < 10:
            st.error("Sequence is too short.")
            st.stop()

        with st.spinner("Calling ESMFold API…"):
            pdb_text = predict_pdb(sequence, timeout_s=timeout_s)

        stats = extract_confidence_stats_from_pdb(pdb_text)

        with col1:
            st.subheader("Result")
            st.write(f"**Name:** {header}")
            st.write(f"**Length:** {len(sequence)} aa")

            if stats.get("count", 0) > 0:
                st.write(
                    f"**Confidence (from B-factor):** min={stats['min']:.3f}, "
                    f"mean={stats['mean']:.3f}, max={stats['max']:.3f} "
                    f"(scale hint: {stats['scale_hint']})"
                )
            else:
                st.warning("No B-factor values parsed from PDB; confidence coloring may not work.")

            st.download_button(
                "Download PDB",
                data=pdb_text,
                file_name=f"{header.replace(' ', '_')}.pdb",
                mime="chemical/x-pdb",
            )
            st.text_area("PDB preview", pdb_text[:8000], height=520)

        with col2:
            st.subheader("3D View (colored by confidence)")
            if stats.get("count", 0) > 0:
                # Normalize range for nicer colors
                if stats["max"] <= 1.5:
                    bmin, bmax = 0.0, 1.0
                else:
                    bmin, bmax = 0.0, 100.0
            else:
                bmin, bmax = 0.0, 1.0

            render_pdb_with_confidence_html(pdb_text, bmin=bmin, bmax=bmax, gradient=gradient)

    except requests.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:1000] if e.response is not None else ""
        except Exception:
            pass
        st.error(f"API error: {e}\n\n{body}")
    except requests.RequestException as e:
        st.error(f"Network error: {e}")
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {e}")

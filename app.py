# app.py
import io
import re
import random
from dataclasses import dataclass

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================
# K-MEANS (FROM SCRATCH)
# ========================= 
@dataclass
class KMeansResult:
    labels: list
    centroids: list
    sse: float
    iterations: int
    restarts_used: int


def sq_dist(a, b):
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return s


def compute_sse(data, labels, centroids):
    sse = 0.0
    for i, x in enumerate(data):
        sse += sq_dist(x, centroids[labels[i]])
    return sse


def random_init_from_data(data, k):
    n = len(data)
    used = set()
    centroids = []
    while len(centroids) < k:
        idx = random.randrange(n)
        if idx in used:
            continue
        used.add(idx)
        centroids.append(list(data[idx]))
    return centroids


def assign_labels(data, centroids, labels):
    changed = False
    for i, x in enumerate(data):
        best = 0
        bestd = sq_dist(x, centroids[0])
        for c in range(1, len(centroids)):
            d = sq_dist(x, centroids[c])
            if d < bestd:
                bestd = d
                best = c
        if labels[i] != best:
            labels[i] = best
            changed = True
    return changed


def recompute_centroids(data, labels, k):
    d = len(data[0])
    centroids = [[0.0] * d for _ in range(k)]
    counts = [0] * k

    for x, lab in zip(data, labels):
        counts[lab] += 1
        for j in range(d):
            centroids[lab][j] += x[j]

    for c in range(k):
        if counts[c] == 0:
            centroids[c] = list(random.choice(data))  # empty cluster -> random point
        else:
            for j in range(d):
                centroids[c][j] /= counts[c]
    return centroids


def centroids_almost_equal(A, B, eps=1e-9):
    for i in range(len(A)):
        for j in range(len(A[i])):
            if abs(A[i][j] - B[i][j]) > eps:
                return False
    return True


def parse_manual_centers(text: str, k: int, d: int):
    """
    Örn: 10 20 | 80 70 | 40 10
    (| ile merkezleri ayır; her merkez d sayı içermeli)
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("Manuel merkezler boş.")

    parts = [p.strip() for p in t.split("|") if p.strip()]
    if len(parts) != k:
        raise ValueError(f"k={k} için {k} merkez girmelisin. Girilen: {len(parts)}")

    float_pattern = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
    centers = []
    for i, p in enumerate(parts, 1):
        tokens = float_pattern.findall(p)
        nums = [float(tok.replace(",", ".")) for tok in tokens]
        if len(nums) != d:
            raise ValueError(f"Merkez #{i} boyutu {d} olmalı. Bulunan: {len(nums)}")
        centers.append(nums)
    return centers


def kmeans_fit(data, k, max_iter, restarts, manual_centroids=None):
    if not data:
        raise ValueError("Veri yok.")
    n = len(data)
    d = len(data[0])
    if k < 2:
        raise ValueError("k en az 2 olmalı.")
    if k > n:
        raise ValueError("k, satır sayısından büyük olamaz.")

    attempts = 1 if manual_centroids is not None else max(1, restarts)
    best = None

    for attempt in range(attempts):
        centroids = [c[:] for c in manual_centroids] if manual_centroids is not None else random_init_from_data(data, k)
        labels = [-1] * n

        prev = None
        stable = 0
        it_done = 0

        for it in range(1, max_iter + 1):
            it_done = it
            changed = assign_labels(data, centroids, labels)
            newc = recompute_centroids(data, labels, k)

            same = (prev is not None) and centroids_almost_equal(prev, newc, 1e-9)
            stable = stable + 1 if same else 0

            prev = centroids
            centroids = newc

            if stable >= 2:
                break
            if not changed:
                break

        sse = compute_sse(data, labels, centroids)
        cand = KMeansResult(labels=labels[:], centroids=[c[:] for c in centroids],
                           sse=sse, iterations=it_done, restarts_used=attempt + 1)

        if best is None or cand.sse < best.sse:
            best = cand

    return best


# =========================
# HELPERS
# =========================
def read_csv_auto(file_bytes: bytes):
    return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", encoding_errors="ignore")


def read_xlsx(file_bytes: bytes):
    return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")


def make_random_df(n: int, d: int, rng: float):
    cols = [f"X{i+1}" for i in range(d)]
    return pd.DataFrame({c: [random.random() * rng for _ in range(n)] for c in cols})


def to_numeric_feature(df: pd.DataFrame, cols: list):
    sub = df[cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    before = len(sub)
    sub = sub.dropna().reset_index(drop=True)
    dropped = before - len(sub)
    return sub, dropped


def normalize_df(df: pd.DataFrame, method: str):
    if method == "none":
        return df
    x = df.copy()
    if method == "minmax":
        for c in x.columns:
            mn, mx = x[c].min(), x[c].max()
            x[c] = 0.0 if (mx - mn) == 0 else (x[c] - mn) / (mx - mn)
        return x
    if method == "zscore":
        for c in x.columns:
            mu, sd = x[c].mean(), x[c].std()
            x[c] = 0.0 if (sd == 0 or pd.isna(sd)) else (x[c] - mu) / sd
        return x
    return df


def reset_all():
    st.session_state.raw_df = None
    st.session_state.feat_df = None
    st.session_state.feat_df_used = None
    st.session_state.selected_cols = []
    st.session_state.dropped = 0
    st.session_state.result = None


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="K-Means (Sıfırdan) - Streamlit", layout="wide")
st.title("K-Means Kümeleme (Sıfırdan) — Streamlit")
st.caption("K-Means tamamen el yazımıdır (hazır KMeans yok).")

# session init
for key, default in [
    ("raw_df", None),
    ("feat_df", None),
    ("feat_df_used", None),
    ("selected_cols", []),
    ("dropped", 0),
    ("result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ===== SIDEBAR =====
with st.sidebar:
    st.header("Kontrol Paneli")

    if st.button("🔄 Her şeyi sıfırla", use_container_width=True):
        reset_all()
        st.rerun()

    st.divider()
    st.subheader("1) Veri yükle")

    uploaded = st.file_uploader("CSV / XLSX seç", type=["csv", "xlsx"])
    load_btn = st.button("📥 Dosyayı yükle", use_container_width=True)

    st.caption("Önemli: Dosya sadece bu butona basınca okunur (rerun’da sıfırlanmaz).")

    st.divider()
    st.subheader("veya Random üret")

    n = st.number_input("N (satır)", min_value=2, value=120, step=1)
    d = st.number_input("d (sütun)", min_value=2, value=5, step=1)
    rng = st.number_input("range", min_value=0.000001, value=100.0, step=1.0)
    gen_btn = st.button("🎲 Random üret", use_container_width=True)

    st.divider()
    st.subheader("2) K-Means ayarları")

    k = st.number_input("k", min_value=2, value=3, step=1)
    max_iter = st.number_input("maxIter", min_value=1, value=50, step=1)
    restarts = st.number_input("restarts", min_value=1, value=5, step=1)

    norm = st.selectbox(
        "Ön-işleme",
        options=[("none", "Yok"), ("minmax", "Min-Max [0,1]"), ("zscore", "Z-Score")],
        format_func=lambda x: x[1],
        index=0
    )[0]

    manual = st.checkbox("Manuel merkez gireceğim")
    manual_text = st.text_area("Merkezler (k adet)  | ile ayır", disabled=not manual, height=90)

    run_btn = st.button("▶️ K-Means çalıştır", type="primary", use_container_width=True)

# ===== ACTIONS (load/random) =====
if gen_btn:
    st.session_state.raw_df = make_random_df(int(n), int(d), float(rng))
    st.session_state.feat_df = None
    st.session_state.feat_df_used = None
    st.session_state.selected_cols = []
    st.session_state.dropped = 0
    st.session_state.result = None
    st.success("✅ Random veri üretildi.")

if load_btn:
    if uploaded is None:
        st.error("Dosya seçmedin.")
    else:
        try:
            name = uploaded.name.lower()
            fb = uploaded.getvalue()
            if name.endswith(".xlsx"):
                df = read_xlsx(fb)
            else:
                df = read_csv_auto(fb)

            st.session_state.raw_df = df
            st.session_state.feat_df = None
            st.session_state.feat_df_used = None
            st.session_state.selected_cols = []
            st.session_state.dropped = 0
            st.session_state.result = None
            st.success(f"✅ Dosya yüklendi: {len(df)} satır / {df.shape[1]} sütun")
        except Exception as e:
            st.error(f"Dosya okunamadı: {e}")

raw_df = st.session_state.raw_df
feat_df = st.session_state.feat_df
result = st.session_state.result

# ===== MAIN =====
tab1, tab2, tab3 = st.tabs(["📥 Veri", "🧠 Model", "📊 Sonuçlar"])

with tab1:
    st.subheader("Veri & Sütun Seçimi")

    c1, c2, c3 = st.columns(3)
    c1.metric("Ham veri", "—" if raw_df is None else f"{len(raw_df)} satır / {raw_df.shape[1]} sütun")
    c2.metric("Feature", "—" if feat_df is None else f"{len(feat_df)} satır / {feat_df.shape[1]} boyut")
    c3.metric("Düşen satır", st.session_state.dropped)

    if raw_df is None:
        st.info("Soldan dosya seçip **Dosyayı yükle**’ye bas veya random üret.")
    else:
        with st.form("col_form"):
            cols = list(raw_df.columns)
            selected = st.multiselect("Kullanılacak sayısal sütunlar (en az 2)", cols, default=st.session_state.selected_cols)
            apply_cols = st.form_submit_button("✅ Seçili sütunları uygula")

        if apply_cols:
            if len(selected) < 2:
                st.error("En az 2 sütun seçmelisin.")
            else:
                sub, dropped = to_numeric_feature(raw_df, selected)
                if len(sub) == 0:
                    st.error("Seçilen sütunlarda sayısal satır kalmadı (hepsi düştü).")
                else:
                    st.session_state.selected_cols = selected
                    st.session_state.feat_df = sub
                    st.session_state.feat_df_used = None
                    st.session_state.dropped = dropped
                    st.session_state.result = None
                    st.success(f"✅ Feature hazır: {len(sub)} satır / {sub.shape[1]} boyut | düşen={dropped}")

        st.markdown("### Ham veri (ilk 20)")
        st.dataframe(raw_df.head(20), use_container_width=True)

        if st.session_state.feat_df is not None:
            st.markdown("### Feature veri (ilk 20)")
            st.dataframe(st.session_state.feat_df.head(20), use_container_width=True)

with tab2:
    st.subheader("Model Çalıştırma")

    if st.session_state.feat_df is None:
        st.info("Önce **Veri** sekmesinde sayısal sütunları seçip uygula.")
    else:
        used = normalize_df(st.session_state.feat_df, norm)
        st.session_state.feat_df_used = used

        st.write("Seçili sütunlar:", st.session_state.selected_cols)
        st.write("Ön-işleme:", {"none": "Yok", "minmax": "Min-Max", "zscore": "Z-Score"}[norm])

        with st.expander("Feature istatistikleri", expanded=False):
            st.dataframe(used.describe().T, use_container_width=True)

        if run_btn:
            try:
                data = used.values.tolist()
                if int(k) > len(data):
                    st.error(f"k={int(k)} satır sayısından büyük olamaz. (satır={len(data)})")
                else:
                    manual_centroids = None
                    if manual:
                        manual_centroids = parse_manual_centers(manual_text, int(k), used.shape[1])

                    r = kmeans_fit(data, int(k), int(max_iter), int(restarts), manual_centroids)
                    st.session_state.result = r
                    st.success(f"✅ Bitti | SSE={r.sse:.6f} | iter={r.iterations} | restartsUsed={r.restarts_used}")
            except Exception as e:
                st.error(str(e))

with tab3:
    st.subheader("Sonuçlar")

    if st.session_state.result is None:
        st.info("Önce **Model** sekmesinde K-Means çalıştır.")
    else:
        used = st.session_state.feat_df_used
        r: KMeansResult = st.session_state.result

        m1, m2, m3 = st.columns(3)
        m1.metric("SSE", f"{r.sse:.6f}")
        m2.metric("İterasyon", r.iterations)
        m3.metric("RestartsUsed", r.restarts_used)

        # cluster counts
        k_val = len(r.centroids)
        counts = [0] * k_val
        for lab in r.labels:
            counts[lab] += 1

        st.markdown("### Cluster başına kaç nokta var?")
        rows = []
        for i in range(k_val):
            c = r.centroids[i]
            short = ", ".join([f"{v:.3f}" for v in c[:min(4, len(c))]]) + (" ,..." if len(c) > 4 else "")
            rows.append({"cluster": i, "count": counts[i], "centroid(short)": short})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("### 2D Görselleştirme")
        cols = list(used.columns)
        x_axis = st.selectbox("X ekseni", cols, index=0)
        y_axis = st.selectbox("Y ekseni", cols, index=1 if len(cols) > 1 else 0)

        x = used[x_axis].values
        y = used[y_axis].values
        labels = r.labels

        fig = plt.figure(figsize=(9, 4.5))
        ax = fig.add_subplot(111)
        for lab in sorted(set(labels)):
            idx = [i for i in range(len(labels)) if labels[i] == lab]
            ax.scatter(x[idx], y[idx], s=18, label=f"C{lab}")

        cx_idx = cols.index(x_axis)
        cy_idx = cols.index(y_axis)
        cx = [c[cx_idx] for c in r.centroids]
        cy = [c[cy_idx] for c in r.centroids]
        ax.scatter(cx, cy, s=160, marker="x")

        ax.set_title("K-Means Sonuçları (2D projeksiyon)")
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.legend(fontsize=8)
        st.pyplot(fig)

        st.divider()
        st.markdown("### Export (CSV)")
        out = used.copy()
        out["label"] = labels
        st.download_button(
            "⬇️ Sonucu CSV indir (feature + label)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="kmeans_result.csv",
            mime="text/csv",
            use_container_width=True
        )

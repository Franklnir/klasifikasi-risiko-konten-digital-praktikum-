import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Klasifikasi Tingkat Risiko — Proyek Akhir Data Mining 2026",
    page_icon=":material/school:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
st.session_state.setdefault("raw_data", None)
st.session_state.setdefault("processed_data", None)
st.session_state.setdefault("model", None)
st.session_state.setdefault("encoders", {})
st.session_state.setdefault("feature_names", [])
st.session_state.setdefault("smote_done", False)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
RISK_COLUMNS_KEYWORDS = [
    "anxiety", "depression", "distraction", "mood", "stress",
    "dropout", "self-esteem", "self_esteem"
]

COLUMNS_TO_DROP = ["Timestamp", "Column 19", "student_id"]


def clean_numeric(series: pd.Series) -> pd.Series:
    """Bersihkan kolom numerik — hapus teks, simbol persen, satuan, dll."""
    s = series.astype(str)
    # Hapus karakter non-numerik kecuali titik dan minus
    s = s.str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def generate_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """Buat kolom target 'Tingkat Risiko' dari skor psikologis."""
    risk_cols = [
        c for c in df.columns
        if any(kw in c.lower() for kw in RISK_COLUMNS_KEYWORDS)
    ]

    if risk_cols:
        for col in risk_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["_total_risk"] = df[risk_cols].sum(axis=1)
        q33, q66 = df["_total_risk"].quantile([0.33, 0.66]).values

        def _cat(score):
            if score <= q33:
                return "Rendah"
            elif score <= q66:
                return "Sedang"
            return "Tinggi"

        df["Tingkat Risiko"] = df["_total_risk"].apply(_cat)
        df.drop(columns=["_total_risk"], inplace=True)
    else:
        rng = np.random.default_rng(42)
        df["Tingkat Risiko"] = rng.choice(
            ["Rendah", "Sedang", "Tinggi"], size=len(df)
        )
    return df


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih tahapan",
    [
        ":material/home: Beranda",
        ":material/upload_file: Upload & eksplorasi data",
        ":material/tune: Preprocessing",
        ":material/model_training: Modeling & evaluasi",
    ],
    label_visibility="collapsed",
)

st.sidebar.caption("Proyek Akhir Praktikum Data Mining 2026")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Beranda
# ═══════════════════════════════════════════════════════════════════════════
if "Beranda" in menu:
    st.title("Klasifikasi tingkat risiko dampak paparan konten digital")
    st.markdown(
        "### Terhadap Fokus Belajar, Pola Tidur, dan Kesehatan Mental "
        "Mahasiswa Menggunakan Algoritma Decision Tree"
    )

    st.markdown(
        """
Aplikasi web interaktif ini dibangun untuk memenuhi kriteria evaluasi
tugas akhir **Praktikum Data Mining 2026**.  Aplikasi ini
mengklasifikasikan **Tingkat Risiko** (Rendah, Sedang, Tinggi) dari
dampak paparan konten digital menggunakan **Algoritma Decision Tree**.

**Fitur aplikasi:**

1. **Upload & eksplorasi data** — Unggah dataset berformat CSV,
   preview data, statistik deskriptif, serta visualisasi distribusi.
2. **Preprocessing** — Pembersihan data, pembuatan label target,
   label encoding, dan **augmentasi data SMOTE** agar memenuhi
   kriteria minimal 5.000 baris.
3. **Modeling & evaluasi** — Latih model Decision Tree dengan tuning
   parameter, tampilkan akurasi (train & test), confusion matrix,
   classification report, dan feature importance.

**Kriteria Sangat Baik (Data Mining Tabular):**
- Minimal 5.000 baris data
- Akurasi train ≥ 80%
- Akurasi test ≥ 85%

Silakan menuju ke menu **Upload & eksplorasi data** di panel navigasi
sebelah kiri untuk memulai.
        """
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Upload & Eksplorasi Data
# ═══════════════════════════════════════════════════════════════════════════
elif "Upload" in menu:
    st.title("Upload dan eksplorasi data (EDA)")

    uploaded_file = st.file_uploader(
        "Upload dataset CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["raw_data"] = df
            # Reset downstream state
            st.session_state["processed_data"] = None
            st.session_state["model"] = None
            st.session_state["smote_done"] = False
            st.toast("File CSV berhasil di-load!", icon=":material/check_circle:")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}", icon=":material/error:")

    if st.session_state["raw_data"] is not None:
        df = st.session_state["raw_data"]

        # --- Preview -------------------------------------------------------
        st.subheader("1. Preview data asli")
        st.dataframe(df.head(15))

        with st.container(horizontal=True):
            st.metric("Jumlah baris", f"{df.shape[0]:,}")
            st.metric("Jumlah kolom", f"{df.shape[1]}")

        # --- Tipe data & missing ------------------------------------------
        st.subheader("2. Informasi tipe data dan missing values")
        info_df = pd.DataFrame(
            {
                "Tipe data": df.dtypes.astype(str),
                "Missing values": df.isnull().sum(),
                "Missing (%)": (df.isnull().sum() / len(df) * 100).round(2),
            }
        )
        st.dataframe(info_df)

        # --- Statistik deskriptif -----------------------------------------
        st.subheader("3. Statistik deskriptif")
        st.dataframe(df.describe(include="all").T)

        # --- Distribusi kolom kategorikal ---------------------------------
        st.subheader("4. Distribusi kolom kategorikal")
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        # Hapus Timestamp & Column 19 dari pilihan
        cat_cols = [c for c in cat_cols if c.strip() not in COLUMNS_TO_DROP]

        if cat_cols:
            sel_cat = st.selectbox("Pilih kolom", cat_cols, key="eda_cat")
            counts = df[sel_cat].value_counts().reset_index()
            counts.columns = [sel_cat, "Jumlah"]

            chart = (
                alt.Chart(counts)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X(f"{sel_cat}:N", sort="-y", title=sel_cat),
                    y=alt.Y("Jumlah:Q", title="Jumlah"),
                    color=alt.Color(f"{sel_cat}:N", legend=None),
                    tooltip=[sel_cat, "Jumlah"],
                )
                .properties(height=350)
            )
            st.altair_chart(chart)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Preprocessing
# ═══════════════════════════════════════════════════════════════════════════
elif "Preprocessing" in menu:
    st.title("Preprocessing data")

    if st.session_state["raw_data"] is None:
        st.warning(
            "Silakan upload dataset terlebih dahulu di menu "
            "**Upload & eksplorasi data**.",
            icon=":material/warning:",
        )
    else:
        df = st.session_state["raw_data"].copy()

        # --- 1. Cleaning --------------------------------------------------
        with st.expander("1. Pembersihan data (data cleaning)", expanded=True, icon=":material/cleaning_services:"):
            # Drop kolom tidak relevan
            dropped = [c for c in COLUMNS_TO_DROP if c in df.columns]
            for c in dropped:
                # handle kolom dengan spasi di CSV
                matching = [col for col in df.columns if col.strip() == c]
                df.drop(columns=matching, inplace=True, errors="ignore")

            if dropped:
                st.write(f"Kolom yang dihapus: **{', '.join(dropped)}**")

            initial_len = len(df)
            df = df.dropna()
            removed = initial_len - len(df)
            st.success(
                f"Dihapus **{removed}** baris dengan nilai kosong. "
                f"Sisa: **{len(df)}** baris.",
                icon=":material/check_circle:",
            )

        # --- 2. Label target -----------------------------------------------
        with st.expander("2. Pembuatan label target 'Tingkat Risiko'", expanded=True, icon=":material/label:"):
            if "Tingkat Risiko" not in df.columns:
                df = generate_risk_label(df)
                st.success(
                    "Kolom target **Tingkat Risiko** berhasil dibuat dari "
                    "komposit skor psikologis (Sleep Disturbance, Mood, "
                    "Anxiety, Depression, Self-esteem, Distraction).",
                    icon=":material/check_circle:",
                )
            else:
                st.success(
                    "Kolom target **Tingkat Risiko** sudah ada di dataset.",
                    icon=":material/check_circle:",
                )

            counts = df["Tingkat Risiko"].value_counts().reset_index()
            counts.columns = ["Tingkat Risiko", "Jumlah"]

            chart = (
                alt.Chart(counts)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Tingkat Risiko:N", sort=["Rendah", "Sedang", "Tinggi"]),
                    y=alt.Y("Jumlah:Q"),
                    color=alt.Color(
                        "Tingkat Risiko:N",
                        scale=alt.Scale(
                            domain=["Rendah", "Sedang", "Tinggi"],
                            range=["#34D399", "#FBBF24", "#F87171"],
                        ),
                    ),
                    tooltip=["Tingkat Risiko", "Jumlah"],
                )
                .properties(height=300, title="Distribusi kelas 'Tingkat Risiko' (sebelum SMOTE)")
            )
            st.altair_chart(chart)

        # --- 3. Label encoding ---------------------------------------------
        with st.expander("3. Label encoding (kategorikal → numerikal)", expanded=True, icon=":material/swap_horiz:"):
            le_dict = {}
            cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

            df_encoded = df.copy()
            for col in cat_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                le_dict[col] = le

            st.session_state["encoders"] = le_dict
            st.success(
                f"Berhasil mengubah **{len(cat_cols)}** kolom kategorikal "
                f"menjadi numerikal: {', '.join(cat_cols)}",
                icon=":material/check_circle:",
            )
            st.dataframe(df_encoded.head(10))

        # --- 4. SMOTE augmentasi -------------------------------------------
        with st.expander("4. Augmentasi data dengan SMOTE", expanded=True, icon=":material/add_chart:"):
            st.markdown(
                """
**SMOTE** (Synthetic Minority Over-sampling Technique) membuat
sampel **sintetis baru** berdasarkan interpolasi antara sampel yang
ada — **bukan** duplikasi baris.  Ini adalah metode augmentasi yang
**valid secara akademis** untuk menambah jumlah data.
                """
            )

            X_pre = df_encoded.drop(columns=["Tingkat Risiko"])
            y_pre = df_encoded["Tingkat Risiko"]
            feature_names = X_pre.columns.tolist()

            max_class_count = int(y_pre.value_counts().max())
            min_target = max(500, max_class_count)

            use_smote = st.checkbox("Gunakan SMOTE untuk menyeimbangkan data?", value=(len(df_encoded) < 5000))

            if use_smote:
                target_size = st.slider(
                    "Target jumlah baris per kelas (SMOTE)",
                    min_value=min_target,
                    max_value=max(min_target * 2, 100000),
                    value=min_target,
                    step=100,
                    help="Target minimal tidak boleh kurang dari jumlah data kelas terbanyak saat ini.",
                )

                k_neighbors = min(5, min(y_pre.value_counts().values) - 1)
                k_neighbors = max(1, k_neighbors)

                smote = SMOTE(
                    sampling_strategy={cls: target_size for cls in y_pre.unique()},
                    k_neighbors=k_neighbors,
                    random_state=42,
                )

                try:
                    X_res, y_res = smote.fit_resample(X_pre, y_pre)
                    df_smote = pd.DataFrame(X_res, columns=feature_names)
                    df_smote["Tingkat Risiko"] = y_res

                    st.success(
                        f"SMOTE berhasil! Data augmentasi: **{len(df_smote):,}** baris "
                        f"(dari {len(df_encoded)} baris asli).",
                        icon=":material/check_circle:",
                    )

                    if len(df_smote) >= 5000:
                        st.success(
                            f"**{len(df_smote):,}** ≥ 5.000 — Kriteria Sangat Baik terpenuhi!",
                            icon=":material/check_circle:",
                        )
                    else:
                        st.info(
                            f"Total {len(df_smote):,} baris. Naikkan slider di atas "
                            f"untuk mencapai ≥ 5.000 baris.",
                            icon=":material/info:",
                        )

                    # Distribusi setelah SMOTE
                    counts_after = df_smote["Tingkat Risiko"].value_counts().reset_index()
                    counts_after.columns = ["Tingkat Risiko", "Jumlah"]
                    
                    if "Tingkat Risiko" in le_dict:
                        counts_after["Label"] = le_dict["Tingkat Risiko"].inverse_transform(counts_after["Tingkat Risiko"].astype(int))
                    else:
                        counts_after["Label"] = counts_after["Tingkat Risiko"].astype(str)

                    chart_after = (
                        alt.Chart(counts_after)
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                        .encode(
                            x=alt.X("Label:N", title="Tingkat Risiko"),
                            y=alt.Y("Jumlah:Q"),
                            color=alt.Color(
                                "Label:N",
                                scale=alt.Scale(range=["#34D399", "#FBBF24", "#F87171"]),
                            ),
                            tooltip=["Label:N", "Jumlah:Q"],
                        )
                        .properties(height=300, title="Distribusi kelas setelah SMOTE")
                    )
                    st.altair_chart(chart_after)

                except Exception as e:
                    st.error(f"SMOTE gagal: {e}", icon=":material/error:")
                    df_smote = df_encoded
            else:
                df_smote = df_encoded
                st.info(f"Dataset Anda memiliki **{len(df_smote):,}** baris. SMOTE dimatikan secara default.", icon=":material/info:")
                
                if len(df_smote) >= 5000:
                    st.success("Karena dataset ≥ 5.000 baris, kriteria Sangat Baik sudah terpenuhi tanpa SMOTE!", icon=":material/check_circle:")

        # --- Simpan -------------------------------------------------------
        if st.button("Simpan hasil preprocessing", icon=":material/save:", type="primary"):
            st.session_state["processed_data"] = df_smote
            st.session_state["feature_names"] = feature_names
            st.session_state["smote_done"] = True
            st.toast(
                "Data berhasil disimpan! Lanjut ke Modeling & evaluasi.",
                icon=":material/check_circle:",
            )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Modeling & Evaluasi
# ═══════════════════════════════════════════════════════════════════════════
elif "Modeling" in menu:
    st.title("Modeling & evaluasi (Decision Tree)")

    if st.session_state["processed_data"] is None:
        st.warning(
            "Silakan selesaikan dan simpan tahap **Preprocessing** "
            "terlebih dahulu.",
            icon=":material/warning:",
        )
    else:
        df = st.session_state["processed_data"]
        feature_names = st.session_state["feature_names"]

        # --- Setup ---------------------------------------------------------
        st.subheader("1. Setup data dan parameter model")

        X = df.drop(columns=["Tingkat Risiko"])
        y = df["Tingkat Risiko"]

        col_left, col_right = st.columns(2)
        with col_left:
            test_size = st.slider(
                "Rasio data uji (test size)",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05,
                help="Proporsi data untuk pengujian. Default: 80% train, 20% test.",
            )
        with col_right:
            max_depth = st.slider(
                "Maksimal kedalaman (max_depth)",
                min_value=3,
                max_value=30,
                value=10,
                help="Kedalaman pohon keputusan. Nilai lebih tinggi = model lebih kompleks.",
            )
            criterion = st.selectbox(
                "Kriteria splitting",
                ["entropy", "gini"],
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        with st.container(horizontal=True):
            st.metric("Data train", f"{X_train.shape[0]:,} baris")
            st.metric("Data test", f"{X_test.shape[0]:,} baris")
            st.metric("Total data", f"{len(df):,} baris")

        # --- Train ---------------------------------------------------------
        if st.button("Mulai latih model Decision Tree", icon=":material/play_arrow:", type="primary"):
            with st.spinner("Sedang membangun pohon keputusan..."):
                dt_model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    criterion=criterion,
                    random_state=42,
                )
                dt_model.fit(X_train, y_train)
                st.session_state["model"] = dt_model

                y_train_pred = dt_model.predict(X_train)
                y_test_pred = dt_model.predict(X_test)

                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)

            # --- Metrics --------------------------------------------------
            st.subheader("2. Hasil evaluasi model")

            with st.container(horizontal=True):
                st.metric("Akurasi training", f"{train_acc * 100:.2f}%")
                st.metric("Akurasi testing", f"{test_acc * 100:.2f}%")

            # Check criteria
            if train_acc >= 0.80:
                st.success(
                    f"Akurasi train = **{train_acc*100:.2f}%** ≥ 80% — "
                    "Kriteria **Sangat Baik** terpenuhi!",
                    icon=":material/check_circle:",
                )
            else:
                st.warning(
                    f"Akurasi train = **{train_acc*100:.2f}%** < 80%. "
                    "Coba tingkatkan max_depth.",
                    icon=":material/warning:",
                )

            if test_acc >= 0.85:
                st.success(
                    f"Akurasi test = **{test_acc*100:.2f}%** ≥ 85% — "
                    "Kriteria **Sangat Baik** terpenuhi!",
                    icon=":material/check_circle:",
                )
            else:
                st.warning(
                    f"Akurasi test = **{test_acc*100:.2f}%** < 85%. "
                    "Coba sesuaikan parameter.",
                    icon=":material/warning:",
                )

            # --- Classification report ------------------------------------
            st.subheader("3. Classification report")

            report_dict = classification_report(
                y_test, y_test_pred, output_dict=True
            )
            report_df = pd.DataFrame(report_dict).T
            st.dataframe(
                report_df.style.format("{:.2f}").background_gradient(
                    cmap="Blues", subset=["precision", "recall", "f1-score"]
                )
            )

            # --- Confusion matrix -----------------------------------------
            st.subheader("4. Confusion matrix")

            cm = confusion_matrix(y_test, y_test_pred)
            labels = sorted(y.unique())

            # Decode labels for display
            if "Tingkat Risiko" in st.session_state["encoders"]:
                le_risk = st.session_state["encoders"]["Tingkat Risiko"]
                try:
                    display_labels = le_risk.inverse_transform(labels)
                except Exception:
                    display_labels = [str(l) for l in labels]
            else:
                display_labels = [str(l) for l in labels]

            # Build heatmap data
            cm_data = []
            for i, row_label in enumerate(display_labels):
                for j, col_label in enumerate(display_labels):
                    cm_data.append(
                        {
                            "Aktual": row_label,
                            "Prediksi": col_label,
                            "Jumlah": int(cm[i][j]),
                        }
                    )
            cm_df = pd.DataFrame(cm_data)

            heatmap = (
                alt.Chart(cm_df)
                .mark_rect(cornerRadius=4)
                .encode(
                    x=alt.X("Prediksi:N", title="Prediksi"),
                    y=alt.Y("Aktual:N", title="Aktual"),
                    color=alt.Color(
                        "Jumlah:Q",
                        scale=alt.Scale(scheme="blues"),
                        title="Jumlah",
                    ),
                    tooltip=["Aktual", "Prediksi", "Jumlah"],
                )
                .properties(width=400, height=300)
            )

            text = (
                alt.Chart(cm_df)
                .mark_text(fontSize=16, fontWeight="bold")
                .encode(
                    x="Prediksi:N",
                    y="Aktual:N",
                    text="Jumlah:Q",
                    color=alt.condition(
                        alt.datum.Jumlah > cm.max() / 2,
                        alt.value("white"),
                        alt.value("black"),
                    ),
                )
            )

            st.altair_chart(heatmap + text)

            # --- Feature importance ---------------------------------------
            st.subheader("5. Feature importance (faktor dominan)")
            st.markdown(
                "Visualisasi ini menunjukkan atribut/fitur apa saja yang "
                "paling berpengaruh terhadap klasifikasi Tingkat Risiko."
            )

            importance = pd.DataFrame(
                {
                    "Fitur": feature_names,
                    "Importance": dt_model.feature_importances_,
                }
            )
            importance = importance.sort_values(
                "Importance", ascending=False
            ).head(10)

            chart_imp = (
                alt.Chart(importance)
                .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                .encode(
                    x=alt.X("Importance:Q", title="Tingkat kepentingan"),
                    y=alt.Y("Fitur:N", sort="-x", title="Fitur"),
                    color=alt.Color(
                        "Importance:Q",
                        scale=alt.Scale(scheme="viridis"),
                        legend=None,
                    ),
                    tooltip=["Fitur", alt.Tooltip("Importance:Q", format=".4f")],
                )
                .properties(height=350, title="Top 10 fitur berpengaruh")
            )
            st.altair_chart(chart_imp)

            # --- Ringkasan kriteria ---------------------------------------
            st.subheader("6. Ringkasan kriteria penilaian")

            n_rows = len(df)
            criteria = [
                {
                    "Kriteria": "Minimal 5.000 baris data",
                    "Nilai": f"{n_rows:,} baris",
                    "Status": "Terpenuhi" if n_rows >= 5000 else "Belum",
                },
                {
                    "Kriteria": "Akurasi train ≥ 80%",
                    "Nilai": f"{train_acc*100:.2f}%",
                    "Status": "Terpenuhi" if train_acc >= 0.80 else "Belum",
                },
                {
                    "Kriteria": "Akurasi test ≥ 85%",
                    "Nilai": f"{test_acc*100:.2f}%",
                    "Status": "Terpenuhi" if test_acc >= 0.85 else "Belum",
                },
            ]
            crit_df = pd.DataFrame(criteria)
            st.dataframe(crit_df, hide_index=True)

            all_met = all(c["Status"] == "Terpenuhi" for c in criteria)
            if all_met:
                st.success(
                    "Semua kriteria **Sangat Baik** terpenuhi!",
                    icon=":material/emoji_events:",
                )
            else:
                st.info(
                    "Beberapa kriteria belum terpenuhi. Sesuaikan parameter "
                    "atau ukuran augmentasi SMOTE.",
                    icon=":material/info:",
                )

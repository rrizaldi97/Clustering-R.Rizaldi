import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

import plotly.express as px
import plotly.graph_objects as go


# --------------------------------------------------
# KONFIGURASI HALAMAN & SEDIKIT STYLING
# --------------------------------------------------
st.set_page_config(
    page_title="Dashboard Clustering PMA Surabaya",
    page_icon="💹",
    layout="wide",
)

# CSS ringan: hanya untuk kartu dan judul
st.markdown(
    """
    <style>
    .app-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 1.25rem;
    }
    .card {
        border-radius: 12px;
        padding: 1rem 1.25rem;
        background-color: #11182711;
        border: 1px solid #e5e7eb22;
        margin-bottom: 0.75rem;
    }
    .card-header {
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .small-note {
        font-size: 12px;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HELPER FUNCTIONS (LOGIKA DATA)
# --------------------------------------------------
def winsorize_like_r(df, lower_q=0.05, upper_q=0.95):
    """
    Winsorizing sederhana:
    - hitung Q1, Q3, IQR
    - nilai di bawah (Q1 - 1.5*IQR) diganti quantile lower_q
    - nilai di atas (Q3 + 1.5*IQR) diganti quantile upper_q
    """
    df = df.copy()
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            continue

        x = df[col].to_numpy(dtype=float)
        q1, q3 = np.quantile(x, [0.25, 0.75])
        caps = np.quantile(x, [lower_q, upper_q])
        H = 1.5 * (q3 - q1)

        lower_bound = q1 - H
        upper_bound = q3 + H

        x = np.where(x < lower_bound, caps[0], x)
        x = np.where(x > upper_bound, caps[1], x)

        df[col] = x

    return df


def compute_elbow_silhouette(X, k_min=2, k_max=10, random_state=42):
    """
    Hitung SSE (elbow) dan Silhouette Score untuk beberapa nilai K.
    """
    sse = []
    Ks = list(range(1, k_max + 1))
    sil_scores = []

    for k in Ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        sse.append(km.inertia_)

        if k >= k_min:
            sil = silhouette_score(X, labels)
            sil_scores.append((k, sil))

    return Ks, sse, sil_scores


def add_cluster_to_df(df_raw, labels, pca_coords, negara_col=None):
    """
    Tambah kolom Cluster + koordinat PCA ke data asli.
    """
    df_result = df_raw.copy()
    df_result["Cluster"] = labels

    pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"])
    df_result = pd.concat([df_result, pca_df], axis=1)

    # urutkan kolom biar rapi
    if negara_col is not None and negara_col in df_result.columns:
        ordered = (
            [negara_col, "Cluster", "PC1", "PC2"]
            + [c for c in df_result.columns if c not in [negara_col, "Cluster", "PC1", "PC2"]]
        )
        df_result = df_result[ordered]

    return df_result


def generate_cluster_text_summary(df_profile, selected_features):
    """
    Bikin ringkasan teks profil cluster.
    """
    lines = []
    for _, row in df_profile.iterrows():
        c = row["Cluster"]
        desc_parts = [
            f"{feat} ≈ {row[feat]:,.2f}" for feat in selected_features
        ]
        lines.append(f"- **Cluster {c}**: " + "; ".join(desc_parts))
    return "\n".join(lines)


# --------------------------------------------------
# FUNGSI UI PER BAGIAN
# --------------------------------------------------
def show_header():
    st.markdown('<div class="app-title">💹 Dashboard Clustering PMA – Kota Surabaya</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Analisis pengelompokan negara investor berbasis K-Means, PCA, dan metrik evaluasi.</div>',
        unsafe_allow_html=True,
    )


def show_overview_tab(df_raw, negara_col, invest_col, proj_col):
    st.subheader("📦 Ringkasan Dataset")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Cuplikan Data**")
        st.dataframe(df_raw.head(), use_container_width=True)
        st.markdown(
            "<p class='small-note'>Lima baris pertama sebagai gambaran awal sebelum pemrosesan.</p>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("**Info singkat**")
        st.write(f"Jumlah baris: **{df_raw.shape[0]}**")
        st.write(f"Jumlah kolom: **{df_raw.shape[1]}**")
        if negara_col:
            st.write(f"Kolom negara: **{negara_col}**")

    st.markdown("---")

    # KPI
    colA, colB, colC, colD = st.columns(4)

    with colA:
        st.metric("Jumlah Negara Investor", df_raw.shape[0])

    if invest_col:
        total_inv = df_raw[invest_col].sum()
        avg_inv = df_raw[invest_col].mean()
        with colB:
            st.metric("Total Nilai Investasi", f"{total_inv:,.0f}")
        with colC:
            st.metric("Rata-rata Investasi/Negara", f"{avg_inv:,.0f}")
    else:
        with colB:
            st.metric("Total Nilai Investasi", "-")
        with colC:
            st.metric("Rata-rata Investasi/Negara", "-")

    if proj_col:
        total_proj = df_raw[proj_col].sum()
        with colD:
            st.metric("Total Jumlah Proyek", f"{int(total_proj):,}")
    else:
        with colD:
            st.metric("Total Jumlah Proyek", "-")

    # Top 10 negara
    if invest_col and negara_col:
        st.markdown("### 🏆 Top 10 Negara Berdasarkan Nilai Investasi")

        top10 = (
            df_raw[[negara_col, invest_col]]
            .sort_values(invest_col, ascending=False)
            .head(10)
        )

        fig_bar = px.bar(
            top10,
            x=negara_col,
            y=invest_col,
            text=invest_col,
            labels={negara_col: "Negara", invest_col: "Nilai Investasi"},
        )
        fig_bar.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_bar.update_layout(xaxis_tickangle=-45, title=None)
        st.plotly_chart(fig_bar, use_container_width=True)

        col_left, col_right = st.columns([1.3, 1])
        with col_left:
            st.markdown(
                """
                **Cara baca grafik:**

                1. Batang paling tinggi = negara dengan nilai investasi terbesar.  
                2. Perbandingan tinggi batang menunjukkan dominasi antarnegara.  
                """
            )

        with col_right:
            fig_pie = px.pie(
                top10,
                values=invest_col,
                names=negara_col,
                hole=0.4,
                title="Pangsa Investasi (Top 10 Negara)",
            )
            st.plotly_chart(fig_pie, use_container_width=True)


def show_preprocessing_tab(data_selected, data_winsor, data_scaled):
    st.subheader("1️⃣ EDA & Preprocessing")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Ringkasan sebelum winsorizing**")
        st.dataframe(data_selected.describe().T, use_container_width=True)

    with col2:
        st.markdown("**Boxplot sebelum winsorizing**")
        df_melt = data_selected.melt(var_name="Fitur", value_name="Nilai")
        fig = px.box(df_melt, x="Fitur", y="Nilai", points=False)
        fig.update_layout(xaxis_tickangle=-45, title=None)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Ringkasan setelah winsorizing**")
        st.dataframe(data_winsor.describe().T, use_container_width=True)

    with col4:
        st.markdown("**Boxplot setelah winsorizing**")
        df_melt2 = data_winsor.melt(var_name="Fitur", value_name="Nilai")
        fig2 = px.box(df_melt2, x="Fitur", y="Nilai", points=False)
        fig2.update_layout(xaxis_tickangle=-45, title=None)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Normalisasi Z-Score")

    df_melt3 = data_scaled.melt(var_name="Fitur", value_name="Nilai Z")
    fig3 = px.box(df_melt3, x="Fitur", y="Nilai Z", points=False)
    fig3.update_layout(xaxis_tickangle=-45, title=None)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Heatmap Korelasi (setelah normalisasi)**")
    corr = data_scaled.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
    )
    st.plotly_chart(fig_corr, use_container_width=True)


def show_pca_tab(pca, X_pca, explained_var, negara_col, df_raw, Ks, sse, sil_scores):
    st.subheader("2️⃣ PCA & Penentuan K")

    st.markdown(
        f"""
        **Rasio varian yang dijelaskan:**

        - PC1: **{explained_var[0]:.3f}**  
        - PC2: **{explained_var[1]:.3f}**  
        - Total: **{explained_var.sum():.3f}**
        """
    )

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    if negara_col and negara_col in df_raw.columns:
        pca_df[negara_col] = df_raw[negara_col]

    fig_scatter = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        hover_data=[negara_col] if negara_col else None,
        title="Proyeksi PCA (tanpa cluster)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown("### Elbow Method (SSE)")

    fig_elbow = go.Figure()
    fig_elbow.add_trace(
        go.Scatter(x=Ks, y=sse, mode="lines+markers", name="SSE")
    )
    fig_elbow.update_layout(
        xaxis_title="Jumlah Cluster (K)",
        yaxis_title="SSE",
        hovermode="x unified",
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    st.markdown("### Silhouette Score")

    if sil_scores:
        ks_sil, vals_sil = zip(*sil_scores)
        fig_sil = go.Figure()
        fig_sil.add_trace(
            go.Scatter(x=ks_sil, y=vals_sil, mode="lines+markers", name="Silhouette")
        )
        fig_sil.update_layout(
            xaxis_title="Jumlah Cluster (K)",
            yaxis_title="Silhouette Score",
            hovermode="x unified",
        )
        st.plotly_chart(fig_sil, use_container_width=True)

        best_k, best_sil = max(sil_scores, key=lambda x: x[1])
        st.info(
            f"Silhouette tertinggi pada K = **{best_k}** dengan nilai **{best_sil:.3f}**."
        )
    else:
        st.warning("Silhouette dihitung mulai K = 2. Atur ulang parameter di sidebar.")


def show_clustering_tab(
    df_clustered,
    X_pca,
    labels,
    negara_col,
    k_selected,
    selected_features,
):
    st.subheader("3️⃣ K-Means Clustering & Insight")

    sil = silhouette_score(X_pca, labels)
    dbi = davies_bouldin_score(X_pca, labels)

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Cluster (K)", k_selected)
    col2.metric("Silhouette Score", f"{sil:.3f}")
    col3.metric("Davies-Bouldin Index", f"{dbi:.3f}")

    st.markdown("### Peta PCA dengan Label Cluster")

    plot_df = df_clustered.copy()
    plot_df["Cluster"] = plot_df["Cluster"].astype(str)

    hover_cols = ["Cluster"]
    if negara_col and negara_col in plot_df.columns:
        hover_cols.append(negara_col)

    fig_cluster = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=hover_cols,
        title=None,
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("### Tabel Hasil Clustering")
    st.dataframe(df_clustered, use_container_width=True)

    st.markdown("### Profil Rata-Rata Tiap Cluster")
    df_profile = (
        df_clustered.groupby("Cluster")[selected_features].mean().reset_index()
    )
    df_profile["Cluster"] = df_profile["Cluster"].astype(str)

    long_profile = df_profile.melt(
        id_vars="Cluster", var_name="Fitur", value_name="Rata-Rata"
    )

    fig_prof = px.bar(
        long_profile,
        x="Fitur",
        y="Rata-Rata",
        color="Cluster",
        barmode="group",
        title=None,
    )
    fig_prof.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_prof, use_container_width=True)

    st.markdown("**Ringkasan singkat:**")
    st.markdown(generate_cluster_text_summary(df_profile, selected_features))


def show_prediction_tab(
    df_raw,
    selected_features,
    scaler,
    pca,
    km,
    plot_df,
):
    st.subheader("🤖 Prediksi Cluster untuk Skenario Baru")

    st.markdown(
        "Isi nilai fitur di bawah, lalu tekan tombol **Prediksi Cluster**. "
        "Model akan menjalankan pipeline yang sama dengan data awal."
    )

    with st.form("form_prediksi"):
        cols = st.columns(2)
        input_values = {}

        for i, feat in enumerate(selected_features):
            col = cols[i % 2]
            min_v = float(df_raw[feat].min())
            max_v = float(df_raw[feat].max())
            default = float(df_raw[feat].mean())
            with col:
                val = st.number_input(
                    feat,
                    value=default,
                    min_value=min_v,
                    max_value=max_v,
                )
            input_values[feat] = val

        submitted = st.form_submit_button("🔮 Prediksi Cluster")

    if not submitted:
        return

    df_input = pd.DataFrame([input_values], columns=selected_features)
    X_scaled_new = scaler.transform(df_input)
    X_pca_new = pca.transform(X_scaled_new)
    cluster_pred = int(km.predict(X_pca_new)[0])

    st.success(f"Data baru diprediksi masuk **Cluster {cluster_pred}**.")

    # gabungkan dengan data lama untuk visualisasi
    pca_new_df = pd.DataFrame(X_pca_new, columns=["PC1", "PC2"])
    pca_new_df["Cluster"] = str(cluster_pred)
    pca_new_df["Jenis"] = "Data Baru"

    pca_old_df = plot_df.copy()
    pca_old_df["Jenis"] = "Data Lama"

    comb = pd.concat(
        [pca_old_df[["PC1", "PC2", "Cluster", "Jenis"]],
         pca_new_df[["PC1", "PC2", "Cluster", "Jenis"]]]
    )

    fig_pred = px.scatter(
        comb,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=["Jenis"],
        symbol="Jenis",
        title="Posisi Data Baru di Ruang PCA",
    )
    st.plotly_chart(fig_pred, use_container_width=True)


def show_download_tab(df_clustered, selected_features, sil, dbi):
    st.subheader("📥 Download & Ringkasan")

    csv = df_clustered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Hasil Clustering (.csv)",
        data=csv,
        file_name="hasil_cluster_pma_surabaya.csv",
        mime="text/csv",
    )

    st.markdown("### Ringkasan Singkat")
    n_cluster = df_clustered["Cluster"].nunique()
    st.markdown(
        f"""
        - Jumlah negara investor: **{df_clustered.shape[0]}**  
        - Jumlah fitur yang dipakai: **{len(selected_features)}**  
        - Jumlah cluster (K-Means): **{n_cluster}**  
        - Silhouette Score: **{sil:.3f}**  
        - Davies-Bouldin Index: **{dbi:.3f}**  
        """
    )


# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
def main():
    show_header()

    # -------- SIDEBAR PENGATURAN --------
    st.sidebar.header("⚙️ Pengaturan")

    st.sidebar.markdown(
        "Pastikan file **`PMA_Investasi.xlsx`** berada di folder yang sama dengan `app2.py`."
    )

    winsor_lower = st.sidebar.slider("Quantile Winsor Bawah", 0.0, 0.2, 0.05, 0.01)
    winsor_upper = st.sidebar.slider("Quantile Winsor Atas", 0.8, 1.0, 0.95, 0.01)

    st.sidebar.markdown("---")
    st.sidebar.subheader("K-Means & PCA")

    k_max_elbow = st.sidebar.slider("Maksimum K (Elbow & Silhouette)", 4, 12, 8, 1)
    k_selected = st.sidebar.slider("Jumlah Cluster (K-Means)", 2, 8, 2, 1)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)

    # -------- LOAD DATA --------
    data_path = "PMA_Investasi.xlsx"
    if not os.path.exists(data_path):
        st.error(f"File `{data_path}` tidak ditemukan.")
        st.stop()

    df_raw = pd.read_excel(data_path)

    # deteksi beberapa kolom penting
    negara_col = next((c for c in df_raw.columns if c.lower().startswith("negara")), None)
    invest_col = next((c for c in df_raw.columns if "invest" in c.lower()), None)
    proj_col = next((c for c in df_raw.columns if "proyek" in c.lower() or "project" in c.lower()), None)

    numeric_cols = [
        c for c in df_raw.columns if np.issubdtype(df_raw[c].dtype, np.number)
    ]

    st.markdown("### Pilih Fitur Numerik untuk Clustering")
    selected_features = st.multiselect(
        "Minimal 2 fitur:",
        options=numeric_cols,
        default=numeric_cols,
    )
    if len(selected_features) < 2:
        st.warning("Pilih minimal 2 fitur numerik.")
        st.stop()

    data_selected = df_raw[selected_features].copy()

    # -------- PIPELINE: PREPROCESSING & MODELING --------
    data_winsor = winsorize_like_r(data_selected, winsor_lower, winsor_upper)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_winsor)
    data_scaled = pd.DataFrame(X_scaled, columns=selected_features)

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_

    Ks, sse, sil_scores = compute_elbow_silhouette(
        X_pca, k_min=2, k_max=k_max_elbow, random_state=random_state
    )

    km = KMeans(n_clusters=k_selected, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_pca)
    df_clustered = add_cluster_to_df(df_raw, labels, X_pca, negara_col=negara_col)

    sil = silhouette_score(X_pca, labels)
    dbi = davies_bouldin_score(X_pca, labels)

    # df ini dipakai juga di tab prediksi
    plot_df = df_clustered[["PC1", "PC2", "Cluster"]].copy()
    plot_df["Cluster"] = plot_df["Cluster"].astype(str)

    # -------- TABS UI --------
    tab_overview, tab_pre, tab_pca, tab_cluster, tab_pred, tab_dl = st.tabs(
        [
            "🏠 Overview",
            "📊 Preprocessing",
            "🧬 PCA & K",
            "🧩 Clustering",
            "🤖 Prediksi",
            "📥 Download",
        ]
    )

    with tab_overview:
        show_overview_tab(df_raw, negara_col, invest_col, proj_col)

    with tab_pre:
        show_preprocessing_tab(data_selected, data_winsor, data_scaled)

    with tab_pca:
        show_pca_tab(pca, X_pca, explained_var, negara_col, df_raw, Ks, sse, sil_scores)

    with tab_cluster:
        show_clustering_tab(
            df_clustered, X_pca, labels, negara_col, k_selected, selected_features
        )

    with tab_pred:
        show_prediction_tab(
            df_raw, selected_features, scaler, pca, km, plot_df
        )

    with tab_dl:
        show_download_tab(df_clustered, selected_features, sil, dbi)


if __name__ == "__main__":
    main()

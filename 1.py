import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt
import warnings

# Configuración de warnings
warnings.filterwarnings('ignore')

# Configuración de la app
st.set_page_config(page_title="K-Means Clustering Avanzado", layout="wide")
st.title("🎯 Clustering Interactivo con K-Means y PCA")
st.write("""
Sube tus datos, aplica **K-Means**, y observa cómo el algoritmo agrupa los puntos en un espacio reducido con **PCA (2D o 3D)**.  
También puedes comparar la distribución **antes y después** del clustering.
""")

# --- Subir archivo ---
st.sidebar.header("📁 Subir datos")
uploaded_file = st.sidebar.file_uploader("Selecciona tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Archivo cargado correctamente.")
    
    # Mostrar información del dataset
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Número de filas", data.shape[0])
    with col2:
        st.metric("Número de columnas", data.shape[1])
    with col3:
        st.metric("Columnas numéricas", len(data.select_dtypes(include=['float64', 'int64']).columns))
    
    st.write("### Vista previa de los datos:")
    st.dataframe(data.head())

    # Filtrar columnas numéricas
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("⚠️ El archivo debe contener al menos dos columnas numéricas.")
    else:
        st.sidebar.header("⚙️ Configuración del modelo")

        # Seleccionar columnas a usar
        selected_cols = st.sidebar.multiselect(
            "Selecciona las columnas numéricas para el clustering:",
            numeric_cols,
            default=numeric_cols
        )

        # --- Parámetros de K-Means ---
        st.sidebar.subheader("Parámetros de K-Means")
        
        k = st.sidebar.slider("Número de clusters (k):", 2, 15, 3)
        
        init_method = st.sidebar.selectbox(
            "Método de inicialización (init):",
            ["k-means++", "random"],
            index=0,
            help="k-means++: selección inteligente de centroides iniciales (recomendado)"
        )
        
        max_iter = st.sidebar.slider(
            "Máximo de iteraciones (max_iter):",
            min_value=100,
            max_value=1000,
            value=300,
            step=50
        )
        
        n_init = st.sidebar.slider(
            "Número de inicializaciones (n_init):",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            help="Número de veces que se ejecutará K-Means con diferentes semillas"
        )
        
        random_state = st.sidebar.number_input(
            "Semilla aleatoria (random_state):",
            min_value=0,
            max_value=1000,
            value=42,
            step=1,
            help="0 para completamente aleatorio"
        )

        # Normalización de datos
        normalize = st.sidebar.checkbox("Normalizar datos", value=True, 
                                      help="Recomendado cuando las variables tienen diferentes escalas")

        # Configuración de visualización PCA
        n_components = st.sidebar.radio("Dimensión de visualización PCA:", [2, 3], index=0)

        # --- Procesamiento de datos ---
        X = data[selected_cols]
        
        # Normalizar datos si se solicita
        if normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_processed = X_scaled
        else:
            X_processed = X.values

        # --- Modelo K-Means ---
        kmeans = KMeans(
            n_clusters=k,
            init=init_method,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state if random_state != 0 else None
        )
        kmeans.fit(X_processed)
        data['Cluster'] = kmeans.labels_

        # --- PCA ---
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_processed)
        pca_cols = [f'PCA{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        pca_df['Cluster'] = data['Cluster']
        
        # Calcular varianza explicada
        explained_variance = pca.explained_variance_ratio_
        
        # --- Métricas de evaluación ---
        inertia = kmeans.inertia_
        try:
            silhouette_avg = silhouette_score(X_processed, kmeans.labels_)
        except:
            silhouette_avg = "No calculable (k=1)"

        # --- Visualización antes del clustering ---
        st.subheader("📊 Distribución original (antes de K-Means)")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if n_components == 2:
                fig_before = px.scatter(
                    pca_df,
                    x='PCA1',
                    y='PCA2',
                    title=f"Datos originales proyectados con PCA (Varianza explicada: {explained_variance[0]:.1%} + {explained_variance[1]:.1%} = {sum(explained_variance):.1%})",
                    color_discrete_sequence=["gray"],
                    opacity=0.7
                )
            else:
                fig_before = px.scatter_3d(
                    pca_df,
                    x='PCA1',
                    y='PCA2',
                    z='PCA3',
                    title=f"Datos originales proyectados con PCA (Varianza explicada: {sum(explained_variance):.1%})",
                    color_discrete_sequence=["gray"],
                    opacity=0.7
                )
            st.plotly_chart(fig_before, use_container_width=True)
        
        with col2:
            st.write("**Varianza explicada por componente:**")
            for i, var in enumerate(explained_variance):
                st.write(f"PCA{i+1}: {var:.1%}")

        # --- Visualización después del clustering ---
        st.subheader(f"🎯 Datos agrupados con K-Means (k = {k})")
        
        if n_components == 2:
            fig_after = px.scatter(
                pca_df,
                x='PCA1',
                y='PCA2',
                color=pca_df['Cluster'].astype(str),
                title="Clusters visualizados en 2D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid,
                hover_data={col: False for col in pca_df.columns},
                opacity=0.8
            )
        else:
            fig_after = px.scatter_3d(
                pca_df,
                x='PCA1',
                y='PCA2',
                z='PCA3',
                color=pca_df['Cluster'].astype(str),
                title="Clusters visualizados en 3D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid,
                hover_data={col: False for col in pca_df.columns},
                opacity=0.8
            )
        st.plotly_chart(fig_after, use_container_width=True)

        # --- Métricas del modelo ---
        st.subheader("📈 Métricas de evaluación del clustering")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Inercia (SSE)", f"{inertia:,.2f}")
        with col2:
            st.metric("Iteraciones realizadas", kmeans.n_iter_)
        with col3:
            if isinstance(silhouette_avg, str):
                st.metric("Silhouette Score", silhouette_avg)
            else:
                st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
        with col4:
            st.metric("Número de clusters", k)

        # --- Centroides ---
        st.subheader("🔍 Centroides de los clusters")
        
        # Centroides en espacio original
        if normalize:
            centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)
        else:
            centroides_original = kmeans.cluster_centers_
            
        centroides_df = pd.DataFrame(
            centroides_original, 
            columns=selected_cols
        )
        centroides_df.index.name = 'Cluster'
        centroides_df.reset_index(inplace=True)
        centroides_df['Cluster'] = centroides_df['Cluster'].astype(str)
        
        st.dataframe(centroides_df.set_index('Cluster').style.format("{:.2f}"))

        # --- Método del Codo y Silhouette ---
        st.subheader("📊 Análisis del número óptimo de clusters")
        
        if st.button("Calcular métricas para diferentes valores de k"):
            with st.spinner("Calculando métricas..."):
                inertias = []
                silhouette_scores = []
                K_range = range(2, 11)
                
                for i in K_range:
                    km = KMeans(
                        n_clusters=i,
                        init=init_method,
                        max_iter=max_iter,
                        n_init=n_init,
                        random_state=random_state if random_state != 0 else None
                    )
                    km.fit(X_processed)
                    inertias.append(km.inertia_)
                    if i > 1:  silhouette_scores.append(silhouette_score(X_processed, km.labels_))
                
                # Gráfico de Método del Codo
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
                ax1.set_title('Método del Codo')
                ax1.set_xlabel('Número de Clusters (k)')
                ax1.set_ylabel('Inercia (SSE)')
                ax1.grid(True, alpha=0.3)
                
                # Gráfico de Silhouette Score
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.plot(range(2, 11), silhouette_scores, 'ro-', linewidth=2, markersize=8)
                ax2.set_title('Silhouette Score')
                ax2.set_xlabel('Número de Clusters (k)')
                ax2.set_ylabel('Silhouette Score')
                ax2.grid(True, alpha=0.3)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig1)
                with col2:
                    st.pyplot(fig2)

        # --- Descarga de resultados ---
        st.subheader("💾 Descargar resultados")
        
        # Preparar datos para descarga
        download_data = data.copy()
        if 'PCA1' in pca_df.columns:
            download_data['PCA1'] = pca_df['PCA1']
            download_data['PCA2'] = pca_df['PCA2']
            if 'PCA3' in pca_df.columns:
                download_data['PCA3'] = pca_df['PCA3']
        
        buffer = BytesIO()
        download_data.to_csv(buffer, index=False)
        buffer.seek(0)
        
        st.download_button(
            label="📥 Descargar CSV con Clusters",
            data=buffer,
            file_name="datos_clusterizados.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Carga un archivo CSV en la barra lateral para comenzar.")
    st.write("""
    **Ejemplo de formato esperado:**
    | usuario | op | co | ex | ag | ne | wordcount | categoria |
    |---------|----|----|----|----|----|-----------|-----------|
    | usuario1 | 34.29 | 28.14 | 41.94 | 29.37 | 9.84 | 37.09 | 7 |
    | usuario2 | 44.98 | 20.52 | 37.93 | 24.27 | 10.36 | 78.79 | 7 |
    
    **Nota:** El archivo debe contener al menos 2 columnas numéricas.
    """)
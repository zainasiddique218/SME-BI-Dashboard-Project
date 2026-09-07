import streamlit as st

from src.kpi_calculator import customer_value_table
from src.segmentation import build_rfm_table, evaluate_cluster_counts, segment_customers
from src.streamlit_helpers import configure_page, get_dashboard_data, show_data_quality, sidebar_filters, stop_if_empty
from src.utils import format_currency, format_number
from src.visualizations import cluster_diagnostic_chart, customer_scatter, rfm_segment_scatter, segment_bar


configure_page("Customer Analytics")

prepared_all, clean_sales, quality = get_dashboard_data()
filtered = sidebar_filters(clean_sales)
show_data_quality(quality)
stop_if_empty(filtered)

st.title("Customer Analytics")
st.caption("RFM analysis, customer segmentation, high-value customers, and purchase frequency.")

customer_df = customer_value_table(filtered)
rfm = build_rfm_table(filtered)

if rfm.empty:
    st.warning("No valid customer IDs are available for RFM analysis in the selected filter range.")
    st.stop()

if len(rfm) < 2:
    st.warning("At least two customers are required for segmentation in the selected filter range.")
    st.stop()

diagnostics = evaluate_cluster_counts(rfm, minimum=2, maximum=6)
recommended_clusters = int(
    diagnostics.loc[diagnostics["SilhouetteScore"].idxmax(), "Clusters"]
) if not diagnostics.empty else 3
n_clusters = st.sidebar.slider(
    "Number of customer clusters",
    min_value=2,
    max_value=6,
    value=recommended_clusters,
    help="The default is the candidate with the highest silhouette score for the current filter selection.",
)
result = segment_customers(rfm, n_clusters=n_clusters)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers analysed", format_number(len(result.rfm)))
col2.metric("Segments", format_number(result.metrics["n_clusters"]))
col3.metric("Top customer revenue", format_currency(customer_df["Revenue"].max()))
silhouette = result.metrics["silhouette_score"]
col4.metric("Silhouette score", "N/A" if silhouette is None else f"{silhouette:.3f}")

left, right = st.columns(2)
with left:
    st.plotly_chart(rfm_segment_scatter(result.rfm), width="stretch")
with right:
    st.plotly_chart(segment_bar(result.summary), width="stretch")

st.subheader("Segment summary")
st.dataframe(result.summary, width="stretch")

if not diagnostics.empty:
    st.plotly_chart(cluster_diagnostic_chart(diagnostics), width="stretch")
    st.caption(
        f"Recommended cluster count for this selection: {recommended_clusters}. "
        "Higher silhouette values indicate better-separated clusters; business interpretability is also considered."
    )

left, right = st.columns(2)
with left:
    st.subheader("High-value customers")
    st.dataframe(customer_df.head(20), width="stretch")
with right:
    st.subheader("Low-value / low-frequency customers")
    low_value = result.rfm.sort_values(["Monetary", "Frequency"], ascending=[True, True]).head(20)
    st.dataframe(low_value, width="stretch")

st.plotly_chart(customer_scatter(customer_df.head(500)), width="stretch")

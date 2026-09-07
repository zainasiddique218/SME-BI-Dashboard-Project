# SME Business Intelligence Decision Support Dashboard

Final year research project implementation for:

**Design and Development of a Business Intelligence-Based Decision Support Dashboard for Strategic Decision-Making in Small and Medium Enterprises**

The system uses the public UCI Online Retail dataset as a representative retail transaction dataset. It does not use private Sri Lankan SME data.

This folder contains the Streamlit application only. The self-contained analytics and training notebook is kept separately at `../01_COLAB_ANALYTICS/BI_Dashboard_Final_Analytics_Colab.ipynb`; that notebook does not generate this application code.

## Final System Features

- Executive KPIs: revenue, orders, customers, average order value, quantity, products, and markets
- Date and country filters shared across dashboard modules
- Sales, country, customer, product, and cancellation analytics
- RFM customer analysis and K-Means segmentation
- Cluster-count comparison using silhouette score and inertia
- Merchandise-aware product rankings that exclude postage and adjustment codes by default
- Product action matrix and slow-moving product recommendations
- Transparent rule-based decision-support alerts
- Revenue and quantity forecasting with three model comparisons
- Chronological holdout evaluation using MAE, RMSE, and MAPE
- Partial-month detection so incomplete December 2011 data is excluded from model evaluation and decline alerts
- Filtered transaction and decision-insight downloads
- Automated tests and reproducible research-result exports

## Repository Structure

```text
app.py
pages/
src/
scripts/
tests/
data/processed/
models/
.streamlit/config.toml
.github/workflows/tests.yml
requirements.txt
README.md
```

## Local Run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in a browser.

The final repository includes `data/processed/prepared_online_retail.csv.gz`, a compressed deployment bundle. The large cleaned CSV and raw Excel file are not required when running the deployed dashboard.

## Rebuild Data and Models

Place `Online Retail.xlsx` in `data/raw/`, then run:

```powershell
python scripts/prepare_data.py
python scripts/train_models.py
python scripts/export_research_results.py
python scripts/benchmark_pipeline.py
```

Generated research evidence is written to `output/research_results/`.

## Automated Tests

```powershell
python -m unittest discover -s tests -v
```

The tests use synthetic transactions, so CI does not need the full UCI dataset.

## Google Colab

Open `../01_COLAB_ANALYTICS/BI_Dashboard_Final_Analytics_Colab.ipynb`. It contains the complete analytical workflow in one notebook and can:

1. Install dependencies.
2. Upload `Online Retail.xlsx`.
3. Prepare the compressed deployment dataset.
4. Train segmentation and forecasting artefacts.
5. Compare cluster counts and forecast models.
6. Run eight final validation checks.
7. Export dissertation evidence and a results ZIP.

It deliberately does not create `app.py` or any Streamlit source file. Copy its generated `data/processed` and `models` outputs into this application folder after the final Colab run.

## GitHub Upload

Commit this application source, the small model artefacts, and `data/processed/prepared_online_retail.csv.gz`. Keep the Colab notebook in the academic submission folder or repository documentation area rather than making the deployed application depend on it.

Do not commit:

- `data/processed/cleaned_online_retail.csv` because it is approximately 81 MB.
- `Online Retail.xlsx` because the compressed prepared bundle is sufficient for deployment.
- `.venv/`, temporary files, or local secrets.

The included GitHub Actions workflow runs the automated analytics tests on pushes and pull requests.

## Streamlit Community Cloud

1. Push the final project files to a GitHub repository.
2. Confirm `data/processed/prepared_online_retail.csv.gz` is present.
3. Sign in to Streamlit Community Cloud.
4. Create an app from the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy and wait for the health check to pass.
7. Record the deployment URL and capture final screenshots for the dissertation placeholders.

No API key, paid service, or secrets file is required.

## Verified Full-Dataset Results

- Raw rows: 541,909
- Valid positive-sales rows: 530,104
- Revenue: GBP 10,666,684.54
- Orders: 19,960
- Customers: 4,338
- Complete calendar months used for forecasting: 12
- Partial months excluded: 1
- Recommended customer clusters: 3
- Recommended-cluster silhouette score: 0.416
- Automated analytics tests: 8 passed

Forecast errors are deliberately reported rather than hidden. The simple models are baseline decision-support tools, and the short 12-month complete history limits predictive reliability.

## Ethical and Analytical Boundaries

- Customer IDs are treated as anonymous analytical keys.
- No private SME or Sri Lankan customer data is used.
- The dataset represents a UK-based online retailer and should not be claimed as statistically representative of all SMEs.
- The annualized customer value field is a behavioural indicator, not a probabilistic lifetime-value estimate.
- Dashboard recommendations are transparent rules and require managerial judgement.
- Forecasts are baseline estimates and should not be treated as guaranteed future performance.

## Dataset Citation

D. Chen, "Online Retail," UCI Machine Learning Repository, 2015, doi: 10.24432/C5BW33.

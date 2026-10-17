# Frac Hyperion V2

Streamlit application for probabilistic hydraulic-fracture screening.

## New in V2
- Hyperion-style probability-cloud plots with P90 (red), P50 (black), P10 (purple)
- Variable **well-count uncertainty** (Low / Most likely / High)
- **IOR per well** scatter plot
- **EOR per well** scatter plot
- **Well-count distribution** histogram with P10 / P50 / P90 markers
- **Display-unit selection**: Field or Metric
- Field outputs support **bopd or mbd** for rates, and **stb or mmstb** for cumulative volumes
- Metric outputs support **m³/d or km³/d** for rates, and **m³ or Mm³** for cumulative volumes

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Cloud
1. Create a GitHub repository.
2. Upload all files from this folder to the repository root.
3. In Streamlit Community Cloud, select the repo and set `streamlit_app.py` as the main file.
4. Deploy.

## Important limitations
This is a **screening model**. It is not a commercial fracture simulator and not a reserves-booking workflow.

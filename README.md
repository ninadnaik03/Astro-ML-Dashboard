Astro ML Dashboard

This project is an end-to-end astronomy data analysis and machine learning dashboard designed to explore large photometric datasets from sky surveys. The system enables users to upload astronomical CSV files and perform interactive visualization, classification, and uncertainty analysis of celestial objects.

The dashboard classifies objects into STAR, GALAXY, and QSO classes using physically motivated photometric color indices (u−g, g−r, r−i, i−z). Beyond standard classification, the project explicitly models prediction confidence, classification ambiguity, and out-of-distribution (OOD) objects, making it suitable for exploratory scientific analysis rather than simple label assignment.

Key features include:
Automatic ingestion of large CSV datasets (100k+ objects)
Photometric color index computation from magnitude data
Machine-learning based object classification
Confidence-aware and ambiguity-aware predictions
Detection of out-of-distribution objects
Interactive 2D and 3D color-space visualizations

This project is intended as a research-oriented ML tool, demonstrating how uncertainty-aware models can be applied to real astronomical survey data.
Tech Stack-
Python, Flask
Pandas, NumPy
Scikit-learn (Random Forest)
Matplotlib, Plotly
Deployed using Gunicorn

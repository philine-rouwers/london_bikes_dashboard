# London Bikes: Demand Lab — FINAL

This is the final dashboard package with the modelling team's real
`model_coefficients.csv`.

## Final model support

The dashboard supports the complete exported model, including:

- `tempmax`
- `windspeed`
- `solarradiation`
- `sealevelpressure`
- `visibility`
- `precip`
- day-of-week effects
- season effects
- weekend effects
- `visibility × season` interactions
- `precipitation × weekend` interaction

The prediction engine evaluates the coefficient terms generically, so the
categorical and interaction coefficients are not ignored.

## Open-Meteo

`open_meteo.py` now retrieves the exact numeric weather inputs used by the
final model. Hourly Open-Meteo data are aggregated to daily values:

- maximum temperature → °C
- mean wind speed → km/h
- mean shortwave radiation → W/m²
- mean sea-level pressure → hPa
- mean visibility → converted from metres to km
- precipitation → daily sum in mm

Season and weekend are derived from the prediction date.

## Predict tab

The final Predict tab includes:

- final-model summary
- R², MAE and RMSE when the historical dataset is available
- actual-vs-predicted credibility chart
- historical predictions for 1–7 January 2026
- next-five-day predictions
- what-if simulator using all final numeric predictors
- day-of-week selector
- season selector
- automatic weekend logic
- interaction-term contribution chart

## Run locally

```bash
uv sync
uv run python app.py
```

Then open:

http://127.0.0.1:8050

## Render

Build:

```bash
pip install uv && uv sync
```

Start:

```bash
uv run gunicorn app:server
```

## Submission reminder

Keep these together in the repository:

- `app.py`
- `open_meteo.py`
- `model_coefficients.csv`
- `pyproject.toml`
- `render.yaml`
- `assets/styles.css`
- `uv.lock` after running `uv sync`


## Final coefficient update v2

This package uses the second final coefficient file supplied by the modelling team.
The model term structure is unchanged, so the dashboard logic remains compatible.


## FINAL v3 Predict-tab fixes

- Fixed the What-if chart so it cannot grow indefinitely.
- Replaced the long vertical contribution plot with a fixed 500px horizontal ranking.
- Shows the Intercept plus the 10 largest active effects only.
- Removed recursive full-height sizing from both simulator cards.
- Day and Season selectors are explicitly dark with white text.
- Slider values are shown inline with units and white tooltip overlays are suppressed.
- Added instructions explaining positive vs negative contribution bars.


## FINAL v4 — exact fitted model correction

This version fixes the earlier dashboard model-fit discrepancy by matching the
submitted modelling notebook exactly.

### Exact training setup

The notebook fits the model only on the modern complete period:

- `date >= 2014-01-01`
- weekend is derived from `wday`
- weekday uses the notebook's Monday–Sunday category ordering
- the original `season_name` column is preserved

### Exact best model

`bikes_hired ~ tempmax + C(day_of_week) + windspeed + sealevelpressure + solarradiation + precip*C(weekend) + visibility*C(season_name)`

The notebook reports:

- Adjusted R²: **0.643**
- Residual SE: **5,488.3**

The dashboard now calculates fitted-model R², adjusted R² and residual SE by
applying the exact exported coefficients to the exact 2014+ training sample,
with **no clipping and no rounding**.

Clipping predictions at zero is used only for future/historical demand forecasts
shown to planners, never for model-fit statistics.


## FINAL v5 — naming, order and future-day forecast

- Tabs now appear as: Explore → Relationships → Prediction Model → Natural Cycles Playground.
- `Predict` renamed to **Prediction Model**.
- `Sun & moon` renamed to **Natural Cycles Playground** and moved to the final tab.
- Live forecast now requests six calendar days from Open-Meteo, excludes today,
  then keeps exactly the next five future dates.


## FINAL v6 — flat GitHub repository

This version requires no folders in the GitHub repository.

The dashboard CSS from `assets/styles.css` has been embedded directly into
`app.py`, so all files can be uploaded at the repository root.

Upload these top-level files directly to GitHub:

- app.py
- open_meteo.py
- model_coefficients.csv
- pyproject.toml
- render.yaml
- README.md
- .gitignore
- uv.lock (if present)

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import ALL, Dash, Input, Output, State, dcc, html, dash_table, no_update
from plotly.subplots import make_subplots

from open_meteo import open_meteo, open_meteo_history


BASE_DIR = Path(__file__).resolve().parent
DATA_URL = "https://raw.githubusercontent.com/kostis-christodoulou/am01-code-sep2026/main/data/london_bikes.csv"
COEFFICIENTS_FILE = BASE_DIR / "model_coefficients.csv"

# Exact modelling setup from the submitted group notebook.
MODEL_FORMULA = (
    "bikes_hired ~ tempmax + C(day_of_week) + windspeed + "
    "sealevelpressure + solarradiation + precip*C(weekend) + "
    "visibility*C(season_name)"
)
MODEL_TRAINING_START = pd.Timestamp("2014-01-01")
# These rows are export conveniences for baseline categories; Statsmodels does
# not estimate a separate parameter for them.
BASELINE_EXPORT_TERMS = {
    "day_Mon",
    "season_Autumn",
    "visibility:season_Autumn",
}
NOTEBOOK_REPORTED_ADJ_R2 = 0.643
NOTEBOOK_REPORTED_RESID_SE = 5488.3

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

VARIABLE_GROUPS = {
    "Temperature": [
        "temp", "tempmax", "tempmin",
        "feelslike", "feelslikemax", "feelslikemin",
    ],
    "Atmosphere": [
        "dew", "humidity", "sealevelpressure",
    ],
    "Rain & snow": [
        "precip", "precipprob", "precipcover", "snow", "snowdepth",
    ],
    "Wind": [
        "windspeed", "windgust", "winddir",
    ],
    "Sky & sun": [
        "cloudcover", "visibility", "solarradiation", "solarenergy", "uvindex",
    ],
}

RELATIONSHIP_GROUPS = {
    "Bikes hired": ["bikes_hired"],
    **VARIABLE_GROUPS,
}

UNITS = {
    "bikes_hired": "hires",
    "temp": "°C",
    "tempmax": "°C",
    "tempmin": "°C",
    "feelslike": "°C",
    "feelslikemax": "°C",
    "feelslikemin": "°C",
    "dew": "°C",
    "humidity": "%",
    "precip": "mm",
    "precipprob": "%",
    "precipcover": "%",
    "snow": "cm",
    "snowdepth": "cm",
    "windgust": "km/h",
    "windspeed": "km/h",
    "winddir": "°",
    "sealevelpressure": "hPa",
    "cloudcover": "%",
    "visibility": "km",
    "solarradiation": "W/m²",
    "solarenergy": "MJ/m²",
    "uvindex": "index",
    "moonphase": "0–1",
    "sunrise_hour": "hour",
    "sunset_hour": "hour",
    "daylight_hours": "hours",
}

SPECIAL_NAMES = {
    "bikes_hired": "Bikes hired",
    "predicted_bikes_hired": "Predicted bikes hired",
    "day_of_week": "Day of week",
    "month_name": "Month",
    "season_name": "Season",
    "temp": "Temperature",
    "tempmax": "Maximum temperature",
    "tempmin": "Minimum temperature",
    "feelslike": "Feels like",
    "feelslikemax": "Maximum feels like",
    "feelslikemin": "Minimum feels like",
    "dew": "Dew point",
    "humidity": "Humidity",
    "precip": "Precipitation",
    "precipprob": "Precipitation probability",
    "precipcover": "Precipitation cover",
    "snow": "Snow",
    "snowdepth": "Snow depth",
    "windgust": "Wind gust",
    "windspeed": "Wind speed",
    "winddir": "Wind direction",
    "sealevelpressure": "Sea-level pressure",
    "cloudcover": "Cloud cover",
    "visibility": "Visibility",
    "solarradiation": "Solar radiation",
    "solarenergy": "Solar energy",
    "uvindex": "UV index",
    "moonphase": "Moon phase",
    "sunrise_hour": "Sunrise",
    "sunset_hour": "Sunset",
    "daylight_hours": "Daylight hours",
}

SCENARIO_CONFIG = {
    "tempmax": {"min": -5, "max": 40, "step": 0.5, "value": 18},
    "windspeed": {"min": 0, "max": 60, "step": 1, "value": 15},
    "solarradiation": {"min": 0, "max": 350, "step": 5, "value": 120},
    "sealevelpressure": {"min": 970, "max": 1045, "step": 1, "value": 1015},
    "visibility": {"min": 1, "max": 30, "step": 0.5, "value": 16},
    "precip": {"min": 0, "max": 35, "step": 0.5, "value": 0},
}

SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
DAY_LABELS = {
    "Mon": "Monday",
    "Tue": "Tuesday",
    "Wed": "Wednesday",
    "Thu": "Thursday",
    "Fri": "Friday",
    "Sat": "Saturday",
    "Sun": "Sunday",
}

C = {
    "bg": "#06111F",
    "panel": "#0C1B2A",
    "panel2": "#10243A",
    "panel3": "#132A43",
    "line": "#203952",
    "text": "#F1F7FF",
    "muted": "#9FB3C8",
    "blue": "#58A6FF",
    "cyan": "#2DE2E6",
    "teal": "#1FD1B5",
    "purple": "#A78BFA",
    "pink": "#F472B6",
    "neon": "#80F5A6",
    "amber": "#F6C85F",
    "red": "#FF6B7A",
}

GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "doubleClick": "reset",
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["select2d", "lasso2d", "drawline", "drawrect", "eraseshape"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "london_bikes_chart",
        "scale": 2,
    },
}

BAR_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "displayModeBar": False,
}


def display_name(variable):
    if variable in SPECIAL_NAMES:
        return SPECIAL_NAMES[variable]
    return variable.replace("_", " ").strip().capitalize()


def axis_label(variable):
    name = display_name(variable)
    unit = UNITS.get(variable)
    return f"{name} ({unit})" if unit else name


def dropdown_options(values):
    return [{"label": display_name(v), "value": v} for v in values]


def card(children, class_name="dashboard-card", **kwargs):
    return dbc.Card(dbc.CardBody(children), className=class_name, **kwargs)


def empty_figure(message, height=520):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": C["muted"]},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor=C["panel"],
        plot_bgcolor=C["panel"],
    )
    return fig


def style_figure(fig, height=None, dragmode="zoom"):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["panel"],
        plot_bgcolor=C["panel"],
        font={"color": C["text"]},
        title_font={"size": 20, "color": C["text"]},
        hoverlabel={"bgcolor": C["panel3"], "font_size": 12, "font_color": C["text"]},
        margin={"l": 72, "r": 38, "t": 92, "b": 68},
        dragmode=dragmode,
        uirevision="keep",
        legend={"font": {"color": C["text"]}},
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=C["line"],
        tickfont={"color": C["muted"]},
        title_font={"color": C["text"]},
    )
    fig.update_yaxes(
        gridcolor=C["line"],
        zeroline=False,
        linecolor=C["line"],
        tickfont={"color": C["muted"]},
        title_font={"color": C["text"]},
    )
    return fig


def parse_time_hour(series):
    parsed = pd.to_datetime(series.astype(str), errors="coerce")
    hours = parsed.dt.hour + parsed.dt.minute / 60 + parsed.dt.second / 3600
    return hours


def load_bike_data():
    try:
        df = pd.read_csv(DATA_URL)
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)

        # EXACTLY mirror the modelling notebook's cleaning.
        if "wday" in df.columns:
            df["weekend"] = df["wday"].isin(["Sat", "Sun"])

        if "day_of_week" in df.columns:
            df["day_of_week"] = pd.Categorical(
                df["day_of_week"],
                categories=DAY_ORDER,
                ordered=True,
            )

        if "month_name" in df.columns:
            df["month_name"] = pd.Categorical(
                df["month_name"],
                categories=MONTH_ORDER,
                ordered=True,
            )

        # The fitted best_model uses the modern complete period only.
        df = df[df["date"] >= MODEL_TRAINING_START].copy()

        if "sunrise" in df.columns:
            df["sunrise_hour"] = parse_time_hour(df["sunrise"])
        if "sunset" in df.columns:
            df["sunset_hour"] = parse_time_hour(df["sunset"])
        if {"sunrise_hour", "sunset_hour"}.issubset(df.columns):
            df["daylight_hours"] = df["sunset_hour"] - df["sunrise_hour"]

        return df, None
    except Exception as exc:
        return pd.DataFrame(), f"Could not load the London bikes dataset: {exc}"


def load_coefficients():
    try:
        coeffs = pd.read_csv(COEFFICIENTS_FILE)
        if not {"term", "coefficient"}.issubset(coeffs.columns):
            raise ValueError("CSV must contain columns named term and coefficient.")
        coeffs["coefficient"] = pd.to_numeric(coeffs["coefficient"], errors="raise")
        return coeffs, None
    except Exception as exc:
        return pd.DataFrame(), f"Could not load model_coefficients.csv: {exc}"


def is_temporary_coefficients(coeffs):
    if coeffs.empty:
        return False
    predictor_rows = coeffs.loc[coeffs["term"] != "Intercept", "coefficient"]
    return len(predictor_rows) > 0 and (predictor_rows.abs() < 1e-12).all()


def season_from_month(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def prepare_model_frame(df):
    """Prepare data using the same category definitions as the fitted notebook model."""
    out = df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Preserve the original training-data columns when they exist.
    if "day_of_week" not in out.columns and "date" in out.columns:
        out["day_of_week"] = out["date"].dt.strftime("%a")

    if "weekend" not in out.columns:
        if "wday" in out.columns:
            out["weekend"] = out["wday"].isin(["Sat", "Sun"])
        elif "day_of_week" in out.columns:
            out["weekend"] = out["day_of_week"].astype(str).isin(["Sat", "Sun"])
        elif "date" in out.columns:
            out["weekend"] = out["date"].dt.dayofweek >= 5

    if "season_name" not in out.columns and "date" in out.columns:
        out["season_name"] = out["date"].dt.month.map(season_from_month)

    if "season_name" in out.columns:
        out["season_name"] = out["season_name"].astype(str).replace({"Fall": "Autumn"})

    if "day_of_week" in out.columns:
        out["day_of_week"] = out["day_of_week"].astype(str)

    return out


def model_atom_values(frame, atom):
    """Return one model-matrix column for a coefficient term atom."""
    if atom in frame.columns:
        return pd.to_numeric(frame[atom], errors="coerce").fillna(0.0).astype(float)

    if atom.startswith("day_"):
        level = atom.split("_", 1)[1]
        return frame["day_of_week"].eq(level).astype(float)

    if atom.startswith("season_"):
        level = atom.split("_", 1)[1]
        return frame["season_name"].eq(level).astype(float)

    if atom.startswith("weekend_"):
        level = atom.split("_", 1)[1].lower() == "true"
        return frame["weekend"].astype(bool).eq(level).astype(float)

    raise ValueError(f"Unsupported model term component: {atom}")


def model_term_values(frame, term):
    if term == "Intercept":
        return pd.Series(1.0, index=frame.index)

    pieces = str(term).split(":")
    values = pd.Series(1.0, index=frame.index)
    for piece in pieces:
        values = values * model_atom_values(frame, piece)
    return values


def model_numeric_variables(coeffs):
    """Return unique raw numeric variables referenced by the model."""
    variables = []
    for term in coeffs["term"].astype(str):
        if term == "Intercept":
            continue
        for piece in term.split(":"):
            if (
                piece.startswith("day_")
                or piece.startswith("season_")
                or piece.startswith("weekend_")
            ):
                continue
            if piece not in variables:
                variables.append(piece)
    return variables


def term_display_name(term):
    if term == "Intercept":
        return "Intercept"

    def atom_name(atom):
        if atom.startswith("day_"):
            code = atom.split("_", 1)[1]
            return f"{DAY_LABELS.get(code, code)} effect"
        if atom.startswith("season_"):
            return f"{atom.split('_', 1)[1]} effect"
        if atom.startswith("weekend_"):
            return "Weekend effect" if atom.endswith("True") else "Weekday effect"
        return display_name(atom)

    return " × ".join(atom_name(piece) for piece in str(term).split(":"))


def predict_hires(weather_df, coeffs, clip_zero=True, round_output=True):
    """Apply the exact exported fitted model, including categorical and interaction terms.

    Future demand predictions are clipped at zero because negative hires are not
    meaningful. Historical model-fit calculations call this with clip_zero=False
    and round_output=False so they reproduce the fitted linear model exactly.
    """
    if weather_df.empty:
        return weather_df.copy()

    frame = prepare_model_frame(weather_df)
    coef_map = dict(zip(coeffs["term"].astype(str), coeffs["coefficient"].astype(float)))

    if "Intercept" not in coef_map:
        raise ValueError("model_coefficients.csv is missing Intercept.")

    # Validate every raw numeric variable before calculating predictions.
    required_numeric = model_numeric_variables(coeffs)
    missing = [name for name in required_numeric if name not in frame.columns]
    if missing:
        raise ValueError(
            "Weather data is missing predictor(s) required by the final model: "
            + ", ".join(missing)
        )

    prediction = pd.Series(0.0, index=frame.index)
    for term, coefficient in coef_map.items():
        prediction += float(coefficient) * model_term_values(frame, term)

    if clip_zero:
        prediction = prediction.clip(lower=0)
    if round_output:
        prediction = prediction.round(2)

    frame["predicted_bikes_hired"] = prediction
    return frame


def model_contributions(frame, coeffs):
    """Return each coefficient's contribution for a one-row scenario frame."""
    prepared = prepare_model_frame(frame)
    rows = []

    for _, row in coeffs.iterrows():
        term = str(row["term"])
        coefficient = float(row["coefficient"])
        value = float(model_term_values(prepared, term).iloc[0])
        contribution = coefficient * value

        # Keep all active terms and the intercept. Zero inactive dummy terms add clutter.
        if term == "Intercept" or abs(contribution) > 1e-9:
            rows.append(
                {
                    "term": term_display_name(term),
                    "contribution": contribution,
                }
            )

    return pd.DataFrame(rows)


def evaluate_model_history(coeffs):
    """Reproduce the fitted best_model on the exact 2014+ training sample.

    The coefficient CSV contains the fitted Statsmodels parameters. Applying
    those exact parameters to the exact training rows gives the same fitted
    values as best_model.fittedvalues, provided no clipping or rounding is used.
    """
    if BIKE_DF.empty or coeffs.empty:
        return None

    eval_df = prepare_model_frame(BIKE_DF)

    required = ["bikes_hired"] + model_numeric_variables(coeffs)
    missing_cols = [c for c in required if c not in eval_df.columns]
    if missing_cols:
        return None

    eval_df = eval_df.dropna(subset=required).copy()
    if len(eval_df) < 20:
        return None

    try:
        pred_df = predict_hires(
            eval_df,
            coeffs,
            clip_zero=False,
            round_output=False,
        )
    except Exception:
        return None

    actual = eval_df["bikes_hired"].astype(float)
    fitted = pred_df["predicted_bikes_hired"].astype(float)

    residual = actual - fitted
    sse = float((residual ** 2).sum())
    sst = float(((actual - actual.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst else np.nan

    # The exported CSV deliberately contains zero rows for baseline categories.
    # Remove those convenience rows when counting the fitted parameters.
    effective_params = coeffs.loc[
        ~coeffs["term"].isin(BASELINE_EXPORT_TERMS),
        "term",
    ]
    p = len(effective_params)   # includes intercept
    n = len(eval_df)
    df_model = p - 1
    df_resid = n - p

    adjusted_r2 = (
        1 - (1 - r2) * (n - 1) / df_resid
        if df_resid > 0 else np.nan
    )
    residual_se = float(np.sqrt(sse / df_resid)) if df_resid > 0 else np.nan
    mae = float(np.abs(residual).mean())

    perf_df = pd.DataFrame(
        {
            "Actual": actual,
            "Fitted": fitted,
        }
    )

    return perf_df, r2, adjusted_r2, residual_se, mae, n, df_model


def metric_card(label, value, detail="", accent=None):
    accent = accent or C["blue"]
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(label, className="metric-label"),
                html.Div(value, className="metric-value"),
                html.Div(detail, className="metric-detail") if detail else None,
            ]
        ),
        className="metric-card",
        style={"borderTop": f"3px solid {accent}"},
    )


def insight_card(kicker, headline, body, accent=None, detail=None):
    accent = accent or C["cyan"]
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(kicker, className="insight-kicker", style={"color": accent}),
                html.Div(headline, className="insight-headline"),
                html.Div(body, className="insight-body"),
                html.Div(detail, className="insight-detail") if detail else None,
            ]
        ),
        className="insight-card",
        style={"borderLeft": f"4px solid {accent}"},
    )


def add_fit_line(fig, x, y):
    temp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(temp) < 3 or temp["x"].nunique() < 2:
        return fig

    slope, intercept = np.polyfit(temp["x"], temp["y"], 1)
    xs = np.linspace(temp["x"].min(), temp["x"].max(), 100)
    ys = slope * xs + intercept

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="Linear trend",
            line={"width": 3, "dash": "dash", "color": C["neon"]},
        )
    )
    return fig


def relationship_stats(x, y="bikes_hired", df=None):
    df = BIKE_DF if df is None else df
    temp = df[[x, y]].dropna()
    if len(temp) < 3 or temp[x].nunique() < 2:
        return None

    r = temp[x].corr(temp[y])
    slope, _ = np.polyfit(temp[x], temp[y], 1)
    q25 = temp[x].quantile(0.25)
    q75 = temp[x].quantile(0.75)
    low = temp.loc[temp[x] <= q25, y].mean()
    high = temp.loc[temp[x] >= q75, y].mean()
    pct = ((high - low) / low * 100) if low else np.nan

    return {
        "r": r,
        "slope": slope,
        "low_mean": low,
        "high_mean": high,
        "quartile_pct": pct,
        "n": len(temp),
    }


def relationship_strength(r):
    a = abs(r)
    if a >= 0.7:
        strength = "Strong"
    elif a >= 0.4:
        strength = "Moderate"
    elif a >= 0.2:
        strength = "Weak"
    else:
        strength = "Very weak"
    direction = "positive" if r > 0 else "negative" if r < 0 else "flat"
    return strength, direction


def dataset_metrics():
    if BIKE_DF.empty:
        return [metric_card("Dataset", "Unavailable")]

    valid = BIKE_DF.dropna(subset=["bikes_hired", "date"])
    peak = valid.loc[valid["bikes_hired"].idxmax()]
    month = (
        valid.groupby("month_name", as_index=False)["bikes_hired"]
        .mean()
        .sort_values("bikes_hired", ascending=False)
    ).iloc[0]

    return [
        metric_card(
            "Days analysed",
            f"{len(valid):,}",
            f"{valid['date'].min():%Y}–{valid['date'].max():%Y}",
            C["blue"],
        ),
        metric_card(
            "Average bikes hired",
            f"{valid['bikes_hired'].mean():,.2f}",
            "Across the cleaned historical period",
            C["teal"],
        ),
        metric_card(
            "Highest bikes hired",
            f"{peak['bikes_hired']:,.2f}",
            f"{peak['date']:%d %b %Y}",
            C["pink"],
        ),
        metric_card(
            "Highest-average month",
            str(month["month_name"]),
            f"{month['bikes_hired']:,.2f} average hires",
            C["purple"],
        ),
    ]


def scenario_controls():
    if COEFFS.empty:
        return html.Div(COEFF_ERROR or "No model coefficients available.")

    variables = model_numeric_variables(COEFFS)
    blocks = []

    for term in variables:
        if term not in SCENARIO_CONFIG:
            continue

        cfg = SCENARIO_CONFIG[term]
        blocks.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(axis_label(term), className="control-label"),
                            html.Div(
                                id={"type": "scenario-value", "term": term},
                                className="control-value",
                            ),
                        ],
                        className="d-flex justify-content-between",
                    ),
                    dcc.Slider(
                        id={"type": "scenario-slider", "term": term},
                        min=cfg["min"],
                        max=cfg["max"],
                        step=cfg["step"],
                        value=cfg["value"],
                        className="neon-slider",
                    ),
                ],
                className="mb-4",
            )
        )

    return html.Div(blocks)


def prediction_section(title, df):
    display_cols = [
        c for c in [
            "date",
            "day_of_week",
            "season_name",
            "tempmax",
            "windspeed",
            "solarradiation",
            "sealevelpressure",
            "visibility",
            "precip",
            "predicted_bikes_hired",
        ] if c in df.columns
    ]

    table_df = df[display_cols].copy()
    if "date" in table_df:
        table_df["date"] = pd.to_datetime(table_df["date"]).dt.strftime("%Y-%m-%d")

    numeric_cols = table_df.select_dtypes(include=[np.number]).columns
    table_df[numeric_cols] = table_df[numeric_cols].round(2)

    fig = px.bar(
        df,
        x="date",
        y="predicted_bikes_hired",
        text=df["predicted_bikes_hired"].map(lambda x: f"{x:,.2f}"),
        labels={
            "date": "Date",
            "predicted_bikes_hired": "Predicted bikes hired (hires)",
        },
        title=title,
    )
    fig.update_traces(
        textposition="outside",
        marker_color=C["blue"],
        hovertemplate="Date=%{x}<br>Predicted bikes hired=%{y:,.2f}<extra></extra>",
    )
    fig.update_layout(yaxis_rangemode="tozero")
    fig = style_figure(fig, 500, False)

    columns = []
    for col in table_df.columns:
        if col in numeric_cols:
            columns.append({
                "name": axis_label(col),
                "id": col,
                "type": "numeric",
                "format": {"specifier": ",.2f"},
            })
        else:
            columns.append({"name": display_name(col), "id": col})

    return card(
        [
            html.H3(title, className="section-heading"),
            dash_table.DataTable(
                data=table_df.to_dict("records"),
                columns=columns,
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "10px",
                    "fontSize": "13px",
                    "backgroundColor": C["panel"],
                    "color": C["text"],
                    "border": "none",
                    "borderBottom": f"1px solid {C['line']}",
                },
                style_header={
                    "fontWeight": "800",
                    "backgroundColor": C["panel2"],
                    "border": "none",
                    "color": C["text"],
                },
            ),
            dcc.Graph(figure=fig, config=BAR_CONFIG),
        ],
        class_name="dashboard-card mt-3",
    )


def model_credibility_content():
    perf = evaluate_model_history(COEFFS)

    if perf is None:
        return [
            insight_card(
                "Model credibility",
                "Final fitted model loaded",
                "The dashboard has the modelling team's final coefficients, but the historical dataset could not be loaded for the fitted-value check.",
                C["amber"],
            )
        ]

    perf_df, r2, adjusted_r2, residual_se, mae, n, df_model = perf

    fig = px.scatter(
        perf_df,
        x="Actual",
        y="Fitted",
        opacity=0.4,
        title="Actual vs fitted bikes hired",
        labels={
            "Actual": "Actual bikes hired",
            "Fitted": "Fitted bikes hired",
        },
    )
    fig.update_traces(
        marker={"color": C["cyan"], "size": 5},
        hovertemplate="Actual=%{x:,.0f}<br>Fitted=%{y:,.0f}<extra></extra>",
    )

    low = min(perf_df["Actual"].min(), perf_df["Fitted"].min())
    high = max(perf_df["Actual"].max(), perf_df["Fitted"].max())
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Perfect fit",
            line={"dash": "dash", "color": C["pink"]},
        )
    )
    fig = style_figure(fig, 560)

    # Small numerical tolerance because the notebook write-up rounds its
    # reported statistics.
    adj_matches = abs(adjusted_r2 - NOTEBOOK_REPORTED_ADJ_R2) < 0.002
    se_matches = abs(residual_se - NOTEBOOK_REPORTED_RESID_SE) < 2.0

    match_text = (
        "Matches the modelling notebook"
        if adj_matches and se_matches
        else "Calculated from the exact exported model on the notebook's 2014+ sample"
    )

    return [
        html.Div(
            [
                metric_card(
                    "R²",
                    f"{r2:.3f}",
                    "Fitted model on the exact training sample",
                    C["blue"],
                ),
                metric_card(
                    "Adjusted R²",
                    f"{adjusted_r2:.3f}",
                    "Notebook reports 0.643",
                    C["teal"],
                ),
                metric_card(
                    "Residual SE",
                    f"{residual_se:,.1f}",
                    "Notebook reports 5,488.3",
                    C["purple"],
                ),
                metric_card(
                    "Training days",
                    f"{n:,}",
                    "Modern complete period: 2014 onward",
                    C["amber"],
                ),
            ],
            className="metrics-grid",
        ),
        insight_card(
            "Model verification",
            match_text,
            f"Formula: {MODEL_FORMULA}",
            C["teal"] if adj_matches and se_matches else C["amber"],
            detail=f"Model degrees of freedom: {df_model}",
        ),
        card(
            [
                html.H3("Actual vs fitted values", className="section-heading"),
                html.P(
                    "This is no longer a separate dashboard approximation. It applies the modelling team's exact exported coefficients to the exact 2014+ training sample, with no clipping or rounding, reproducing the fitted linear model. Clipping at zero is used only for future demand forecasts.",
                    className="section-copy",
                ),
                dcc.Graph(figure=fig, config=GRAPH_CONFIG),
            ],
            class_name="dashboard-card mt-3",
        ),
    ]


def category_correlation_matrix():
    groups = {
        name: [v for v in values if v in BIKE_DF.columns]
        for name, values in RELATIONSHIP_GROUPS.items()
    }
    names = [name for name, values in groups.items() if values]
    z = np.full((len(names), len(names)), np.nan)
    pair_text = np.empty((len(names), len(names)), dtype=object)

    for i, row_group in enumerate(names):
        for j, col_group in enumerate(names):
            best_corr = None
            best_pair = None

            for y in groups[row_group]:
                for x in groups[col_group]:
                    pair = BIKE_DF[[x, y]].dropna()
                    if len(pair) < 3:
                        continue
                    corr = 1.0 if x == y else pair[x].corr(pair[y])
                    if pd.isna(corr):
                        continue
                    if best_corr is None or abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_pair = (x, y)

            if best_corr is not None:
                z[i, j] = best_corr
                x, y = best_pair
                pair_text[i, j] = f"{display_name(x)} ↔ {display_name(y)}"
            else:
                pair_text[i, j] = "No usable pair"

    return names, z, pair_text


def moon_glyph(phase):
    phase = float(phase) % 1.0
    phases = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]
    idx = int(np.floor((phase + 0.0625) * 8)) % 8
    return phases[idx]


def hour_text(h):
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        hh += 1
        mm = 0
    return f"{hh:02d}:{mm:02d}"


BIKE_DF, BIKE_ERROR = load_bike_data()
COEFFS, COEFF_ERROR = load_coefficients()

AVAILABLE_GROUPS = {
    group: [v for v in variables if v in BIKE_DF.columns]
    for group, variables in VARIABLE_GROUPS.items()
}
AVAILABLE_GROUPS = {g: v for g, v in AVAILABLE_GROUPS.items() if v}

AVAILABLE_REL_GROUPS = {
    group: [v for v in variables if v in BIKE_DF.columns]
    for group, variables in RELATIONSHIP_GROUPS.items()
}
AVAILABLE_REL_GROUPS = {g: v for g, v in AVAILABLE_REL_GROUPS.items() if v}

DEFAULT_GROUP = "Temperature" if "Temperature" in AVAILABLE_GROUPS else next(iter(AVAILABLE_GROUPS), None)
DEFAULT_VAR = AVAILABLE_GROUPS.get(DEFAULT_GROUP, [None])[0]

if not BIKE_DF.empty:
    min_date = BIKE_DF["date"].min().date().isoformat()
    max_date = BIKE_DF["date"].max().date().isoformat()
    data_through = BIKE_DF["date"].max().strftime("%d %b %Y")
else:
    min_date = "2014-01-01"
    max_date = "2026-01-01"
    data_through = "Unavailable"

SUNMOON_AVAILABLE = (
    not BIKE_DF.empty
    and {"moonphase", "sunrise_hour", "sunset_hour", "temp"}.issubset(BIKE_DF.columns)
    and BIKE_DF[["moonphase", "sunrise_hour", "sunset_hour", "temp"]].notna().any().all()
)

if SUNMOON_AVAILABLE:
    sunrise_min = float(np.floor(BIKE_DF["sunrise_hour"].quantile(0.01) * 2) / 2)
    sunrise_max = float(np.ceil(BIKE_DF["sunrise_hour"].quantile(0.99) * 2) / 2)
    sunset_min = float(np.floor(BIKE_DF["sunset_hour"].quantile(0.01) * 2) / 2)
    sunset_max = float(np.ceil(BIKE_DF["sunset_hour"].quantile(0.99) * 2) / 2)
    sunrise_default = float(BIKE_DF["sunrise_hour"].median())
    sunset_default = float(BIKE_DF["sunset_hour"].median())
    temp_default = float(round(BIKE_DF["temp"].median(), 1))
else:
    sunrise_min, sunrise_max = 4.0, 9.0
    sunset_min, sunset_max = 15.0, 22.0
    sunrise_default = 7.0
    sunset_default = 18.0
    temp_default = 15.0

temporary_model = is_temporary_coefficients(COEFFS)
last_rendered = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server


# CSS is embedded directly so the GitHub repository can remain completely flat.
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
:root {
  --bg: #06111F;
  --panel: #0C1B2A;
  --panel-2: #10243A;
  --panel-3: #132A43;
  --line: #203952;
  --text: #F1F7FF;
  --muted: #9FB3C8;
  --blue: #58A6FF;
  --cyan: #2DE2E6;
  --teal: #1FD1B5;
  --purple: #A78BFA;
  --pink: #F472B6;
  --neon: #80F5A6;
  --amber: #F6C85F;
}

html, body, #react-entry-point {
  background: var(--bg) !important;
  color: var(--text);
  min-height: 100%;
}

body {
  margin: 0;
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.app-shell {
  max-width: 1440px;
  margin: 0 auto;
  padding: 28px 24px 60px;
  min-height: 100vh;
  background:
    radial-gradient(circle at 90% 4%, rgba(167,139,250,.13), transparent 28%),
    radial-gradient(circle at 8% 15%, rgba(45,226,230,.08), transparent 22%),
    var(--bg);
}

.hero-panel {
  padding: 30px;
  border-radius: 20px;
  border: 1px solid rgba(88,166,255,.25);
  background:
    linear-gradient(120deg, rgba(16,36,58,.98), rgba(12,27,42,.98)),
    var(--panel);
  box-shadow: 0 18px 50px rgba(0,0,0,.28);
}

.eyebrow {
  font-size: 11px;
  letter-spacing: 1.7px;
  font-weight: 800;
  color: var(--cyan);
}

.hero-title {
  margin: 6px 0 5px;
  font-size: clamp(32px, 4vw, 44px);
  font-weight: 800;
  color: var(--text);
}

.hero-subtitle {
  margin: 0;
  color: #C5D4E5;
  max-width: 860px;
  font-size: 16px;
}

.status-badge {
  font-size: 12px;
}

.metrics-grid,
.insights-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 14px;
  margin-top: 18px;
}

.metric-card,
.insight-card,
.dashboard-card {
  background: rgba(12,27,42,.96) !important;
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  box-shadow: 0 10px 28px rgba(0,0,0,.18);
  color: var(--text);
}

.dashboard-card .card-body {
  padding: 20px;
}

.metric-card .card-body {
  padding: 16px 18px;
}

.metric-label,
.insight-kicker {
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .7px;
  text-transform: uppercase;
  color: var(--muted);
}

.metric-value {
  font-size: 27px;
  font-weight: 800;
  margin-top: 4px;
  color: var(--text);
}

.metric-detail,
.insight-detail {
  font-size: 12px;
  color: var(--muted);
  margin-top: 4px;
}

.insight-headline {
  font-size: 20px;
  line-height: 1.2;
  font-weight: 800;
  color: var(--text);
  margin: 5px 0 7px;
}

.insight-body {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.55;
}

.section-heading {
  font-size: 20px;
  font-weight: 800;
  color: var(--text);
  margin: 0 0 8px;
}

.section-copy,
.chart-help {
  color: var(--muted);
  line-height: 1.5;
  margin-bottom: 16px;
}

.chart-help {
  padding: 9px 12px;
  border-radius: 9px;
  background: rgba(88,166,255,.08);
  border: 1px solid rgba(88,166,255,.15);
  font-size: 13px;
}

.chart-card {
  overflow: visible;
}

.product-tabs {
  border: none !important;
}

.product-tabs .nav {
  gap: 8px;
  border-bottom: 1px solid var(--line);
  padding-bottom: 8px;
}

.product-tabs .nav-link {
  color: var(--muted) !important;
  border: none !important;
  border-radius: 9px !important;
  background: transparent !important;
  font-weight: 700;
  padding: 10px 15px;
}

.product-tabs .nav-link:hover {
  color: var(--text) !important;
  background: rgba(88,166,255,.08) !important;
}

.product-tabs .nav-link.active {
  color: #06111F !important;
  background: linear-gradient(90deg, var(--blue), var(--cyan)) !important;
  box-shadow: 0 5px 16px rgba(88,166,255,.22);
}

.control-label {
  display: block;
  color: var(--text);
  font-weight: 700;
  margin-bottom: 7px;
  font-size: 13px;
}

.control-value {
  color: var(--cyan);
  font-weight: 800;
}



.neon-slider .rc-slider-track,
.moon-slider .rc-slider-track,
.sun-range-slider .rc-slider-track {
  background: linear-gradient(90deg, var(--purple), var(--cyan)) !important;
}

.neon-slider .rc-slider-handle,
.moon-slider .rc-slider-handle,
.sun-range-slider .rc-slider-handle {
  border-color: var(--cyan) !important;
  background: var(--panel-2) !important;
  box-shadow: 0 0 0 4px rgba(45,226,230,.12), 0 0 18px rgba(45,226,230,.42) !important;
}

.rc-slider-rail {
  background: #263F58 !important;
}

.rc-slider-mark-text {
  color: var(--muted) !important;
}

.modebar {
  opacity: .92 !important;
}
.modebar-btn path {
  fill: #BCD0E5 !important;
}

.sunmoon-card {
  overflow: visible;
}

.sunmoon-game-grid {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(260px, .9fr) minmax(220px, .7fr);
  gap: 18px;
  margin-top: 18px;
}

.game-panel {
  min-height: 300px;
  border-radius: 16px;
  border: 1px solid var(--line);
  background: linear-gradient(145deg, #0E2035, #0A1828);
  padding: 18px;
  position: relative;
  overflow: hidden;
}

.game-label {
  font-size: 11px;
  letter-spacing: 1.2px;
  font-weight: 800;
  color: var(--cyan);
  margin-bottom: 14px;
}



.sky-track {
  height: 150px;
  border-radius: 16px;
  position: relative;
  overflow: hidden;
  padding: 0 28px;
  background:
    radial-gradient(circle at 50% 115%, rgba(246,200,95,.34), transparent 34%),
    linear-gradient(180deg, #07111F 0%, #102D4E 52%, #194A6B 100%);
  border: 1px solid rgba(88,166,255,.18);
}

.stars::before,
.stars::after {
  content: "·   ✦      ·      ✧      ·  ✦     ·       ✧";
  position: absolute;
  color: #C9E7FF;
  opacity: .72;
  letter-spacing: 18px;
  font-size: 14px;
  white-space: nowrap;
}
.stars-a::before { top: 20px; left: 12px; }
.stars-b::after { top: 58px; right: -30px; opacity: .42; }


.sun-window-readout {
  position: absolute;
  left: 18px;
  right: 18px;
  bottom: 16px;
}

.sun-readout-flex {
  display: flex;
  justify-content: space-between;
  color: #E9F4FF;
  font-weight: 700;
  font-size: 13px;
}

.sun-daylight-label {
  color: var(--amber);
}

.moon-game {
  text-align: center;
}

.moon-visual {
  font-size: 106px;
  line-height: 1;
  margin: 12px auto 28px;
  filter: drop-shadow(0 0 24px rgba(167,139,250,.45));
}

.moon-slider .rc-slider-handle {
  width: 24px !important;
  height: 24px !important;
  margin-top: -9px !important;
  background: var(--purple) !important;
  box-shadow: 0 0 22px rgba(167,139,250,.62) !important;
}


.temperature-game-panel {
  min-height: 285px;
}

.thermometer-game {
  position: relative;
  min-height: 230px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-end;
  padding: 36px 18px 4px;
}

.thermometer-column {
  position: relative;
  height: 142px;
  width: 34px;
  border-radius: 18px 18px 12px 12px;
  border: 3px solid #E9F2FA;
  overflow: hidden;
  background: #091522;
  box-shadow: inset 0 0 0 3px rgba(255,255,255,.04);
}

.thermometer-column::after {
  content: "";
  position: absolute;
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background: #D93025;
  bottom: -25px;
  left: 50%;
  transform: translateX(-50%);
  box-shadow: 0 0 18px rgba(217,48,37,.35);
  z-index: 3;
}

.thermo-fill {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: linear-gradient(180deg, #FF8A65 0%, #FF4D4F 62%, #D93025 100%);
  transition: height .06s linear;
}

.temperature-readout {
  color: var(--text);
  font-size: 23px;
  font-weight: 800;
  margin-bottom: 10px;
}

.thermometer-scale-labels {
  width: min(260px, 92%);
  display: flex;
  justify-content: space-between;
  color: var(--muted);
  font-size: 11px;
  margin: 16px 0 0;
}

.temperature-slider {
  width: min(260px, 92%);
  margin-top: 4px;
}

.temperature-slider .rc-slider-track {
  height: 7px !important;
  background: linear-gradient(90deg, #2F80ED 0%, #56CCF2 30%, #F2C94C 65%, #EB5757 100%) !important;
}

.temperature-slider .rc-slider-rail {
  height: 7px !important;
  background: #263F58 !important;
}

.temperature-slider .rc-slider-handle {
  width: 22px !important;
  height: 22px !important;
  margin-top: -7px !important;
  border: 3px solid #FFFFFF !important;
  background: #EB5757 !important;
  box-shadow: 0 0 14px rgba(235,87,87,.45) !important;
}

.dashboard-footer {
  margin-top: 34px;
  padding: 18px 2px 4px;
  border-top: 1px solid var(--line);
  display: flex;
  justify-content: space-between;
  gap: 20px;
  flex-wrap: wrap;
  font-size: 12px;
  color: var(--muted);
}

.error-text {
  color: #FF9BA8;
}

@media (max-width: 992px) {
  .sunmoon-game-grid {
    grid-template-columns: 1fr;
  }
  .game-panel {
    min-height: 250px;
  }
}

@media (max-width: 640px) {
  .app-shell {
    padding: 16px 12px 40px;
  }
  .hero-panel {
    padding: 22px 18px;
  }
  .dashboard-card .card-body {
    padding: 15px;
  }
  .product-tabs .nav {
    overflow-x: auto;
    flex-wrap: nowrap;
  }
  .product-tabs .nav-link {
    white-space: nowrap;
  }
}


/* ---- Native dark selects: dbc.Select ---- */
.form-select.dark-select,
.dark-select.form-select,
select.dark-select {
  background-color: #10243A !important;
  color: #F1F7FF !important;
  border: 1px solid #35506B !important;
  border-radius: 9px !important;
  min-height: 48px;
  padding: 10px 42px 10px 14px !important;
  font-weight: 600;
  box-shadow: none !important;
  color-scheme: dark;
}

.form-select.dark-select:focus,
.dark-select.form-select:focus,
select.dark-select:focus {
  background-color: #132A43 !important;
  color: #FFFFFF !important;
  border-color: #58A6FF !important;
  box-shadow: 0 0 0 3px rgba(88,166,255,.16) !important;
}

.form-select.dark-select option,
.dark-select.form-select option,
select.dark-select option {
  background-color: #10243A !important;
  color: #F1F7FF !important;
}

.form-select.dark-select:disabled,
.dark-select.form-select:disabled {
  background-color: #0C1B2A !important;
  color: #7F93A8 !important;
}

/* ---- Make every Dash slider label and tooltip readable in dark mode ---- */
.rc-slider-mark-text,
.rc-slider-mark-text-active {
  color: #C8D8E8 !important;
  opacity: 1 !important;
  font-weight: 600 !important;
}

.rc-slider-tooltip-inner {
  background-color: #132A43 !important;
  color: #FFFFFF !important;
  border: 1px solid #58A6FF !important;
  box-shadow: 0 8px 22px rgba(0,0,0,.35) !important;
  font-weight: 800 !important;
  min-width: 48px;
}

.rc-slider-tooltip-arrow {
  border-top-color: #58A6FF !important;
}

.sun-range-slider .rc-slider-mark-text,
.moon-slider .rc-slider-mark-text,
.temperature-slider .rc-slider-mark-text,
.neon-slider .rc-slider-mark-text {
  color: #D6E4F2 !important;
  opacity: 1 !important;
}

/* Sun-range endpoint tooltips must not become white boxes */
.sun-range-slider .rc-slider-tooltip-inner,
.moon-slider .rc-slider-tooltip-inner,
.temperature-slider .rc-slider-tooltip-inner {
  background: #132A43 !important;
  color: #FFFFFF !important;
  border-color: #2DE2E6 !important;
}

/* Stronger readout contrast */
.sun-window-readout,
.sun-readout-flex,
.sun-visual-label,
.sun-visual-time,
.temperature-readout,
.thermometer-scale-labels,
.game-label {
  text-shadow: 0 1px 2px rgba(0,0,0,.45);
}

.sun-readout-flex {
  color: #FFFFFF !important;
}

.sun-visual-label,
.thermometer-scale-labels {
  color: #C8D8E8 !important;
}

.sun-visual-time,
.temperature-readout {
  color: #FFFFFF !important;
}

/* Make labels beneath sliders legible against the game panels */
.moon-game .rc-slider-mark-text,
.temperature-game-panel .rc-slider-mark-text {
  color: #D7E5F2 !important;
}

/* Ensure control labels stay crisp */
.control-label {
  color: #F1F7FF !important;
}


/* ---- Native date controls: fully dark, no third-party white picker shell ---- */
.dark-date-input.form-control,
input.dark-date-input[type="date"] {
  background-color: #10243A !important;
  color: #F1F7FF !important;
  border: 1px solid #35506B !important;
  border-radius: 9px !important;
  min-height: 48px;
  padding: 10px 12px !important;
  font-weight: 650;
  color-scheme: dark;
  box-shadow: none !important;
}

.dark-date-input.form-control:focus,
input.dark-date-input[type="date"]:focus {
  background-color: #132A43 !important;
  color: #FFFFFF !important;
  border-color: #58A6FF !important;
  box-shadow: 0 0 0 3px rgba(88,166,255,.16) !important;
}

.dark-date-input::-webkit-calendar-picker-indicator {
  filter: invert(1) brightness(1.5);
  opacity: .9;
  cursor: pointer;
}

.mini-control-label {
  color: #9FB3C8;
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .6px;
  margin-bottom: 5px;
}

/* ---- Custom slider scales: always readable, no white tooltip boxes ---- */
.custom-slider-scale {
  width: 100%;
  display: flex;
  justify-content: space-between;
  gap: 8px;
  margin-top: 10px;
  color: #D6E4F2;
  font-size: 11px;
  font-weight: 700;
  line-height: 1.2;
}

.custom-slider-scale span {
  color: #D6E4F2 !important;
  opacity: 1 !important;
  text-align: center;
}

.moon-scale {
  padding: 0 2px;
}

.temperature-scale {
  width: min(280px, 96%);
  margin-left: auto;
  margin-right: auto;
}

.temperature-scale span {
  min-width: 38px;
}

/* Hide any tooltip remnants from rc-slider in the game controls */
.sunmoon-card .rc-slider-tooltip,
.sunmoon-card .rc-slider-tooltip-content,
.sunmoon-card .rc-slider-tooltip-inner,
.sunmoon-card .rc-slider-tooltip-arrow {
  display: none !important;
}

/* No built-in marks are used in Sun & Moon now */
.sunmoon-card .rc-slider-mark {
  display: none !important;
}


/* ---- v10 layout alignment ---- */
.date-range-inline {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 28px minmax(0, 1fr);
  align-items: center;
  gap: 6px;
  min-height: 48px;
}

.date-range-arrow {
  color: #9FB3C8;
  font-size: 22px;
  font-weight: 800;
  text-align: center;
}

.date-range-inline .dark-date-input {
  width: 100%;
  min-width: 0;
}

.sunmoon-game-grid {
  align-items: stretch;
  grid-template-columns: minmax(0, 1.75fr) minmax(300px, .82fr) minmax(270px, .72fr);
}

.game-panel {
  min-height: 520px;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.sun-controls-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-bottom: 14px;
}

.sun-control-card {
  min-height: 225px;
  border: 1px solid rgba(246,200,95,.22);
  background: rgba(246,200,95,.055);
  border-radius: 14px;
  padding: 14px 18px 12px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.sun-control-card .sun-main-icon {
  font-size: 66px;
  line-height: 1;
}

.sun-control-card .sun-visual-label {
  margin-top: 6px;
  color: #C8D8E8;
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .7px;
}

.sun-control-card .sun-visual-time {
  color: #FFFFFF;
  font-size: 21px;
  font-weight: 800;
  margin: 2px 0 12px;
}

.single-sun-slider {
  width: 92%;
  margin: 2px auto 0;
}

.single-sun-slider .rc-slider-track {
  background: linear-gradient(90deg, #F6C85F, #FF9966) !important;
  height: 6px !important;
}

.single-sun-slider .rc-slider-rail {
  background: #294159 !important;
  height: 6px !important;
}

.single-sun-slider .rc-slider-handle {
  width: 22px !important;
  height: 22px !important;
  margin-top: -8px !important;
  background: #F6C85F !important;
  border: 3px solid #FFF !important;
  box-shadow: 0 0 18px rgba(246,200,95,.55) !important;
}

.sunset-slider .rc-slider-track {
  background: linear-gradient(90deg, #FF9966, #F472B6) !important;
}

.sun-slider-scale {
  width: 92%;
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
  color: #B9CADB;
  font-size: 10px;
  font-weight: 700;
}

.sky-track {
  flex: 1 1 auto;
  min-height: 160px;
  height: auto;
  margin-top: 0;
}

.sky-sun-marker {
  position: absolute;
  top: 48%;
  transform: translate(-50%, -50%);
  font-size: 28px;
  line-height: 1;
  z-index: 3;
  filter: drop-shadow(0 0 14px rgba(246,200,95,.7));
}

.sky-sun-marker::before {
  content: "☀";
}

.sunset-sky-marker {
  filter: drop-shadow(0 0 14px rgba(244,114,182,.58));
}

.moon-game,
.temperature-game-panel {
  justify-content: flex-start;
}

.game-spacer-top {
  height: 20px;
}

.moon-visual {
  margin: 22px auto 0;
  font-size: 112px;
}

.game-control-bottom {
  margin-top: auto;
  width: 100%;
  padding: 0 8px 10px;
}

.temperature-control-bottom {
  padding-bottom: 4px;
}

.thermometer-visual-stage {
  flex: 1 1 auto;
  min-height: 285px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.thermometer-column {
  height: 175px;
}

.temperature-readout {
  margin-bottom: 14px;
}

.thermometer-scale-labels,
.temperature-slider,
.temperature-scale {
  width: 94%;
  max-width: none;
}

.custom-slider-scale {
  font-size: 10px;
}

.moon-scale {
  margin-top: 12px;
}

@media (max-width: 992px) {
  .sunmoon-game-grid {
    grid-template-columns: 1fr;
  }
  .game-panel {
    min-height: 450px;
  }
}

@media (max-width: 640px) {
  .date-range-inline {
    grid-template-columns: 1fr;
  }
  .date-range-arrow {
    transform: rotate(90deg);
    line-height: 1;
  }
  .sun-controls-grid {
    grid-template-columns: 1fr;
  }
}


/* ---- FINAL v3: Predict tab ---- */
.predict-control-card,
.predict-output-card {
  height: auto !important;
  min-height: 0 !important;
  align-self: start !important;
}

.predict-control-card .card-body,
.predict-output-card .card-body {
  height: auto !important;
  min-height: 0 !important;
}

.scenario-chart-wrap {
  width: 100%;
  height: 500px !important;
  min-height: 500px !important;
  max-height: 500px !important;
  overflow: hidden;
}

.scenario-chart-wrap .js-plotly-plot,
.scenario-chart-wrap .plot-container,
.scenario-chart-wrap .svg-container {
  height: 500px !important;
  max-height: 500px !important;
}

.scenario-chart-help {
  margin-top: 14px;
  margin-bottom: 8px;
}

/* Native Predict dropdowns: dark in all states */
#scenario-day,
#scenario-season,
#scenario-day.form-select,
#scenario-season.form-select {
  background-color: #10243A !important;
  color: #F1F7FF !important;
  -webkit-text-fill-color: #F1F7FF !important;
  border: 1px solid #35506B !important;
  color-scheme: dark !important;
}

#scenario-day option,
#scenario-season option {
  background-color: #10243A !important;
  color: #F1F7FF !important;
}

/* Scenario sliders: values live beside labels; never render white tooltips/inputs */
.neon-slider input,
.neon-slider .rc-slider-tooltip,
.neon-slider .rc-slider-tooltip-content,
.neon-slider .rc-slider-tooltip-inner,
.neon-slider .rc-slider-tooltip-arrow {
  display: none !important;
}

.neon-slider .rc-slider-rail {
  height: 7px !important;
  background: #263F58 !important;
}

.neon-slider .rc-slider-track {
  height: 7px !important;
  background: linear-gradient(90deg, #58A6FF, #A78BFA, #2DE2E6) !important;
}

.neon-slider .rc-slider-handle {
  width: 22px !important;
  height: 22px !important;
  margin-top: -7px !important;
  border: 3px solid #FFFFFF !important;
  background: #58A6FF !important;
  box-shadow: 0 0 16px rgba(88,166,255,.55) !important;
}

.control-value {
  min-width: 112px;
  text-align: right;
  color: #2DE2E6 !important;
  font-weight: 800;
}

/* Keep Dash DataTable dark on Predict */
.dash-table-container,
.dash-spreadsheet-container,
.dash-spreadsheet-inner,
.dash-spreadsheet-inner table {
  background-color: #0C1B2A !important;
  color: #F1F7FF !important;
}

        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""
app.title = "London Bikes Demand Lab"


def explore_tab():
    return html.Div(
        [
            card(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Weather category", className="control-label"),
                                    dbc.Select(
                                        id="explore-group",
                                        options=[{"label": g, "value": g} for g in AVAILABLE_GROUPS],
                                        value=DEFAULT_GROUP,
                                        className="dark-select",
                                    ),
                                    html.Label("Specific variable", className="control-label mt-3"),
                                    dbc.Select(
                                        id="explore-variable",
                                        options=dropdown_options(AVAILABLE_GROUPS.get(DEFAULT_GROUP, [])),
                                        value=DEFAULT_VAR,
                                        className="dark-select",
                                    ),
                                ],
                                md=7,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Colour observations by", className="control-label"),
                                    dbc.Select(
                                        id="explore-colour",
                                        options=[
                                            {"label": "Weekend / weekday", "value": "weekend"},
                                            {"label": "Season", "value": "season_name"},
                                            {"label": "Weather conditions", "value": "conditions"},
                                            {"label": "Day of week", "value": "day_of_week"},
                                        ],
                                        value="season_name",
                                        className="dark-select",
                                    ),
                                ],
                                md=5,
                            ),
                        ],
                        className="g-4",
                    ),
                ],
                class_name="dashboard-card mt-3",
            ),

            html.Div(id="explore-insights", className="insights-grid mt-3"),

            card(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Relationship explorer", className="section-heading"),
                                    html.P(
                                        "Hover for exact values. Scroll to zoom, drag to zoom into an area, and double-click to reset.",
                                        className="chart-help",
                                    ),
                                ]
                            ),
                        ],
                        className="d-flex justify-content-between align-items-start",
                    ),
                    dcc.Graph(id="explore-scatter", config=GRAPH_CONFIG),
                ],
                class_name="dashboard-card mt-3 chart-card",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                dcc.Graph(id="weekday-bar", config=BAR_CONFIG),
                            ],
                            class_name="dashboard-card predict-control-card",
                        ),
                        lg=6,
                    ),
                    dbc.Col(
                        card(
                            [
                                dcc.Graph(id="month-bar", config=GRAPH_CONFIG),
                            ],
                            class_name="dashboard-card predict-output-card",
                        ),
                        lg=6,
                    ),
                ],
                className="g-3 mt-1",
            ),

            card(
                [
                    html.H3("Demand through time", className="section-heading"),
                    html.P(
                        "Use the date controls for broad filtering. On the chart itself, scroll to zoom, drag over a period to zoom further, and double-click to reset.",
                        className="chart-help",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Date range", className="control-label"),
                                    html.Div(
                                        [
                                            dbc.Input(
                                                id="time-start-date",
                                                type="date",
                                                value=min_date,
                                                min=min_date,
                                                max=max_date,
                                                className="dark-date-input",
                                            ),
                                            html.Div("→", className="date-range-arrow"),
                                            dbc.Input(
                                                id="time-end-date",
                                                type="date",
                                                value=max_date,
                                                min=min_date,
                                                max=max_date,
                                                className="dark-date-input",
                                            ),
                                        ],
                                        className="date-range-inline",
                                    ),
                                ],
                                lg=4,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Aggregation", className="control-label"),
                                    dbc.Select(
                                        id="time-aggregation",
                                        options=[
                                            {"label": "Daily", "value": "D"},
                                            {"label": "Weekly", "value": "W"},
                                            {"label": "Monthly", "value": "MS"},
                                        ],
                                        value="W",
                                        className="dark-select",
                                    ),
                                ],
                                lg=2,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Overlay category", className="control-label"),
                                    dbc.Select(
                                        id="time-overlay-group",
                                        options=[{"label": "None", "value": "__none__"}]
                                        + [{"label": g, "value": g} for g in AVAILABLE_GROUPS],
                                        value="Temperature",
                                        className="dark-select",
                                    ),
                                ],
                                lg=3,
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.Label("Overlay variable", className="control-label"),
                                            dbc.Select(
                                                id="time-overlay-variable",
                                                options=dropdown_options(AVAILABLE_GROUPS.get("Temperature", [])),
                                                value=AVAILABLE_GROUPS.get("Temperature", [None])[0]
                                                if "Temperature" in AVAILABLE_GROUPS else None,
                                                className="dark-select",
                                            ),
                                        ],
                                        id="time-overlay-variable-container",
                                    ),
                                ],
                                lg=3,
                            ),
                        ],
                        className="g-3",
                    ),
                    html.Div(id="time-insights", className="insights-grid mt-3"),
                    dcc.Graph(id="time-series", config=GRAPH_CONFIG),
                ],
                class_name="dashboard-card mt-3 chart-card",
            ),
        ]
    )


def relationships_tab():
    return html.Div(
        [
            card(
                [
                    html.H3("Correlation overview", className="section-heading"),
                    html.P(
                        "Start at category level. Each square shows the strongest Pearson relationship inside that category pair. Hover to see which two variables produced it, then click a square to drill down. This overview is intentionally fixed rather than zoomable.",
                        className="section-copy",
                    ),
                    dcc.Graph(id="corr-heatmap", config=BAR_CONFIG),
                ],
                class_name="dashboard-card mt-3",
            ),

            card(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("X category", className="control-label"),
                                    dbc.Select(
                                        id="corr-x-group",
                                        options=[{"label": g, "value": g} for g in AVAILABLE_REL_GROUPS],
                                        value="Temperature",
                                        className="dark-select",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("X variable", className="control-label mt-3"),
                                            dbc.Select(
                                                id="corr-x",
                                                options=dropdown_options(AVAILABLE_REL_GROUPS.get("Temperature", [])),
                                                value=AVAILABLE_REL_GROUPS.get("Temperature", [None])[0],
                                                className="dark-select",
                                            ),
                                        ],
                                        id="corr-x-variable-container",
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Y category", className="control-label"),
                                    dbc.Select(
                                        id="corr-y-group",
                                        options=[{"label": g, "value": g} for g in AVAILABLE_REL_GROUPS],
                                        value="Bikes hired",
                                        className="dark-select",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Y variable", className="control-label mt-3"),
                                            dbc.Select(
                                                id="corr-y",
                                                options=dropdown_options(AVAILABLE_REL_GROUPS.get("Bikes hired", [])),
                                                value="bikes_hired",
                                                className="dark-select",
                                            ),
                                        ],
                                        id="corr-y-variable-container",
                                        style={"display": "none"},
                                    ),
                                ],
                                md=6,
                            ),
                        ],
                        className="g-4",
                    )
                ],
                class_name="dashboard-card mt-3",
            ),

            html.Div(id="corr-insights", className="insights-grid mt-3"),

            card(
                [
                    html.H3("Detailed relationship", className="section-heading"),
                    html.P(
                        "Hover for exact values. Scroll to zoom, drag to zoom into a cluster, and double-click to reset.",
                        className="chart-help",
                    ),
                    dcc.Graph(id="corr-scatter", config=GRAPH_CONFIG),
                ],
                class_name="dashboard-card mt-3 chart-card",
            ),

            card(
                [dcc.Graph(id="bikes-corr-ranking", config=BAR_CONFIG)],
                class_name="dashboard-card mt-3",
            ),
        ]
    )


def sunmoon_tab():
    if not SUNMOON_AVAILABLE:
        return dbc.Alert("Sun, moon and temperature fields are unavailable in this dataset.", color="warning")

    return html.Div(
        [
            card(
                [
                    html.H3("Natural cycles playground", className="section-heading"),
                    html.P(
                        "Build a hypothetical day by moving the controls directly on the visual. The dashboard then finds the most similar historical days and compares their bike demand with the overall average.",
                        className="section-copy",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("SUNRISE ↔ SUNSET", className="game-label"),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div("☀️", className="sun-main-icon"),
                                                    html.Div("Sunrise", className="sun-visual-label"),
                                                    html.Div(id="sunrise-time", className="sun-visual-time"),
                                                    dcc.Slider(
                                                        id="sunrise-slider",
                                                        min=sunrise_min,
                                                        max=sunrise_max,
                                                        step=0.05,
                                                        value=sunrise_default,
                                                        marks=None,
                                                        updatemode="mouseup",
                                                        className="single-sun-slider",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Span(hour_text(sunrise_min)),
                                                            html.Span(hour_text(sunrise_max)),
                                                        ],
                                                        className="sun-slider-scale",
                                                    ),
                                                ],
                                                className="sun-control-card",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div("☀️", className="sun-main-icon sunset-sun"),
                                                    html.Div("Sunset", className="sun-visual-label"),
                                                    html.Div(id="sunset-time", className="sun-visual-time"),
                                                    dcc.Slider(
                                                        id="sunset-slider",
                                                        min=sunset_min,
                                                        max=sunset_max,
                                                        step=0.05,
                                                        value=sunset_default,
                                                        marks=None,
                                                        updatemode="mouseup",
                                                        className="single-sun-slider sunset-slider",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Span(hour_text(sunset_min)),
                                                            html.Span(hour_text(sunset_max)),
                                                        ],
                                                        className="sun-slider-scale",
                                                    ),
                                                ],
                                                className="sun-control-card",
                                            ),
                                        ],
                                        className="sun-controls-grid",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(className="stars stars-a"),
                                            html.Div(className="stars stars-b"),
                                            html.Div(id="sunrise-sky-marker", className="sky-sun-marker sunrise-sky-marker"),
                                            html.Div(id="sunset-sky-marker", className="sky-sun-marker sunset-sky-marker"),
                                            html.Div(id="sun-window-readout", className="sun-window-readout"),
                                        ],
                                        className="sky-track",
                                    ),
                                ],
                                className="game-panel sky-game",
                            ),

                            html.Div(
                                [
                                    html.Div("MOON PHASE", className="game-label"),
                                    html.Div(className="game-spacer-top"),
                                    html.Div(id="moon-visual", className="moon-visual"),
                                    html.Div(className="game-control-bottom", children=[
                                        dcc.Slider(
                                            id="moon-slider",
                                            min=0,
                                            max=1,
                                            step=0.01,
                                            value=0.5,
                                            marks=None,
                                            updatemode="mouseup",
                                            className="moon-slider",
                                        ),
                                        html.Div(
                                            [
                                                html.Span("New"),
                                                html.Span("First quarter"),
                                                html.Span("Full"),
                                                html.Span("Last quarter"),
                                                html.Span("New"),
                                            ],
                                            className="custom-slider-scale moon-scale",
                                        ),
                                    ]),
                                ],
                                className="game-panel moon-game",
                            ),

                            html.Div(
                                [
                                    html.Div("TEMPERATURE", className="game-label"),
                                    html.Div(
                                        [
                                            html.Div(id="temperature-readout", className="temperature-readout"),
                                            html.Div(
                                                html.Div(id="thermo-fill", className="thermo-fill"),
                                                className="thermometer-column",
                                            ),
                                        ],
                                        className="thermometer-visual-stage",
                                    ),
                                    html.Div(className="game-control-bottom temperature-control-bottom", children=[
                                        html.Div(
                                            [
                                                html.Span("Cold"),
                                                html.Span("Hot"),
                                            ],
                                            className="thermometer-scale-labels",
                                        ),
                                        dcc.Slider(
                                            id="sunmoon-temp-slider",
                                            min=-5,
                                            max=35,
                                            step=0.5,
                                            value=temp_default,
                                            marks=None,
                                            updatemode="mouseup",
                                            className="temperature-slider",
                                        ),
                                        html.Div(
                                            [
                                                html.Span("-5 °C"),
                                                html.Span("5 °C"),
                                                html.Span("15 °C"),
                                                html.Span("25 °C"),
                                                html.Span("35 °C"),
                                            ],
                                            className="custom-slider-scale temperature-scale",
                                        ),
                                    ]),
                                ],
                                className="game-panel temperature-game-panel",
                            ),
                        ],
                        className="sunmoon-game-grid",
                    ),
                ],
                class_name="dashboard-card mt-3 sunmoon-card",
            ),

            html.Div(id="sunmoon-insights", className="metrics-grid mt-3"),

            card(
                [
                    html.H3("Historically similar days", className="section-heading"),
                    html.P(
                        "These are the historical days most similar to your chosen moon phase, daylight window and temperature. This is an exploratory comparison, not a causal prediction.",
                        className="chart-help",
                    ),
                    dcc.Graph(id="sunmoon-chart", config=GRAPH_CONFIG),
                ],
                class_name="dashboard-card mt-3 chart-card",
            ),
        ]
    )


def predict_tab():
    final_model_terms = (
        "Maximum temperature, wind speed, solar radiation, sea-level pressure, "
        "visibility and precipitation, with day-of-week, season, weekend and interaction effects."
    )

    return html.Div(
        [
            card(
                [
                    html.H3("Final model", className="section-heading"),
                    html.P(
                        final_model_terms,
                        className="section-copy",
                    ),
                    html.Div(
                        [
                            metric_card("Coefficients loaded", f"{len(COEFFS):,}", "Including intercept, categorical and interaction terms", C["blue"]),
                            metric_card("Weather predictors", f"{len(model_numeric_variables(COEFFS))}", "Used by both forecast and simulator", C["teal"]),
                            metric_card("Interactions", "3", "Visibility × season and precipitation × weekend", C["purple"]),
                        ],
                        className="metrics-grid",
                    ),
                ],
                class_name="dashboard-card mt-3",
            ),

            card(
                [
                    html.H3("Model credibility", className="section-heading"),
                    html.P(
                        "Before using the model for decisions, check how closely its historical predictions match actual bike hires.",
                        className="section-copy",
                    ),
                    html.Div(model_credibility_content(), id="model-credibility"),
                ],
                class_name="dashboard-card mt-3",
            ),

            card(
                [
                    html.H3("Real-weather predictions", className="section-heading"),
                    html.P(
                        "The first section uses historical Open-Meteo weather for 1–7 January 2026. "
                        "The second pulls Open-Meteo's live London weather forecast for the next five days each time predictions are refreshed. "
                        "Those forecast weather values are aggregated daily and then passed into the fitted demand model.",
                        className="section-copy",
                    ),
                    dbc.Button("Refresh predictions", id="refresh-predictions", n_clicks=0, color="primary"),
                    html.Div(COEFF_ERROR or "", className="error-text mt-2"),
                ],
                class_name="dashboard-card mt-3",
            ),

            dcc.Loading(html.Div(id="prediction-content"), type="circle"),

            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                html.H3("What-if weather simulator", className="section-heading"),
                                html.P(
                                    "Adjust a hypothetical day's weather, weekday and season. "
                                    "Weekend status is derived automatically from the selected weekday.",
                                    className="section-copy",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Day of week", className="control-label"),
                                                dbc.Select(
                                                    id="scenario-day",
                                                    options=[
                                                        {"label": DAY_LABELS[d], "value": d}
                                                        for d in DAY_ORDER
                                                    ],
                                                    value="Mon",
                                                    className="dark-select mb-3",
                                                ),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Season", className="control-label"),
                                                dbc.Select(
                                                    id="scenario-season",
                                                    options=[
                                                        {"label": s, "value": s}
                                                        for s in SEASON_ORDER
                                                    ],
                                                    value="Spring",
                                                    className="dark-select mb-3",
                                                ),
                                            ],
                                            md=6,
                                        ),
                                    ],
                                    className="g-2",
                                ),
                                scenario_controls(),
                            ],
                            class_name="dashboard-card h-100",
                        ),
                        lg=5,
                    ),
                    dbc.Col(
                        card(
                            [
                                html.Div(id="scenario-prediction"),
                                html.P(
                                    "This shows the intercept plus the 10 largest active effects for your selected scenario. "
                                    "Bars to the right increase predicted hires; bars to the left decrease them.",
                                    className="chart-help scenario-chart-help",
                                ),
                                html.Div(
                                    dcc.Graph(
                                        id="scenario-contribution-chart",
                                        config=BAR_CONFIG,
                                        style={"height": "500px", "width": "100%"},
                                    ),
                                    className="scenario-chart-wrap",
                                ),
                            ],
                            class_name="dashboard-card h-100",
                        ),
                        lg=7,
                    ),
                ],
                className="g-3 mt-1",
            ),
        ]
    )


tabs = [
    dbc.Tab(explore_tab(), label="Explore", tab_id="explore"),
    dbc.Tab(relationships_tab(), label="Relationships", tab_id="relationships"),
    dbc.Tab(predict_tab(), label="Prediction Model", tab_id="predict"),
]
if SUNMOON_AVAILABLE:
    tabs.append(
        dbc.Tab(
            sunmoon_tab(),
            label="Natural Cycles Playground",
            tab_id="sunmoon",
        )
    )


app.layout = html.Div(
    [
        html.Div(
            [
                html.Div("TfL ANALYTICS", className="eyebrow"),
                html.H1("London Bikes: Demand Lab", className="hero-title"),
                html.P(
                    "Explore what drives daily bike hires, uncover seasonal patterns and test the fitted demand model.",
                    className="hero-subtitle",
                ),
                dbc.Badge(
                    "Final fitted model loaded",
                    color="success",
                    className="mt-3 status-badge",
                ) if not COEFFS.empty else None,
            ],
            className="hero-panel",
        ),

        html.Div(dataset_metrics(), className="metrics-grid"),

        dbc.Tabs(
            tabs,
            id="main-tabs",
            active_tab="explore",
            className="product-tabs mt-4",
            persistence=True,
            persistence_type="session",
        ),

        html.Footer(
            [
                html.Div("Data sources: London Bikes course dataset + Open-Meteo weather · Model formula, fit statistics and coefficients from the submitted group notebook", className="footer-left"),
                html.Div(
                    f"Historical data through {data_through} · Dashboard rendered {last_rendered}",
                    className="footer-right",
                ),
            ],
            className="dashboard-footer",
        ),
    ],
    className="app-shell",
)


@app.callback(
    Output("explore-variable", "options"),
    Output("explore-variable", "value"),
    Input("explore-group", "value"),
)
def update_explore_variable_options(group):
    variables = AVAILABLE_GROUPS.get(group, [])
    return dropdown_options(variables), (variables[0] if variables else None)


@app.callback(
    Output("time-overlay-variable", "options"),
    Output("time-overlay-variable", "value"),
    Output("time-overlay-variable-container", "style"),
    Input("time-overlay-group", "value"),
)
def update_overlay_variable_options(group):
    if group == "__none__":
        return [], None, {"display": "none"}

    variables = AVAILABLE_GROUPS.get(group, [])
    return dropdown_options(variables), (variables[0] if variables else None), {}


@app.callback(
    Output("corr-x", "options"),
    Output("corr-x", "value"),
    Output("corr-x-variable-container", "style"),
    Input("corr-x-group", "value"),
)
def update_corr_x_options(group):
    variables = AVAILABLE_REL_GROUPS.get(group, [])
    if group == "Bikes hired":
        return dropdown_options(variables), "bikes_hired", {"display": "none"}
    return dropdown_options(variables), (variables[0] if variables else None), {}


@app.callback(
    Output("corr-y", "options"),
    Output("corr-y", "value"),
    Output("corr-y-variable-container", "style"),
    Input("corr-y-group", "value"),
)
def update_corr_y_options(group):
    variables = AVAILABLE_REL_GROUPS.get(group, [])
    if group == "Bikes hired":
        return dropdown_options(variables), "bikes_hired", {"display": "none"}
    return dropdown_options(variables), (variables[0] if variables else None), {}


@app.callback(
    Output("explore-scatter", "figure"),
    Output("weekday-bar", "figure"),
    Output("month-bar", "figure"),
    Output("explore-insights", "children"),
    Input("explore-variable", "value"),
    Input("explore-colour", "value"),
)
def update_explore(variable, colour_by):
    if BIKE_DF.empty or not variable:
        msg = BIKE_ERROR or "Bike data is unavailable."
        return empty_figure(msg), empty_figure(msg), empty_figure(msg), []

    df = BIKE_DF.copy()

    if colour_by == "weekend":
        df["colour_display"] = df["weekend"].map(
            {True: "Weekend", False: "Weekday", 1: "Weekend", 0: "Weekday"}
        ).fillna(df["weekend"].astype(str))
    else:
        df["colour_display"] = df[colour_by].astype(str)

    scatter = px.scatter(
        df,
        x=variable,
        y="bikes_hired",
        color="colour_display",
        opacity=0.55,
        color_discrete_sequence=[
            C["blue"], C["teal"], C["purple"], C["pink"], C["amber"], C["cyan"]
        ],
        title=f"Bikes hired vs {display_name(variable)}",
        labels={
            variable: axis_label(variable),
            "bikes_hired": axis_label("bikes_hired"),
            "colour_display": display_name(colour_by),
        },
        hover_data={
            "date": "|%d %b %Y",
            "day_of_week": True,
            variable: ":.2f",
            "bikes_hired": ":,.2f",
        },
    )
    scatter = add_fit_line(scatter, df[variable], df["bikes_hired"])
    scatter.update_layout(legend_title_text=display_name(colour_by))
    scatter = style_figure(scatter, 720, "zoom")

    weekday = df.groupby("day_of_week", as_index=False)["bikes_hired"].mean()
    weekday["day_of_week"] = pd.Categorical(weekday["day_of_week"], categories=DAY_ORDER, ordered=True)
    weekday = weekday.sort_values("day_of_week")

    weekday_fig = px.bar(
        weekday,
        x="day_of_week",
        y="bikes_hired",
        text=weekday["bikes_hired"].map(lambda x: f"{x:,.0f}"),
        title="Average bikes hired by day of week",
        labels={"day_of_week": "Day of week", "bikes_hired": axis_label("bikes_hired")},
    )
    weekday_fig.update_traces(
        marker_color=C["teal"],
        textposition="outside",
        hovertemplate="Day=%{x}<br>Average bikes hired=%{y:,.2f}<extra></extra>",
    )
    weekday_fig = style_figure(weekday_fig, 430, False)

    month = df.groupby("month_name", as_index=False)["bikes_hired"].mean()
    month["month_name"] = pd.Categorical(month["month_name"], categories=MONTH_ORDER, ordered=True)
    month = month.sort_values("month_name")

    month_fig = px.line(
        month,
        x="month_name",
        y="bikes_hired",
        markers=True,
        title="Seasonality: average bikes hired by month",
        labels={"month_name": "Month", "bikes_hired": axis_label("bikes_hired")},
    )
    month_fig.update_traces(
        line={"color": C["blue"], "width": 3},
        marker={"size": 8, "color": C["pink"]},
        hovertemplate="Month=%{x}<br>Average bikes hired=%{y:,.2f}<extra></extra>",
    )
    month_fig = style_figure(month_fig, 430)

    stats = relationship_stats(variable)
    insights = []
    if stats:
        strength, direction = relationship_strength(stats["r"])
        unit = UNITS.get(variable, "unit")
        change_word = "increase" if stats["slope"] >= 0 else "decrease"
        quartile_word = "higher" if stats["quartile_pct"] >= 0 else "lower"

        insights = [
            insight_card(
                "What it means",
                f"{strength} {direction} relationship",
                f"Pearson r = {stats['r']:.2f}. Historically, {display_name(variable).lower()} and bikes hired tend to move together with a {strength.lower()} {direction} pattern.",
                C["blue"],
            ),
            insight_card(
                "Trend-line estimate",
                f"{abs(stats['slope']):,.0f} hires per {unit}",
                f"The fitted line shows an average {change_word} of about {abs(stats['slope']):,.0f} bikes hired for a one-{unit} increase in {display_name(variable).lower()}.",
                C["teal"],
                "Descriptive association, not a causal effect.",
            ),
            insight_card(
                "High vs low days",
                f"{abs(stats['quartile_pct']):.1f}% {quartile_word}",
                f"Days in the highest quarter of {display_name(variable).lower()} averaged {stats['high_mean']:,.0f} hires, compared with {stats['low_mean']:,.0f} in the lowest quarter.",
                C["pink"],
            ),
        ]

    return scatter, weekday_fig, month_fig, insights


@app.callback(
    Output("time-series", "figure"),
    Output("time-insights", "children"),
    Input("time-start-date", "value"),
    Input("time-end-date", "value"),
    Input("time-aggregation", "value"),
    Input("time-overlay-group", "value"),
    Input("time-overlay-variable", "value"),
)
def update_time_series(start_date, end_date, aggregation, overlay_group, overlay_variable):
    if BIKE_DF.empty:
        return empty_figure(BIKE_ERROR or "Bike data unavailable."), []

    df = BIKE_DF.copy()
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    if df.empty:
        return empty_figure("No observations in this date range."), []

    overlay = None
    if overlay_group != "__none__" and overlay_variable in df.columns:
        overlay = overlay_variable

    columns = ["bikes_hired"] + ([overlay] if overlay else [])
    temp = df.set_index("date")[columns]

    if aggregation == "D":
        agg = temp.resample("D").mean()
        label = "Daily"
    elif aggregation == "W":
        agg = temp.resample("W").mean()
        label = "Weekly average"
    else:
        agg = temp.resample("MS").mean()
        label = "Monthly average"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=agg.index,
            y=agg["bikes_hired"],
            mode="lines",
            name="Bikes hired",
            line={"color": C["blue"], "width": 2.8},
            hovertemplate="%{x|%d %b %Y}<br>Bikes hired=%{y:,.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.update_yaxes(title_text=axis_label("bikes_hired"), secondary_y=False)

    if overlay and overlay in agg.columns:
        fig.add_trace(
            go.Scatter(
                x=agg.index,
                y=agg[overlay],
                mode="lines",
                name=display_name(overlay),
                line={"color": C["teal"], "width": 2.2},
                opacity=0.9,
                hovertemplate="%{x|%d %b %Y}<br>"
                + f"{display_name(overlay)}=%{{y:,.2f}}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text=axis_label(overlay), secondary_y=True)

    fig.update_layout(
        title=f"{label} bikes hired over time",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.10},
    )
    fig = style_figure(fig, 620, "zoom")

    peak = df.loc[df["bikes_hired"].idxmax()]
    average = df["bikes_hired"].mean()

    trend_df = agg["bikes_hired"].dropna()
    trend_text = "Flat"
    if len(trend_df) >= 2:
        x = np.arange(len(trend_df))
        trend_slope = np.polyfit(x, trend_df.values, 1)[0]
        trend_text = "Upward" if trend_slope > 0 else "Downward" if trend_slope < 0 else "Flat"

    insights = [
        insight_card(
            "Selected period",
            f"{average:,.0f} average bikes hired",
            f"The busiest day in this date range was {peak['date']:%d %b %Y}, with {peak['bikes_hired']:,.0f} hires.",
            C["blue"],
        ),
        insight_card(
            "Trend",
            f"{trend_text} at {label.lower()} level",
            f"Across the displayed {label.lower()} points, the fitted direction is {trend_text.lower()}. Change the date range to see whether that pattern persists.",
            C["purple"],
        ),
    ]

    if overlay:
        pair = df[["bikes_hired", overlay]].dropna()
        if len(pair) >= 3:
            r = pair["bikes_hired"].corr(pair[overlay])
            strength, direction = relationship_strength(r)
            insights.append(
                insight_card(
                    "Overlay relationship",
                    f"{strength} {direction}: r = {r:.2f}",
                    f"Within the selected dates, {display_name(overlay).lower()} and bikes hired show a {strength.lower()} {direction} linear relationship.",
                    C["teal"],
                )
            )

    return fig, insights


@app.callback(
    Output("corr-heatmap", "figure"),
    Output("bikes-corr-ranking", "figure"),
    Input("corr-x-group", "value"),
)
def correlation_overview(_):
    if BIKE_DF.empty:
        msg = BIKE_ERROR or "Bike data unavailable."
        return empty_figure(msg), empty_figure(msg)

    names, z, pair_text = category_correlation_matrix()

    heat = go.Figure(
        data=go.Heatmap(
            z=z,
            x=names,
            y=names,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0, C["blue"]],
                [0.5, C["panel2"]],
                [1, C["pink"]],
            ],
            colorbar={"title": "Pearson r"},
            customdata=pair_text,
            hovertemplate=(
                "%{y} × %{x}<br>"
                "Strongest pair: %{customdata}<br>"
                "r=%{z:.2f}<extra></extra>"
            ),
        )
    )
    heat.update_layout(
        title="Category-level correlation overview",
        height=600,
        margin={"l": 120, "b": 100, "t": 78, "r": 42},
    )
    heat = style_figure(heat, 600, False)

    rows = []
    for group, variables in AVAILABLE_GROUPS.items():
        best_corr = None
        best_var = None
        for variable in variables:
            pair = BIKE_DF[["bikes_hired", variable]].dropna()
            if len(pair) < 3:
                continue
            corr = pair["bikes_hired"].corr(pair[variable])
            if pd.isna(corr):
                continue
            if best_corr is None or abs(corr) > abs(best_corr):
                best_corr = corr
                best_var = variable

        if best_corr is not None:
            rows.append({
                "category": group,
                "correlation": best_corr,
                "variable": display_name(best_var),
            })

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        bar = empty_figure("No correlation ranking available.")
    else:
        ranking["abs_corr"] = ranking["correlation"].abs()
        ranking = ranking.sort_values("abs_corr", ascending=True)

        bar = px.bar(
            ranking,
            x="correlation",
            y="category",
            orientation="h",
            custom_data=["variable"],
            title="Strongest bikes-hired relationship within each weather category",
            labels={"correlation": "Pearson correlation with bikes hired", "category": ""},
        )
        bar.update_traces(
            marker_color=C["purple"],
            hovertemplate=(
                "%{y}<br>"
                "Variable=%{customdata[0]}<br>"
                "r=%{x:.2f}<extra></extra>"
            ),
        )
        bar = style_figure(bar, 470, False)

    return heat, bar


@app.callback(
    Output("corr-x-group", "value"),
    Output("corr-y-group", "value"),
    Input("corr-heatmap", "clickData"),
    prevent_initial_call=True,
)
def heatmap_click(click_data):
    if not click_data or not click_data.get("points"):
        return no_update, no_update

    point = click_data["points"][0]
    x_group = point.get("x")
    y_group = point.get("y")

    if x_group in AVAILABLE_REL_GROUPS and y_group in AVAILABLE_REL_GROUPS:
        return x_group, y_group

    return no_update, no_update


@app.callback(
    Output("corr-scatter", "figure"),
    Output("corr-insights", "children"),
    Input("corr-x", "value"),
    Input("corr-y", "value"),
)
def relationship_explorer(x, y):
    if BIKE_DF.empty or not x or not y:
        return empty_figure("Choose two variables."), []

    temp = BIKE_DF[[x, y]].dropna()
    if temp.empty:
        return empty_figure("No complete observations for this pair."), []

    r = 1.0 if x == y else temp[x].corr(temp[y])
    strength, direction = relationship_strength(r)

    fig = px.scatter(
        temp,
        x=x,
        y=y,
        opacity=0.55,
        title=f"{display_name(y)} vs {display_name(x)}",
        labels={x: axis_label(x), y: axis_label(y)},
        color_discrete_sequence=[C["blue"]],
    )
    fig = add_fit_line(fig, temp[x], temp[y])
    fig.update_traces(
        selector={"mode": "markers"},
        hovertemplate=f"{display_name(x)}=%{{x:,.2f}}<br>{display_name(y)}=%{{y:,.2f}}<extra></extra>",
    )
    fig = style_figure(fig, 720, "zoom")

    slope = None
    quartile_pct = None
    if x != y and temp[x].nunique() > 1:
        slope = np.polyfit(temp[x], temp[y], 1)[0]
        q25 = temp[x].quantile(0.25)
        q75 = temp[x].quantile(0.75)
        low = temp.loc[temp[x] <= q25, y].mean()
        high = temp.loc[temp[x] >= q75, y].mean()
        quartile_pct = ((high - low) / low * 100) if low else np.nan

    insights = [
        insight_card(
            "Correlation",
            f"{strength} {direction}: r = {r:.2f}",
            f"{display_name(x)} and {display_name(y)} show a {strength.lower()} {direction} linear relationship across {len(temp):,} complete observations.",
            C["blue"],
            "Correlation is association, not causation.",
        )
    ]

    if slope is not None:
        insights.append(
            insight_card(
                "Trend-line slope",
                f"{slope:,.2f}",
                f"For each one-unit increase in {display_name(x).lower()}, the fitted line changes {display_name(y).lower()} by about {slope:,.2f} units on average.",
                C["teal"],
            )
        )

    if quartile_pct is not None and np.isfinite(quartile_pct):
        word = "higher" if quartile_pct >= 0 else "lower"
        insights.append(
            insight_card(
                "High vs low comparison",
                f"{abs(quartile_pct):.1f}% {word}",
                f"The highest quarter of {display_name(x).lower()} has {abs(quartile_pct):.1f}% {word} average {display_name(y).lower()} than the lowest quarter.",
                C["pink"],
            )
        )

    return fig, insights


if SUNMOON_AVAILABLE:
    @app.callback(
        Output("sunrise-time", "children"),
        Output("sunset-time", "children"),
        Output("sun-window-readout", "children"),
        Output("sunrise-sky-marker", "style"),
        Output("sunset-sky-marker", "style"),
        Output("moon-visual", "children"),
        Output("temperature-readout", "children"),
        Output("thermo-fill", "style"),
        Output("sunmoon-insights", "children"),
        Output("sunmoon-chart", "figure"),
        Input("sunrise-slider", "value"),
        Input("sunset-slider", "value"),
        Input("moon-slider", "value"),
        Input("sunmoon-temp-slider", "value"),
    )
    def update_sunmoon(sunrise, sunset, moon, temperature):
        if sunset <= sunrise:
            return (
                hour_text(sunrise),
                hour_text(sunset),
                html.Div("Sunset must be later than sunrise", className="sun-readout-flex"),
                {"display": "none"},
                {"display": "none"},
                moon_glyph(moon),
                f"{temperature:.1f} °C",
                {"height": "0%", "background": "#D93025"},
                [],
                empty_figure("Choose a sunset later than sunrise."),
            )

        daylight = sunset - sunrise

        sun_readout = html.Div(
            [
                html.Span(f"Sunrise {hour_text(sunrise)}"),
                html.Span(f"{daylight:.2f} h daylight", className="sun-daylight-label"),
                html.Span(f"Sunset {hour_text(sunset)}"),
            ],
            className="sun-readout-flex",
        )

        # Position the visual markers across a 24-hour sky strip.
        sunrise_left = max(2, min(96, sunrise / 24 * 100))
        sunset_left = max(2, min(96, sunset / 24 * 100))
        sunrise_marker_style = {"left": f"{sunrise_left}%"}
        sunset_marker_style = {"left": f"{sunset_left}%"}

        temp_pct = max(0, min(100, (temperature + 5) / 40 * 100))
        thermo_style = {
            "height": f"{temp_pct}%",
            "background": "linear-gradient(180deg, #FF8A65 0%, #FF4D4F 62%, #D93025 100%)",
        }

        df = BIKE_DF[
            ["date", "bikes_hired", "moonphase", "sunrise_hour", "sunset_hour", "daylight_hours", "temp"]
        ].dropna().copy()

        moon_dist = np.minimum(
            np.abs(df["moonphase"] - moon),
            1 - np.abs(df["moonphase"] - moon),
        )

        df["similarity_score"] = (
            (moon_dist / 0.14) ** 2
            + ((df["sunrise_hour"] - sunrise) / 0.55) ** 2
            + ((df["sunset_hour"] - sunset) / 0.55) ** 2
            + ((df["temp"] - temperature) / 4.0) ** 2
        )

        n_match = min(150, len(df))
        matched = df.nsmallest(n_match, "similarity_score").copy()
        matched_avg = matched["bikes_hired"].mean()
        overall_avg = df["bikes_hired"].mean()
        diff = matched_avg - overall_avg
        diff_pct = diff / overall_avg * 100 if overall_avg else np.nan
        direction = "above" if diff >= 0 else "below"

        metrics = [
            metric_card(
                "Similar-day average",
                f"{matched_avg:,.2f}",
                f"Based on the {n_match} most similar historical days",
                C["blue"],
            ),
            metric_card(
                "Vs overall average",
                f"{abs(diff_pct):.1f}% {direction}",
                f"{abs(diff):,.0f} hires difference",
                C["teal"] if diff >= 0 else C["pink"],
            ),
            metric_card(
                "Your daylight",
                f"{daylight:.2f} h",
                f"Sunrise {hour_text(sunrise)} · sunset {hour_text(sunset)}",
                C["amber"],
            ),
            metric_card(
                "Your temperature",
                f"{temperature:.1f} °C",
                f"Moon phase {moon:.2f}",
                C["purple"],
            ),
        ]

        matched = matched.sort_values("date")
        fig = px.scatter(
            matched,
            x="date",
            y="bikes_hired",
            color="similarity_score",
            color_continuous_scale=["#F472B6", "#A78BFA", "#58A6FF", "#2DE2E6"],
            title="Most similar historical days",
            labels={
                "date": "Date",
                "bikes_hired": axis_label("bikes_hired"),
                "similarity_score": "Similarity score",
            },
            hover_data={
                "moonphase": ":.2f",
                "sunrise_hour": ":.2f",
                "sunset_hour": ":.2f",
                "temp": ":.2f",
                "similarity_score": ":.2f",
            },
        )
        fig.add_hline(
            y=matched_avg,
            line_dash="dash",
            line_color=C["teal"],
            annotation_text=f"Similar-day average {matched_avg:,.0f}",
        )
        fig.add_hline(
            y=overall_avg,
            line_dash="dot",
            line_color=C["amber"],
            annotation_text=f"Overall average {overall_avg:,.0f}",
        )
        fig = style_figure(fig, 640, "zoom")

        return (
            hour_text(sunrise),
            hour_text(sunset),
            sun_readout,
            sunrise_marker_style,
            sunset_marker_style,
            moon_glyph(moon),
            f"{temperature:.1f} °C",
            thermo_style,
            metrics,
            fig,
        )


@app.callback(
    Output("prediction-content", "children"),
    Input("refresh-predictions", "n_clicks"),
)
def refresh_predictions(_):
    if COEFF_ERROR:
        return dbc.Alert(COEFF_ERROR, color="danger", className="mt-3")

    sections = []

    try:
        history = open_meteo_history("London", "2026-01-01", "2026-01-07")
        history_pred = predict_hires(history, COEFFS)
        sections.append(
            prediction_section("Predicted bikes hired: 1–7 January 2026", history_pred)
        )
    except Exception as exc:
        sections.append(dbc.Alert(f"Historical prediction failed: {exc}", color="danger", className="mt-3"))

    try:
        # Open-Meteo counts today as one of forecast_days. Request six calendar
        # days, then remove today so this section contains five genuinely future days.
        forecast = open_meteo("London", 6)
        forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
        today_local = pd.Timestamp.now().normalize()
        forecast = (
            forecast.loc[forecast["date"] > today_local]
            .sort_values("date")
            .head(5)
            .copy()
        )

        forecast_pred = predict_hires(forecast, COEFFS)
        sections.append(
            prediction_section("Predicted bikes hired: next five future days", forecast_pred)
        )
    except Exception as exc:
        sections.append(dbc.Alert(f"Five-day forecast failed: {exc}", color="danger", className="mt-3"))

    return sections


@app.callback(
    Output({"type": "scenario-value", "term": ALL}, "children"),
    Input({"type": "scenario-slider", "term": ALL}, "value"),
    State({"type": "scenario-slider", "term": ALL}, "id"),
)
def show_scenario_values(values, ids):
    rendered = []
    for value, id_obj in zip(values, ids):
        term = id_obj["term"]
        unit = UNITS.get(term)
        rendered.append(f"{value:,.2f} {unit}" if unit else f"{value:,.2f}")
    return rendered


@app.callback(
    Output("scenario-prediction", "children"),
    Output("scenario-contribution-chart", "figure"),
    Input("scenario-day", "value"),
    Input("scenario-season", "value"),
    Input({"type": "scenario-slider", "term": ALL}, "value"),
    State({"type": "scenario-slider", "term": ALL}, "id"),
)
def scenario_prediction(day, season, values, ids):
    if COEFFS.empty:
        return dbc.Alert(COEFF_ERROR or "Model unavailable.", color="danger"), empty_figure("Model unavailable.")

    scenario = {
        id_obj["term"]: float(value)
        for value, id_obj in zip(values, ids)
    }
    scenario["day_of_week"] = day
    scenario["season_name"] = season
    scenario["weekend"] = day in ["Sat", "Sun"]

    scenario_df = pd.DataFrame([scenario])

    try:
        predicted = predict_hires(scenario_df, COEFFS)["predicted_bikes_hired"].iloc[0]
        contrib_df = model_contributions(scenario_df, COEFFS)
    except Exception as exc:
        return (
            dbc.Alert(f"Could not calculate scenario: {exc}", color="danger"),
            empty_figure("Scenario could not be calculated."),
        )

    pred_card = metric_card(
        "Scenario prediction",
        f"{predicted:,.2f}",
        f"Predicted bikes hired · {DAY_LABELS.get(day, day)} · {season}",
        C["blue"],
    )

    contrib_df["abs_contribution"] = contrib_df["contribution"].abs()

    intercept_rows = contrib_df[contrib_df["term"] == "Intercept"]
    effect_rows = (
        contrib_df[contrib_df["term"] != "Intercept"]
        .sort_values("abs_contribution", ascending=False)
        .head(10)
    )

    chart_df = pd.concat([intercept_rows, effect_rows], ignore_index=True)
    chart_df = chart_df.sort_values("contribution", ascending=True)

    fig = px.bar(
        chart_df,
        x="contribution",
        y="term",
        orientation="h",
        text=chart_df["contribution"].map(lambda x: f"{x:,.0f}"),
        title="Largest active model contributions",
        labels={
            "term": "",
            "contribution": "Contribution to predicted bikes hired",
        },
    )
    fig.update_traces(
        marker_color=C["purple"],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>Contribution=%{x:,.2f}<extra></extra>",
    )
    fig.add_vline(x=0, line_color=C["muted"], line_width=1)
    fig = style_figure(fig, 500, False)
    fig.update_layout(
        showlegend=False,
        margin={"l": 215, "r": 95, "t": 78, "b": 60},
        bargap=0.28,
    )
    fig.update_yaxes(automargin=True)

    return pred_card, fig


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)

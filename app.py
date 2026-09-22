import os
import io
import json
import re
import time
import base64
from typing import Any

import pandas as pd
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from openai import AzureOpenAI


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HDFC Bank AI Strategic Intelligence",
    page_icon="📊",
    layout="wide"
)
# ============================================================
# PASSWORD PROTECTION
# ============================================================

def get_app_password():
    try:
        if "APP_PASSWORD" in st.secrets:
            return str(st.secrets["APP_PASSWORD"])
    except Exception:
        pass

    # Local fallback
    return os.getenv("APP_PASSWORD", "5511")


def password_gate():
    """
    Prevent access to the application until the correct
    password is entered.
    """

    if st.session_state.get("authenticated", False):
        return True

    st.markdown(
        """
        <style>
        .login-container {
            max-width: 420px;
            margin: 100px auto 0 auto;
            padding: 35px;
            border-radius: 15px;
            border: 1px solid #e5e7eb;
            background: white;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }

        .login-title {
            text-align: center;
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .login-subtitle {
            text-align: center;
            color: #6b7280;
            margin-bottom: 25px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="login-container">
            <div class="login-title">
                📊 HDFC Bank
            </div>
            <div class="login-subtitle">
                AI Strategic Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter password",
        key="login_password"
    )

    if st.button(
        "🔐 Access Dashboard",
        use_container_width=True
    ):

        correct_password = get_app_password()

        if password == correct_password:

            st.session_state["authenticated"] = True

            # Remove password from session state
            st.session_state.pop("login_password", None)

            st.rerun()

        else:

            st.error("Incorrect password.")

    return False

# ============================================================
# AUTHENTICATION GATE
# ============================================================

if not password_gate():
    st.stop()


# ============================================================
# ENVIRONMENT
# ============================================================

# ============================================================
# ENVIRONMENT / STREAMLIT SECRETS
# ============================================================

load_dotenv()

def get_secret(name, default=""):
    """
    Get configuration from Streamlit Secrets first.
    Fall back to local .env/environment variables when running locally.
    """

    # Streamlit Cloud
    try:
        value = st.secrets.get(name)

        if value is not None:
            return str(value).strip()

    except Exception:
        pass

    # Local .env / environment variable
    return os.getenv(name, default).strip()


AZURE_OPENAI_API_KEY = get_secret(
    "AZURE_OPENAI_API_KEY"
)

AZURE_OPENAI_ENDPOINT = get_secret(
    "AZURE_OPENAI_ENDPOINT"
)

AZURE_OPENAI_API_VERSION = get_secret(
    "AZURE_OPENAI_API_VERSION",
    "2025-03-01-preview"
)

AZURE_OPENAI_DEPLOYMENT = get_secret(
    "AZURE_OPENAI_DEPLOYMENT",
    "gpt-4o"
)

# ============================================================
# AZURE OPENAI CLIENT
# ============================================================

client = None

if (
    AZURE_OPENAI_API_KEY
    and AZURE_OPENAI_ENDPOINT
):

    try:

        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION
        )

    except Exception as e:

        st.error(
            f"Azure OpenAI initialization failed: {e}"
        )


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_EXTENSIONS = [
    "csv",
    "xlsx",
    "xls"
]

MAX_TEXT_LENGTH = 1000

LLM_BATCH_SIZE = 50


# ============================================================
# SAFE VALUE
# ============================================================

def safe_value(value):

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except Exception:
        pass

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):

        if np.isnan(value) or np.isinf(value):
            return None

        return float(value)

    if isinstance(value, pd.Timestamp):

        return value.strftime(
            "%Y-%m-%d"
        )

    return value


# ============================================================
# CLEAN COLUMNS
# ============================================================

def clean_columns(df):

    df = df.copy()

    new_cols = []
    used = {}

    for col in df.columns:

        col = str(col).strip()

        if not col:

            col = "Unnamed"

        if col not in used:

            used[col] = 0
            new_cols.append(col)

        else:

            used[col] += 1

            new_cols.append(
                f"{col}_{used[col]}"
            )

    df.columns = new_cols

    return df


# ============================================================
# CLEAN DATAFRAME
# ============================================================

def clean_dataframe(df):

    df = df.copy()

    df = df.dropna(
        how="all"
    )

    df = df.dropna(
        axis=1,
        how="all"
    )

    df = clean_columns(
        df
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# LOAD FILE
# ============================================================

def load_file(uploaded_file):

    name = uploaded_file.name.lower()

    uploaded_file.seek(0)

    if name.endswith(".csv"):

        try:

            df = pd.read_csv(
                uploaded_file,
                low_memory=False
            )

        except UnicodeDecodeError:

            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                encoding="latin1",
                low_memory=False
            )

    elif name.endswith(
        (".xlsx", ".xls")
    ):

        df = pd.read_excel(
            uploaded_file
        )

    else:

        raise ValueError(
            "Unsupported file format."
        )

    return clean_dataframe(
        df
    )


# ============================================================
# LOAD MULTIPLE FILES
# ============================================================

def load_multiple_files(files):

    frames = []
    file_info = []

    for file in files:

        df = load_file(
            file
        )

        df["Source File"] = file.name

        frames.append(
            df
        )

        file_info.append({

            "File":
                file.name,

            "Rows":
                len(df),

            "Columns":
                len(df.columns)

        })

    if not frames:

        return pd.DataFrame(), []

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )

    return combined, file_info


# ============================================================
# NORMALIZE COLUMN NAME
# ============================================================

def normalize_name(value):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower()
    )


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    df,
    candidates
):

    if isinstance(
        candidates,
        str
    ):

        candidates = [
            candidates
        ]

    normalized = {
        normalize_name(col): col
        for col in df.columns
    }

    # Exact match
    for candidate in candidates:

        key = normalize_name(
            candidate
        )

        if key in normalized:

            return normalized[key]

    # Contains match
    for candidate in candidates:

        key = normalize_name(
            candidate
        )

        if not key:
            continue

        for col in df.columns:

            ncol = normalize_name(
                col
            )

            if (
                key in ncol
                or ncol in key
            ):

                return col

    return None


# ============================================================
# HDFC DATA MODEL
# ============================================================

def detect_hdfc_columns(df):

    return {

        "date":
            find_column(
                df,
                [
                    "date",
                    "published date",
                    "publication date",
                    "publish date",
                    "created date",
                    "datetime",
                    "timestamp",
                    "posted date",
                    "published"
                ]
            ),

        "text":
            find_column(
                df,
                [
                    "post text",
                    "post",
                    "content",
                    "description",
                    "message",
                    "title",
                    "headline",
                    "text",
                    "article",
                    "article text",
                    "body",
                    "mention"
                ]
            ),

        "sentiment":
            find_column(
                df,
                [
                    "sentiment",
                    "sentiment label",
                    "sentiment polarity",
                    "tone",
                    "sentiment type"
                ]
            ),

        "outlet":
            find_column(
                df,
                [
                    "source",
                    "outlet",
                    "publication",
                    "publisher",
                    "media",
                    "channel",
                    "source name",
                    "website",
                    "news source"
                ]
            ),

        "engagement":
            find_column(
                df,
                [
                    "engagement",
                    "engagements",
                    "total engagement",
                    "interaction",
                    "interactions",
                    "total interactions"
                ]
            ),

        "reach":
            find_column(
                df,
                [
                    "reach",
                    "potential reach",
                    "impressions",
                    "views",
                    "audience"
                ]
            ),

        "author":
            find_column(
                df,
                [
                    "author",
                    "journalist",
                    "creator",
                    "user",
                    "author name",
                    "reporter",
                    "writer",
                    "journalist name"
                ]
            ),

        "url":
            find_column(
                df,
                [
                    "url",
                    "link",
                    "article url",
                    "post url"
                ]
            )
    }


# ============================================================
# SENTIMENT NORMALIZATION
# ============================================================

def normalize_sentiment(value):

    value = str(
        value
    ).lower().strip()

    if not value:

        return "Unknown"

    if any(
        x in value
        for x in [
            "positive",
            "pos",
            "favorable",
            "favourable"
        ]
    ):

        return "Positive"

    if any(
        x in value
        for x in [
            "negative",
            "neg",
            "unfavorable",
            "unfavourable"
        ]
    ):

        return "Negative"

    if any(
        x in value
        for x in [
            "neutral",
            "neu"
        ]
    ):

        return "Neutral"

    return "Unknown"


# ============================================================
# HDFC RELEVANCE
# ============================================================

def is_hdfc_relevant(text):

    text = str(
        text
    ).lower()

    if not text.strip():

        return False

    strong_terms = [

        "hdfc bank",

        "hdfcbank",

        "hdfcbank.com",

        "hdfc bank ltd",

        "hdfc bank limited"

    ]

    if any(
        term in text
        for term in strong_terms
    ):

        return True

    context_terms = [

        "hdfc netbanking",

        "hdfc net banking",

        "hdfc upi",

        "hdfc credit card",

        "hdfc debit card",

        "hdfc loan",

        "hdfc emi",

        "hdfc branch",

        "hdfc customer",

        "hdfc account",

        "hdfc app",

        "hdfc payment"

    ]

    if any(
        term in text
        for term in context_terms
    ):

        return True

    return False


# ============================================================
# PREPARE HDFC DATA
# ============================================================

def prepare_hdfc_data(df):

    df = df.copy()

    cols = detect_hdfc_columns(
        df
    )

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    if cols["date"]:

        df["_date"] = pd.to_datetime(
            df[cols["date"]],
            errors="coerce"
        )

    else:

        df["_date"] = pd.NaT

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    if cols["text"]:

        df["_text"] = (
            df[cols["text"]]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    else:

        # Fallback:
        # combine likely text columns

        possible_text = []

        for col in [
            "title",
            "headline",
            "description",
            "content",
            "message"
        ]:

            found = find_column(
                df,
                [col]
            )

            if found:

                possible_text.append(
                    df[found]
                    .fillna("")
                    .astype(str)
                )

        if possible_text:

            combined = possible_text[0]

            for series in possible_text[1:]:

                combined = (
                    combined
                    + " "
                    + series
                )

            df["_text"] = (
                combined
                .str.strip()
            )

        else:

            df["_text"] = ""

    # --------------------------------------------------------
    # OUTLET
    # --------------------------------------------------------

    if cols["outlet"]:

        df["_outlet"] = (
            df[cols["outlet"]]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    else:

        df["_outlet"] = "Unknown"

    # --------------------------------------------------------
    # AUTHOR
    # --------------------------------------------------------

    if cols["author"]:

        df["_author"] = (
            df[cols["author"]]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    else:

        df["_author"] = ""

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    if cols["sentiment"]:

        df["_sentiment_raw"] = (
            df[cols["sentiment"]]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

    else:

        df["_sentiment_raw"] = ""

    df["_sentiment"] = (
        df["_sentiment_raw"]
        .apply(
            normalize_sentiment
        )
    )

    # --------------------------------------------------------
    # ENGAGEMENT
    # --------------------------------------------------------

    if cols["engagement"]:

        df["_engagement"] = pd.to_numeric(
            df[cols["engagement"]],
            errors="coerce"
        ).fillna(0)

    else:

        df["_engagement"] = 0

    # --------------------------------------------------------
    # REACH
    # --------------------------------------------------------

    if cols["reach"]:

        df["_reach"] = pd.to_numeric(
            df[cols["reach"]],
            errors="coerce"
        ).fillna(0)

    else:

        df["_reach"] = 0

    # --------------------------------------------------------
    # RELEVANCE
    # --------------------------------------------------------

    df["_hdfc_relevant"] = (
        df["_text"]
        .apply(
            is_hdfc_relevant
        )
    )

    # --------------------------------------------------------
    # ROW ID
    # --------------------------------------------------------

    df["_row_id"] = np.arange(
        len(df)
    )

    return df, cols


# ============================================================
# GET RELEVANT DATA
# ============================================================

def get_relevant_data(df):

    relevant = df[
        df["_hdfc_relevant"]
    ].copy()

    relevant = relevant[
        relevant["_text"].str.len() > 0
    ]

    return relevant.reset_index(
        drop=True
    )


# ============================================================
# ADD PERIODS
# ============================================================

def add_periods(df):

    df = df.copy()

    df["_week"] = (
        df["_date"]
        .dt.to_period("W-SUN")
        .astype(str)
    )

    df["_month"] = (
        df["_date"]
        .dt.to_period("M")
        .astype(str)
    )

    df["_quarter"] = (
        df["_date"]
        .dt.to_period("Q")
        .astype(str)
    )

    return df


# ============================================================
# SENTIMENT SUMMARY
# ============================================================

def sentiment_summary(df):

    total = len(df)

    if total == 0:

        return {

            "total_mentions": 0,

            "positive": 0,

            "neutral": 0,

            "negative": 0,

            "unknown": 0,

            "positive_pct": 0,

            "negative_pct": 0,

            "neutral_pct": 0,

            "net_sentiment": 0

        }

    positive = int(
        (
            df["_sentiment"]
            == "Positive"
        ).sum()
    )

    neutral = int(
        (
            df["_sentiment"]
            == "Neutral"
        ).sum()
    )

    negative = int(
        (
            df["_sentiment"]
            == "Negative"
        ).sum()
    )

    unknown = (
        total
        - positive
        - neutral
        - negative
    )

    positive_pct = (
        positive
        / total
        * 100
    )

    negative_pct = (
        negative
        / total
        * 100
    )

    neutral_pct = (
        neutral
        / total
        * 100
    )

    net = (
        positive_pct
        - negative_pct
    )

    return {

        "total_mentions":
            total,

        "positive":
            positive,

        "neutral":
            neutral,

        "negative":
            negative,

        "unknown":
            unknown,

        "positive_pct":
            round(
                positive_pct,
                2
            ),

        "negative_pct":
            round(
                negative_pct,
                2
            ),

        "neutral_pct":
            round(
                neutral_pct,
                2
            ),

        "net_sentiment":
            round(
                net,
                2
            )
    }


# ============================================================
# QUARTERLY RESILIENCE
# ============================================================

def calculate_quarterly_resilience(
    df
):

    temp = df[
        df["_date"].notna()
    ].copy()

    if temp.empty:

        return []

    grouped = []

    for quarter, group in (
        temp
        .groupby("_quarter")
    ):

        summary = sentiment_summary(
            group
        )

        grouped.append({

            "quarter":
                quarter,

            "mentions":
                summary[
                    "total_mentions"
                ],

            "positive_pct":
                summary[
                    "positive_pct"
                ],

            "negative_pct":
                summary[
                    "negative_pct"
                ],

            "neutral_pct":
                summary[
                    "neutral_pct"
                ],

            "net_sentiment":
                summary[
                    "net_sentiment"
                ]

        })

    grouped = sorted(
        grouped,
        key=lambda x:
            x["quarter"]
    )

    # --------------------------------------------------------
    # DERIVED DIRECTIONAL INDEX
    # --------------------------------------------------------

    for item in grouped:

        item[
            "resilience_index"
        ] = round(

            50
            + (
                item[
                    "net_sentiment"
                ]
                / 2
            ),

            1
        )

    return grouped


# ============================================================
# NARRATIVES
# ============================================================

NARRATIVES = [

    "Customer service",

    "Governance / regulatory",

    "Fees / charges",

    "Fraud / card disputes",

    "UPI / digital banking",

    "Leadership",

    "Campaign / innovation",

    "Rural / inclusion",

    "Digital outage",

    "Other"
]


NARRATIVE_KEYWORDS = {

    "Customer service": [

        "customer service",

        "customer care",

        "complaint",

        "complaints",

        "support",

        "branch staff",

        "service issue",

        "poor service",

        "customer experience",

        "customer support"

    ],

    "Governance / regulatory": [

        "rbi",

        "reserve bank",

        "regulator",

        "regulatory",

        "compliance",

        "governance",

        "penalty",

        "fine",

        "audit",

        "regulatory action"

    ],

    "Fees / charges": [

        "fee",

        "fees",

        "charge",

        "charges",

        "hidden charge",

        "hidden charges",

        "annual fee",

        "processing fee",

        "service charge",

        "convenience fee"

    ],

    "Fraud / card disputes": [

        "fraud",

        "scam",

        "credit card fraud",

        "debit card fraud",

        "card fraud",

        "unauthorized transaction",

        "unauthorised transaction",

        "cyber fraud",

        "fraudulent transaction"

    ],

    "UPI / digital banking": [

        "upi",

        "netbanking",

        "net banking",

        "mobile banking",

        "digital banking",

        "payment failed",

        "transaction failed",

        "digital payment",

        "mobile app"

    ],

    "Leadership": [

        "ceo",

        "md",

        "managing director",

        "executive",

        "leadership",

        "senior management",

        "chief executive"

    ],

    "Campaign / innovation": [

        "campaign",

        "launch",

        "innovation",

        "initiative",

        "partnership",

        "product launch",

        "new product",

        "technology"

    ],

    "Rural / inclusion": [

        "rural",

        "village",

        "farmer",

        "financial inclusion",

        "inclusion",

        "semi urban",

        "semi-urban",

        "rural banking"

    ],

    "Digital outage": [

        "outage",

        "server down",

        "app not working",

        "netbanking down",

        "technical issue",

        "service unavailable",

        "system down",

        "banking outage"

    ]

}


# ============================================================
# LOCAL NARRATIVE CLASSIFICATION
# ============================================================

def classify_narrative_local(
    text
):

    text = str(
        text
    ).lower()

    scores = {}

    for narrative, keywords in (
        NARRATIVE_KEYWORDS.items()
    ):

        score = 0

        for keyword in keywords:

            if keyword in text:

                score += 1

        scores[
            narrative
        ] = score

    best = max(
        scores,
        key=scores.get
    )

    if scores[best] == 0:

        return "Other"

    return best


# ============================================================
# LOCAL NARRATIVE METRICS
# ============================================================

def calculate_narratives(
    df
):

    temp = df.copy()

    temp["_narrative"] = (
        temp["_text"]
        .apply(
            classify_narrative_local
        )
    )

    rows = []

    for narrative, group in (
        temp.groupby(
            "_narrative"
        )
    ):

        summary = sentiment_summary(
            group
        )

        rows.append({

            "narrative":
                narrative,

            "mentions":
                summary[
                    "total_mentions"
                ],

            "positive":
                summary[
                    "positive"
                ],

            "neutral":
                summary[
                    "neutral"
                ],

            "negative":
                summary[
                    "negative"
                ],

            "positive_pct":
                summary[
                    "positive_pct"
                ],

            "negative_pct":
                summary[
                    "negative_pct"
                ],

            "net_sentiment":
                summary[
                    "net_sentiment"
                ]

        })

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result

    total = result[
        "mentions"
    ].sum()

    if total:

        result[
            "pull_through_pct"
        ] = (
            result[
                "mentions"
            ]
            / total
            * 100
        ).round(2)

    else:

        result[
            "pull_through_pct"
        ] = 0

    return result.sort_values(
        "mentions",
        ascending=False
    ).reset_index(
        drop=True
    )


# ============================================================
# NARRATIVE WIN / LOSS
# ============================================================

def narrative_win_loss(
    narrative_df
):

    if narrative_df.empty:

        return []

    output = []

    for _, row in (
        narrative_df.iterrows()
    ):

        net = float(
            row[
                "net_sentiment"
            ]
        )

        negative = int(
            row[
                "negative"
            ]
        )

        positive = int(
            row[
                "positive"
            ]
        )

        mentions = int(
            row[
                "mentions"
            ]
        )

        if (
            positive > negative
            and net > 5
        ):

            outcome = "Won"

        elif (
            negative > positive
            and net < -5
        ):

            outcome = "Losing"

        else:

            outcome = "Contested"

        output.append({

            "narrative":
                row[
                    "narrative"
                ],

            "outcome":
                outcome,

            "mentions":
                mentions,

            "positive":
                positive,

            "negative":
                negative,

            "net_sentiment":
                net,

            "pull_through_pct":
                float(
                    row[
                        "pull_through_pct"
                    ]
                )

        })

    return output


# ============================================================
# OUTLET WATCHLIST
# ============================================================

def calculate_outlet_watchlist(
    df
):

    temp = df.copy()

    if temp.empty:

        return pd.DataFrame()

    rows = []

    for outlet, group in (
        temp.groupby(
            "_outlet"
        )
    ):

        if outlet in [
            "",
            "Unknown",
            "nan",
            "None"
        ]:

            continue

        summary = sentiment_summary(
            group
        )

        rows.append({

            "outlet":
                outlet,

            "mentions":
                summary[
                    "total_mentions"
                ],

            "positive_pct":
                summary[
                    "positive_pct"
                ],

            "negative_pct":
                summary[
                    "negative_pct"
                ],

            "net_sentiment":
                summary[
                    "net_sentiment"
                ]

        })

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result

    # --------------------------------------------------------
    # WARMING / COOLING / STABLE
    # --------------------------------------------------------

    trend_values = []

    for outlet in result[
        "outlet"
    ]:

        outlet_df = temp[
            temp["_outlet"]
            == outlet
        ].copy()

        outlet_df = outlet_df[
            outlet_df["_date"].notna()
        ].sort_values(
            "_date"
        )

        if len(outlet_df) < 4:

            trend = "Stable"

        else:

            midpoint = (
                len(outlet_df)
                // 2
            )

            previous = (
                outlet_df
                .iloc[:midpoint]
            )

            current = (
                outlet_df
                .iloc[midpoint:]
            )

            prev_net = (
                sentiment_summary(
                    previous
                )[
                    "net_sentiment"
                ]
            )

            curr_net = (
                sentiment_summary(
                    current
                )[
                    "net_sentiment"
                ]
            )

            delta = (
                curr_net
                - prev_net
            )

            if delta >= 5:

                trend = "Warming"

            elif delta <= -5:

                trend = "Cooling"

            else:

                trend = "Stable"

        trend_values.append(
            trend
        )

    result[
        "trend"
    ] = trend_values

    return result.sort_values(
        "mentions",
        ascending=False
    ).reset_index(
        drop=True
    )


# ============================================================
# SILENCE RISK FLAGS
# ============================================================

def calculate_silence_risks(
    df
):

    risks = []

    total = len(df)

    if total == 0:

        return risks

    # --------------------------------------------------------
    # FEES
    # --------------------------------------------------------

    fee_mask = (
        df["_text"]
        .str.lower()
        .str.contains(
            r"\bfee\b|\bfees\b|\bcharge\b|\bcharges\b",
            regex=True,
            na=False
        )
    )

    fee_count = int(
        fee_mask.sum()
    )

    if fee_count:

        fee_pct = (
            fee_count
            / total
            * 100
        )

        risks.append({

            "risk":
                "Fee / charge narrative",

            "mentions":
                fee_count,

            "percentage":
                round(
                    fee_pct,
                    2
                ),

            "signal":
                "Monitor",

            "basis":
                "Actual fee/charge mentions in HDFC-relevant records."

        })

    # --------------------------------------------------------
    # OUTAGE
    # --------------------------------------------------------

    outage_mask = (
        df["_text"]
        .str.lower()
        .str.contains(
            r"outage|server down|technical issue|app not working|system down",
            regex=True,
            na=False
        )
    )

    outage_count = int(
        outage_mask.sum()
    )

    if outage_count:

        risks.append({

            "risk":
                "Digital outage / service disruption",

            "mentions":
                outage_count,

            "percentage":
                round(
                    outage_count
                    / total
                    * 100,
                    2
                ),

            "signal":
                "Monitor",

            "basis":
                "Actual outage/service disruption mentions."

        })

    # --------------------------------------------------------
    # FRAUD
    # --------------------------------------------------------

    fraud_mask = (
        df["_text"]
        .str.lower()
        .str.contains(
            r"fraud|scam|unauthorized transaction|unauthorised transaction",
            regex=True,
            na=False
        )
    )

    fraud_count = int(
        fraud_mask.sum()
    )

    if fraud_count:

        risks.append({

            "risk":
                "Fraud / scam narrative",

            "mentions":
                fraud_count,

            "percentage":
                round(
                    fraud_count
                    / total
                    * 100,
                    2
                ),

            "signal":
                "Monitor",

            "basis":
                "Actual fraud/scam mentions."

        })

    # --------------------------------------------------------
    # NEGATIVE SENTIMENT
    # --------------------------------------------------------

    negative = int(
        (
            df["_sentiment"]
            == "Negative"
        ).sum()
    )

    negative_pct = (
        negative
        / total
        * 100
    )

    if negative_pct >= 30:

        risks.append({

            "risk":
                "Elevated negative conversation",

            "mentions":
                negative,

            "percentage":
                round(
                    negative_pct,
                    2
                ),

            "signal":
                "Elevated",

            "basis":
                "Negative mentions as a percentage of relevant conversation."

        })

    return risks


# ============================================================
# WEEKLY NARRATIVE PULL-THROUGH
# ============================================================

def calculate_weekly_pullthrough(
    df
):

    temp = df[
        df["_date"].notna()
    ].copy()

    if temp.empty:

        return pd.DataFrame()

    temp[
        "_narrative"
    ] = (
        temp["_text"]
        .apply(
            classify_narrative_local
        )
    )

    weekly = (
        temp
        .groupby(
            [
                "_week",
                "_narrative"
            ]
        )
        .size()
        .reset_index(
            name="mentions"
        )
    )

    pivot = (
        weekly
        .pivot(
            index="_week",
            columns="_narrative",
            values="mentions"
        )
        .fillna(0)
    )

    pivot[
        "Total"
    ] = pivot.sum(
        axis=1
    )

    for narrative in pivot.columns:

        if narrative == "Total":
            continue

        pivot[
            f"{narrative} %"
        ] = np.where(

            pivot["Total"] > 0,

            pivot[narrative]
            / pivot["Total"]
            * 100,

            0

        )

    return pivot.reset_index()


# ============================================================
# IMAGE TO DATA URL
# ============================================================

def image_to_data_url(
    uploaded_image
):

    if uploaded_image is None:

        return None

    image_bytes = (
        uploaded_image
        .getvalue()
    )

    mime_type = (
        uploaded_image.type
        or "image/png"
    )

    encoded = (
        base64.b64encode(
            image_bytes
        )
        .decode(
            "utf-8"
        )
    )

    return (
        f"data:{mime_type};base64,{encoded}"
    )


# ============================================================
# ANALYZE REFERENCE SCREENSHOT
# ============================================================

def analyze_reference_screenshot(
    screenshot,
    data_columns=None
):

    if client is None:

        raise RuntimeError(
            "Azure OpenAI client is not configured."
        )

    image_url = (
        image_to_data_url(
            screenshot
        )
    )

    columns_text = ""

    if data_columns is not None and len(data_columns) > 0:

        columns_text = json.dumps(
            list(data_columns),
            indent=2
        )

    prompt = f"""
You are analyzing a reference screenshot for an HDFC Bank
strategic media intelligence dashboard.

The screenshot is ONLY a design and analytical reference.

IMPORTANT:
- Do NOT copy numerical values from the screenshot.
- Do NOT treat screenshot numbers as actual data.
- Do NOT invent missing metrics.
- The uploaded Excel/CSV data is the source of truth.

Identify the analytical structure shown in the screenshot.

Return ONLY valid JSON.

Format:

{{
    "report_title": "",
    "sections": [],
    "metrics": [],
    "charts": [],
    "tables": [],
    "narratives": [],
    "insights_expected": []
}}

For every metric identify:

- metric name
- apparent purpose
- calculation concept if visible
- required data fields if visible

For every chart identify:

- chart type
- x axis
- y axis
- grouping/category
- time dimension

Available dataset columns:

{columns_text}
"""

    response = client.chat.completions.create(

        model=AZURE_OPENAI_DEPLOYMENT,

        messages=[

            {
                "role":
                    "system",

                "content":
                    (
                        "You are a strict business "
                        "intelligence dashboard analyst. "
                        "Never invent data."
                    )
            },

            {
                "role":
                    "user",

                "content": [

                    {
                        "type":
                            "text",

                        "text":
                            prompt
                    },

                    {
                        "type":
                            "image_url",

                        "image_url": {

                            "url":
                                image_url

                        }

                    }

                ]
            }

        ],

        temperature=0
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    content = re.sub(
        r"```json|```",
        "",
        content,
        flags=re.I
    ).strip()

    return json.loads(
        content
    )


# ============================================================
# AZURE OPENAI CLASSIFICATION
# ============================================================

def openai_classify_batch(
    records
):

    if client is None:

        raise RuntimeError(
            "Azure OpenAI is not configured."
        )

    payload = []

    for item in records:

        payload.append({

            "id":
                item["id"],

            "text":
                str(
                    item["text"]
                )[
                    :MAX_TEXT_LENGTH
                ]

        })

    prompt = f"""
You are classifying HDFC Bank media/social conversation.

Classify every record into exactly ONE primary narrative.

Allowed narratives:

1. Customer service
2. Governance / regulatory
3. Fees / charges
4. Fraud / card disputes
5. UPI / digital banking
6. Leadership
7. Campaign / innovation
8. Rural / inclusion
9. Digital outage
10. Other

Also classify sentiment only when clearly supported
by the supplied text.

Do NOT invent facts.

Return ONLY valid JSON.

Format:

[
    {{
        "id": 1,
        "narrative": "Customer service",
        "sentiment": "Negative"
    }}
]

Records:

{json.dumps(
    payload,
    ensure_ascii=False
)}
"""

    response = client.chat.completions.create(

        model=AZURE_OPENAI_DEPLOYMENT,

        messages=[

            {
                "role":
                    "system",

                "content":
                    (
                        "You are a strict business-data "
                        "classifier. Never invent facts."
                    )
            },

            {
                "role":
                    "user",

                "content":
                    prompt
            }

        ],

        temperature=0
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    content = re.sub(
        r"```json|```",
        "",
        content,
        flags=re.I
    ).strip()

    return json.loads(
        content
    )


# ============================================================
# AI CLASSIFICATION
# ============================================================

def classify_with_openai(
    df,
    max_records=1000
):

    temp = df.copy()

    temp = temp[
        temp["_text"].str.len() > 10
    ]

    temp = temp.drop_duplicates(
        subset=[
            "_text"
        ]
    )

    temp = temp.head(
        max_records
    ).copy()

    temp[
        "_ai_narrative"
    ] = ""

    temp[
        "_ai_sentiment"
    ] = ""

    records = []

    for idx, row in (
        temp.iterrows()
    ):

        records.append({

            "id":
                int(idx),

            "text":
                row["_text"]

        })

    results = []

    for start in range(
        0,
        len(records),
        LLM_BATCH_SIZE
    ):

        batch = records[
            start:
            start
            + LLM_BATCH_SIZE
        ]

        batch_results = (
            openai_classify_batch(
                batch
            )
        )

        if isinstance(
            batch_results,
            list
        ):

            results.extend(
                batch_results
            )

        time.sleep(
            0.2
        )

    result_map = {

        int(item["id"]):
            item

        for item in results

        if isinstance(
            item,
            dict
        )
        and "id" in item

    }

    for idx in temp.index:

        result = (
            result_map.get(
                int(idx)
            )
        )

        if result:

            temp.loc[
                idx,
                "_ai_narrative"
            ] = result.get(
                "narrative",
                "Other"
            )

            temp.loc[
                idx,
                "_ai_sentiment"
            ] = result.get(
                "sentiment",
                ""
            )

    return temp


# ============================================================
# APPLY AI CLASSIFICATION TO DATA
# ============================================================

def apply_ai_classification(
    df,
    max_records=1000
):

    if client is None:

        return df

    classified = (
        classify_with_openai(
            df,
            max_records
        )
    )

    if classified.empty:

        return df

    output = df.copy()

    ai_map = (
        classified
        [
            [
                "_text",
                "_ai_narrative",
                "_ai_sentiment"
            ]
        ]
        .drop_duplicates(
            "_text"
        )
        .set_index(
            "_text"
        )
    )

    output[
        "_ai_narrative"
    ] = output[
        "_text"
    ].map(
        ai_map[
            "_ai_narrative"
        ]
    )

    output[
        "_ai_sentiment"
    ] = output[
        "_text"
    ].map(
        ai_map[
            "_ai_sentiment"
        ]
    )

    output[
        "_ai_narrative"
    ] = (
        output[
            "_ai_narrative"
        ]
        .fillna("")
    )

    output[
        "_ai_sentiment"
    ] = (
        output[
            "_ai_sentiment"
        ]
        .fillna("")
    )

    # Use AI narrative when available
    output[
        "_narrative"
    ] = np.where(

        output[
            "_ai_narrative"
        ].str.len() > 0,

        output[
            "_ai_narrative"
        ],

        output[
            "_text"
        ].apply(
            classify_narrative_local
        )

    )

    # Use AI sentiment only if the dataset has
    # no reliable sentiment
    if (
        output[
            "_sentiment"
        ]
        == "Unknown"
    ).all():

        output[
            "_sentiment"
        ] = (
            output[
                "_ai_sentiment"
            ]
            .apply(
                normalize_sentiment
            )
        )

    return output


# ============================================================
# NARRATIVES USING AI DATA
# ============================================================

def calculate_narratives_from_column(
    df
):

    if "_narrative" not in df.columns:

        temp = df.copy()

        temp[
            "_narrative"
        ] = temp[
            "_text"
        ].apply(
            classify_narrative_local
        )

    else:

        temp = df.copy()

    rows = []

    for narrative, group in (
        temp.groupby(
            "_narrative"
        )
    ):

        summary = sentiment_summary(
            group
        )

        rows.append({

            "narrative":
                narrative,

            "mentions":
                summary[
                    "total_mentions"
                ],

            "positive":
                summary[
                    "positive"
                ],

            "neutral":
                summary[
                    "neutral"
                ],

            "negative":
                summary[
                    "negative"
                ],

            "positive_pct":
                summary[
                    "positive_pct"
                ],

            "negative_pct":
                summary[
                    "negative_pct"
                ],

            "net_sentiment":
                summary[
                    "net_sentiment"
                ]

        })

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result

    total = result[
        "mentions"
    ].sum()

    result[
        "pull_through_pct"
    ] = np.where(

        total > 0,

        result[
            "mentions"
        ]
        / total
        * 100,

        0

    )

    result[
        "pull_through_pct"
    ] = result[
        "pull_through_pct"
    ].round(2)

    return result.sort_values(
        "mentions",
        ascending=False
    ).reset_index(
        drop=True
    )


# ============================================================
# WEEKLY PULL-THROUGH USING AI NARRATIVES
# ============================================================

def calculate_weekly_pullthrough_ai(
    df
):

    temp = df[
        df["_date"].notna()
    ].copy()

    if temp.empty:

        return pd.DataFrame()

    if "_narrative" not in temp.columns:

        temp[
            "_narrative"
        ] = temp[
            "_text"
        ].apply(
            classify_narrative_local
        )

    weekly = (
        temp
        .groupby(
            [
                "_week",
                "_narrative"
            ]
        )
        .size()
        .reset_index(
            name="mentions"
        )
    )

    pivot = (
        weekly
        .pivot(
            index="_week",
            columns="_narrative",
            values="mentions"
        )
        .fillna(0)
    )

    pivot[
        "Total"
    ] = pivot.sum(
        axis=1
    )

    return pivot.reset_index()


# ============================================================
# DATASET SUMMARY
# ============================================================

def build_summary(
    df
):

    summary = sentiment_summary(
        df
    )

    dates = df[
        "_date"
    ].dropna()

    if len(dates):

        start_date = (
            dates.min()
            .strftime(
                "%Y-%m-%d"
            )
        )

        end_date = (
            dates.max()
            .strftime(
                "%Y-%m-%d"
            )
        )

    else:

        start_date = None
        end_date = None

    return {

        "total_rows":
            len(df),

        "date_start":
            start_date,

        "date_end":
            end_date,

        "sentiment":
            summary,

        "outlets":
            int(
                df[
                    "_outlet"
                ].nunique()
            )

    }


# ============================================================
# STRATEGIC REPORT
# ============================================================

def generate_report(

    actual_data,

    narrative_df,

    win_loss,

    outlet_df,

    risks,

    quarterly,

    pullthrough,

    screenshot_analysis=None

):

    if client is None:

        raise RuntimeError(
            "Azure OpenAI is not configured."
        )

    payload = {

        "dataset_summary":
            build_summary(
                actual_data
            ),

        "narratives":
            narrative_df.to_dict(
                orient="records"
            ),

        "narrative_win_loss":
            win_loss,

        "outlet_watchlist":
            outlet_df.head(
                20
            ).to_dict(
                orient="records"
            )
            if not outlet_df.empty
            else [],

        "silence_risk_flags":
            risks,

        "quarterly_resilience":
            quarterly,

        "weekly_pullthrough":
            pullthrough.to_dict(
                orient="records"
            )
            if not pullthrough.empty
            else [],

        "reference_structure":
            screenshot_analysis
            if screenshot_analysis
            else {}

    }

    prompt = f"""

Create a factual HDFC Bank strategic media/social
listening review.

==================================================
SOURCE OF TRUTH
==================================================

The uploaded Excel/CSV data and the deterministic
Python calculations are the ONLY source of numerical truth.

The screenshot is ONLY a reference for:

- layout
- section structure
- metric types
- chart types
- analytical expectations

NEVER copy numbers from the screenshot.

NEVER invent numbers.

==================================================
ACTUAL CALCULATED DATA
==================================================

{json.dumps(
    payload,
    indent=2,
    ensure_ascii=False,
    default=str
)}

==================================================
REFERENCE SCREENSHOT STRUCTURE
==================================================

{json.dumps(
    screenshot_analysis
    if screenshot_analysis
    else {},
    indent=2,
    ensure_ascii=False,
    default=str
)}

==================================================
REQUIRED OUTPUT
==================================================

# Monthly Strategic Review

Write a concise senior leadership review.

---

## 1. Reputation Resilience Index

Use actual quarterly data.

Show:

| Quarter | Mentions | Positive % | Negative % | Neutral % | Net Sentiment | Derived Index |
|---|---:|---:|---:|---:|---:|---:|

The derived directional index is:

50 + (Net Sentiment / 2)

Do NOT call this an official HDFC Bank reputation index.

Call it:

"Derived Directional Resilience Index"

If fewer than four quarters are available,
state that clearly.

Do not fabricate quarters.

---

## 2. Narrative Win / Loss Ledger

Use:

| Narrative | Outcome | Mentions | Pull-through % | Net Sentiment | Evidence |
|---|---|---:|---:|---:|---|

Allowed outcomes:

Won
Contested
Losing

The classification is based on the supplied
positive/negative narrative data.

Do not claim causality.

---

## 3. Emerging Risks & Pre-Flight Learnings

Use actual risk flags.

For every risk include:

- risk
- actual mentions
- actual percentage
- why it matters
- monitoring/action implication

Do not invent evidence.

---

## 4. Journalist & Outlet Watchlist

Use:

| Voice / Outlet | Trend | Mentions | Positive % | Negative % | Net Sentiment |
|---|---|---:|---:|---:|---:|

Trend values:

▲ Warming
▼ Cooling
▬ Stable

Only use the calculated trend.

---

## 5. Silence-Risk Flags

Use:

| Risk | Actual Signal | Evidence | Action |
|---|---|---|---|

Use actual counts and percentages.

---

## 6. Weekly Narrative Pull-Through

Explain how the actual narrative mix changes over time.

Do not use screenshot percentages.

Use actual weekly data.

---

## 7. Leadership Takeaways

Give 4-6 concise observations.

Every quantitative observation must come from
the supplied data.

Do not introduce new numerical values.

---

## Data Validation

Include:

- total relevant mentions
- date range
- positive
- neutral
- negative
- unknown
- net sentiment
- number of narratives
- number of outlets
- number of risk flags
- whether four quarters are available
- whether a reference screenshot was supplied

==================================================
STYLE
==================================================

Senior leadership language.

Concise.

Evidence-based.

Numbers first.

No fabricated claims.

No screenshot numbers.

No unsupported causal claims.

"""

    response = client.chat.completions.create(

        model=AZURE_OPENAI_DEPLOYMENT,

        messages=[

            {
                "role":
                    "system",

                "content":
                    """
You are a senior media intelligence analyst.

Use only supplied calculated data.

Never fabricate numerical values.

Never copy numerical values from a screenshot.

The screenshot is a structural reference only.
"""
            },

            {
                "role":
                    "user",

                "content":
                    prompt
            }

        ],

        temperature=0
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# SIDEBAR
# ============================================================

# with st.sidebar:

#     st.header(
#         "⚙️ Azure OpenAI"
#     )

#     st.write(
#         f"Deployment: "
#         f"`{AZURE_OPENAI_DEPLOYMENT}`"
#     )

#     st.write(
#         f"API Version: "
#         f"`{AZURE_OPENAI_API_VERSION}`"
#     )

#     st.write(
#         f"Endpoint: "
#         f"`{AZURE_OPENAI_ENDPOINT}`"
#     )

#     if client is not None:

#         st.success(
#             "✅ Azure OpenAI configured"
#         )

#     else:

#         st.error(
#             "❌ Azure OpenAI not configured"
#         )


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 HDFC Bank AI Strategic Data Intelligence"
)

st.caption(
    "Actual dataset → Python calculations → "
    "Azure GPT-4o semantic analysis → Strategic Review"
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.subheader(
    "📥 Upload Inputs"
)

col1, col2 = st.columns(
    2
)


with col1:

    files = st.file_uploader(

        "📁 Upload Excel / CSV",

        type=
            SUPPORTED_EXTENSIONS,

        accept_multiple_files=True,

        help=(
            "Upload the actual HDFC Bank "
            "media/social listening dataset."
        )

    )


with col2:

    screenshot = st.file_uploader(

        "🖼️ Upload Reference Screenshot",

        type=[
            "png",
            "jpg",
            "jpeg"
        ],

        accept_multiple_files=False,

        help=(
            "Upload the reference screenshot "
            "whose analytical structure you want "
            "the strategic review to follow."
        )

    )


# ============================================================
# SCREENSHOT PREVIEW
# ============================================================

if screenshot:

    st.success(
        f"Reference screenshot uploaded: "
        f"{screenshot.name}"
    )

    with st.expander(
        "🖼️ Preview Reference Screenshot",
        expanded=True
    ):

        st.image(
            screenshot,
            caption=
                "Reference only — screenshot numbers "
                "will NOT be used as actual data.",
            use_container_width=True
        )


# ============================================================
# MAIN PROCESSING
# ============================================================

if files:

    try:

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        raw_df, file_info = (
            load_multiple_files(
                files
            )
        )

        st.success(

            f"Loaded {len(files):,} file(s) — "
            f"{len(raw_df):,} rows × "
            f"{len(raw_df.columns):,} columns"

        )

        with st.expander(
            "📁 Uploaded Files"
        ):

            st.dataframe(

                pd.DataFrame(
                    file_info
                ),

                use_container_width=True

            )

        # ----------------------------------------------------
        # PREPARE
        # ----------------------------------------------------

        prepared_df, detected_columns = (
            prepare_hdfc_data(
                raw_df
            )
        )

        st.subheader(
            "🔎 Detected Dataset Columns"
        )

        st.json(
            detected_columns
        )

        # ----------------------------------------------------
        # RELEVANCE
        # ----------------------------------------------------

        hdfc_df = get_relevant_data(
            prepared_df
        )

        st.info(

            f"""
**HDFC-relevant records:** {len(hdfc_df):,}

**Original records:** {len(raw_df):,}

**Relevant share:** {
    (
        len(hdfc_df)
        / len(raw_df)
        * 100
    )
    if len(raw_df)
    else 0
    :.2f}%
"""

        )

        # ----------------------------------------------------
        # DATA PREVIEW
        # ----------------------------------------------------

        with st.expander(
            "👀 HDFC Relevant Data Preview"
        ):

            preview_cols = [

                "_date",

                "_outlet",

                "_author",

                "_sentiment",

                "_engagement",

                "_reach",

                "_text"

            ]

            preview_cols = [
                c
                for c in preview_cols
                if c in hdfc_df.columns
            ]

            st.dataframe(

                hdfc_df[
                    preview_cols
                ].head(30),

                use_container_width=True

            )

        # ====================================================
        # RUN
        # ====================================================

        if st.button(

            "🚀 Generate Actual Strategic Review",

            type="primary",

            use_container_width=True

        ):

            if client is None:

                st.error(

                    """
Azure OpenAI is not configured.

Please check:

AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_DEPLOYMENT
"""

                )

                st.stop()

            progress = st.progress(
                0
            )

            status = st.empty()

            # =================================================
            # STEP 1
            # =================================================

            status.info(
                "1/7 — Preparing actual HDFC data..."
            )

            hdfc_df = add_periods(
                hdfc_df
            )

            progress.progress(
                10
            )

            # =================================================
            # STEP 2 SCREENSHOT
            # =================================================

            screenshot_analysis = None

            if screenshot:

                status.info(
                    "2/7 — Analyzing reference screenshot with Azure GPT-4o..."
                )

                screenshot_analysis = (
                    analyze_reference_screenshot(

                        screenshot,

                        raw_df.columns

                    )
                )

                progress.progress(
                    20
                )

            else:

                status.info(
                    "2/7 — No screenshot uploaded. "
                    "Using default strategic structure."
                )

                progress.progress(
                    20
                )

            # =================================================
            # STEP 3 SENTIMENT
            # =================================================

            status.info(
                "3/7 — Calculating actual sentiment..."
            )

            sentiment = sentiment_summary(
                hdfc_df
            )

            progress.progress(
                30
            )

            # =================================================
            # STEP 4 AI NARRATIVE CLASSIFICATION
            # =================================================

            status.info(
                "4/7 — Classifying actual narratives with Azure GPT-4o..."
            )

            # Limit AI classification to avoid
            # excessive API usage.
            ai_limit = min(
                len(hdfc_df),
                1000
            )

            try:

                hdfc_df = (
                    apply_ai_classification(
                        hdfc_df,
                        max_records=ai_limit
                    )
                )

            except Exception as ai_error:

                st.warning(

                    "Azure semantic classification "
                    f"could not be completed: {ai_error}. "
                    "Falling back to deterministic "
                    "local narrative classification."

                )

                hdfc_df[
                    "_narrative"
                ] = (
                    hdfc_df[
                        "_text"
                    ].apply(
                        classify_narrative_local
                    )
                )

            narrative_df = (
                calculate_narratives_from_column(
                    hdfc_df
                )
            )

            win_loss = (
                narrative_win_loss(
                    narrative_df
                )
            )

            progress.progress(
                50
            )

            # =================================================
            # STEP 5 OTHER METRICS
            # =================================================

            status.info(
                "5/7 — Calculating outlets, risks and trends..."
            )

            outlet_df = (
                calculate_outlet_watchlist(
                    hdfc_df
                )
            )

            risks = (
                calculate_silence_risks(
                    hdfc_df
                )
            )

            quarterly = (
                calculate_quarterly_resilience(
                    hdfc_df
                )
            )

            pullthrough = (
                calculate_weekly_pullthrough_ai(
                    hdfc_df
                )
            )

            progress.progress(
                70
            )

            # =================================================
            # STEP 6 REPORT
            # =================================================

            status.info(
                "6/7 — Generating strategic review with Azure GPT-4o..."
            )

            report = generate_report(

                hdfc_df,

                narrative_df,

                win_loss,

                outlet_df,

                risks,

                quarterly,

                pullthrough,

                screenshot_analysis

            )

            progress.progress(
                90
            )

            # =================================================
            # STEP 7
            # =================================================

            status.info(
                "7/7 — Preparing dashboard..."
            )

            progress.progress(
                100
            )

            status.success(
                "✅ Actual-data strategic review generated."
            )

            # =================================================
            # KPI
            # =================================================

            st.subheader(
                "📌 Actual HDFC Metrics"
            )

            c1, c2, c3, c4, c5 = (
                st.columns(5)
            )

            c1.metric(

                "Relevant Mentions",

                f"{sentiment['total_mentions']:,}"

            )

            c2.metric(

                "Positive",

                f"{sentiment['positive_pct']:.1f}%"

            )

            c3.metric(

                "Negative",

                f"{sentiment['negative_pct']:.1f}%"

            )

            c4.metric(

                "Net Sentiment",

                f"{sentiment['net_sentiment']:+.1f}%"

            )

            c5.metric(

                "Outlets",

                f"{hdfc_df['_outlet'].nunique():,}"

            )

            # =================================================
            # SCREENSHOT STRUCTURE
            # =================================================

            if screenshot_analysis:

                st.subheader(
                    "🖼️ Reference Screenshot Analysis"
                )

                with st.expander(
                    "View extracted screenshot structure"
                ):

                    st.json(
                        screenshot_analysis
                    )

            # =================================================
            # QUARTERLY RESILIENCE
            # =================================================

            st.subheader(
                "1. Resilience Index — Actual Quarterly Data"
            )

            if quarterly:

                quarterly_display = (
                    pd.DataFrame(
                        quarterly
                    )
                )

                st.dataframe(

                    quarterly_display,

                    use_container_width=True

                )

                if len(
                    quarterly
                ) < 4:

                    st.warning(

                        "The uploaded dataset does not contain "
                        "four quarters. Missing quarters are "
                        "not fabricated."

                    )

            else:

                st.warning(
                    "No valid date data available."
                )

            # =================================================
            # NARRATIVE
            # =================================================

            st.subheader(
                "2. Narrative Win / Loss Ledger"
            )

            if win_loss:

                ledger_df = (
                    pd.DataFrame(
                        win_loss
                    )
                )

                st.dataframe(

                    ledger_df,

                    use_container_width=True

                )

            else:

                st.info(
                    "No narrative data available."
                )

            # =================================================
            # NARRATIVE PULL THROUGH
            # =================================================

            st.subheader(
                "3. Narrative Pull-Through"
            )

            if not pullthrough.empty:

                pullthrough_display = (
                    pullthrough.copy()
                )

                st.dataframe(

                    pullthrough_display,

                    use_container_width=True

                )

                # Create percentage data
                # for the chart

                chart_data = (
                    pullthrough
                    .set_index(
                        "_week"
                    )
                    .copy()
                )

                percentage_columns = []

                for col in chart_data.columns:

                    if col == "Total":

                        continue

                    percentage_columns.append(
                        col
                    )

                if percentage_columns:

                    percentage_df = (
                        chart_data[
                            percentage_columns
                        ]
                        .div(
                            chart_data[
                                "Total"
                            ],
                            axis=0
                        )
                        * 100
                    )

                    percentage_df = (
                        percentage_df
                        .fillna(0)
                        .round(2)
                    )

                    st.line_chart(
                        percentage_df
                    )

                    st.caption(
                        "Narrative share is calculated "
                        "from actual HDFC-relevant records."
                    )

            else:

                st.info(
                    "No valid dated records available for pull-through."
                )

            # =================================================
            # OUTLETS
            # =================================================

            st.subheader(
                "4. Journalist / Outlet Watchlist"
            )

            if not outlet_df.empty:

                display_outlets = (
                    outlet_df[
                        [
                            "outlet",
                            "trend",
                            "mentions",
                            "positive_pct",
                            "negative_pct",
                            "net_sentiment"
                        ]
                    ]
                    .head(20)
                )

                st.dataframe(

                    display_outlets,

                    use_container_width=True

                )

            else:

                st.info(
                    "No outlet information detected."
                )

            # =================================================
            # RISKS
            # =================================================

            st.subheader(
                "5. Silence-Risk Flags"
            )

            if risks:

                st.dataframe(

                    pd.DataFrame(
                        risks
                    ),

                    use_container_width=True

                )

            else:

                st.info(
                    "No rule-based risk flags detected."
                )

            # =================================================
            # REPORT
            # =================================================

            st.divider()

            st.header(
                "📋 Strategic Review"
            )

            st.markdown(
                report
            )

            st.download_button(

                "⬇️ Download Strategic Review",

                data=report,

                file_name=
                    "HDFC_Bank_Strategic_Review.md",

                mime=
                    "text/markdown",

                use_container_width=True

            )

            # =================================================
            # DOWNLOAD NARRATIVE DATA
            # =================================================

            narrative_csv = (
                narrative_df
                .to_csv(
                    index=False
                )
            )

            st.download_button(

                "⬇️ Download Narrative Metrics",

                data=narrative_csv,

                file_name=
                    "HDFC_Narrative_Metrics.csv",

                mime=
                    "text/csv",

                use_container_width=True

            )

            # =================================================
            # AUDIT
            # =================================================

            st.divider()

            st.header(
                "🔍 Calculation Audit"
            )

            with st.expander(
                "Actual Sentiment"
            ):

                st.json(
                    sentiment
                )

            with st.expander(
                "Narrative Data"
            ):

                st.dataframe(

                    narrative_df,

                    use_container_width=True

                )

            with st.expander(
                "Quarterly Data"
            ):

                st.json(
                    quarterly
                )

            with st.expander(
                "Risk Flags"
            ):

                st.json(
                    risks
                )

            with st.expander(
                "Detected Columns"
            ):

                st.json(
                    detected_columns
                )

            with st.expander(
                "Azure Configuration"
            ):

                st.write(
                    "Endpoint:",
                    AZURE_OPENAI_ENDPOINT
                )

                st.write(
                    "API Version:",
                    AZURE_OPENAI_API_VERSION
                )

                st.write(
                    "Deployment:",
                    AZURE_OPENAI_DEPLOYMENT
                )

                st.write(
                    "API Key:",
                    "Configured"
                    if AZURE_OPENAI_API_KEY
                    else "Missing"
                )

    except Exception as e:

        st.error(
            f"❌ Processing failed: {e}"
        )

        with st.expander(
            "Technical error details"
        ):

            st.exception(
                e
            )

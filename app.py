import streamlit as st
import pandas as pd
import numpy as np
import secrets
import hashlib
import json
from datetime import datetime, timezone

# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Genomic Data MPC",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genomic Data Sharing & MPC")
st.subheader("Advanced Multi-Hospital Privacy-Aware Research Prototype")

st.info(
    "Educational prototype only. "
    "Not a production-grade MPC, blockchain, "
    "federated learning or differential privacy system. "
    "Use synthetic data only."
)

PRIME = 2**61 - 1
SCALE = 1000

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def read_hospital_file(uploaded_file):
    df = pd.read_csv(uploaded_file)

    if df.empty:
        raise ValueError("CSV file is empty.")

    df.columns = df.columns.astype(str).str.strip()

    if not df.columns.is_unique:
        raise ValueError("Duplicate column names detected.")

    return df


def prepare_numeric_data(df):
    numeric = df.select_dtypes(include="number").copy()

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return numeric


def create_secret_shares(data):
    rows, cols = data.shape

    share_a = np.empty((rows, cols), dtype=np.int64)
    share_b = np.empty((rows, cols), dtype=np.int64)
    share_c = np.empty((rows, cols), dtype=np.int64)

    for i in range(rows):
        for j in range(cols):

            value = int(data[i, j])

            a = secrets.randbelow(PRIME)
            b = secrets.randbelow(PRIME)
            c = (value - a - b) % PRIME

            share_a[i, j] = a
            share_b[i, j] = b
            share_c[i, j] = c

    return share_a, share_b, share_c


def decode_signed(values):
    return np.array([
        int(x) if int(x) <= PRIME // 2
        else int(x) - PRIME
        for x in values
    ], dtype=object)


def make_audit_log(events):
    previous_hash = "GENESIS"
    records = []

    for event in events:

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        record = {
            "timestamp": timestamp,
            "event": event,
            "previous_hash": previous_hash
        }

        content = json.dumps(
            record,
            sort_keys=True
        )

        current_hash = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        record["hash"] = current_hash

        records.append(record)
        previous_hash = current_hash

    return pd.DataFrame(records)


def verify_audit_log(log_df):
    previous_hash = "GENESIS"

    for _, row in log_df.iterrows():

        record = {
            "timestamp": row["timestamp"],
            "event": row["event"],
            "previous_hash": previous_hash
        }

        content = json.dumps(
            record,
            sort_keys=True
        )

        expected_hash = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        if expected_hash != row["hash"]:
            return False

        previous_hash = row["hash"]

    return True


def train_local_model(df, features, target, epochs=300, learning_rate=0.03):
    """
    Educational

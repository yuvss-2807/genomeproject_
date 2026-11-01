import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Genomic Data MPC", page_icon="🧬")

st.title("🧬 Genomic Data Sharing & Multi-Party Computation")
st.write("Privacy-aware genomic data analysis demonstration.")

st.info("Demo only: This project demonstrates additive secret sharing. It is not production-grade secure MPC.")

uploaded_file = st.file_uploader(
    "Upload genomic data CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.success("CSV uploaded successfully!")

        st.subheader("1. Uploaded Data")
        st.write("Rows:", len(df))
        st.write("Columns:", len(df.columns))
        st.dataframe(df.head())

        st.subheader("2. Privacy Screening")

        sensitive_words = [
            "name", "email", "phone", "address",
            "patient_id", "dob", "birth", "identifier"
        ]

        flagged = [
            col for col in df.columns
            if any(word in col.lower() for word in sensitive_words)
        ]

        if flagged:
            st.warning("Potentially sensitive columns detected:")
            st.write(flagged)
        else:
            st.success("No obvious personal identifier column names detected.")

        st.subheader("3. Multi-Party Computation Demo")

        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            st.warning("No numeric columns found for secret sharing.")
        else:
            data = numeric_df.to_numpy(dtype=float)

            share1 = np.random.random(data.shape) * 1000
            share2 = np.random.random(data.shape) * 1000
            share3 = data - share1 - share2

            st.success("Data divided into three demonstration shares.")

            st.write("Participant 1 Share")
            st.dataframe(pd.DataFrame(share1, columns=numeric_df.columns).head())

            st.write("Participant 2 Share")
            st.dataframe(pd.DataFrame(share2, columns=numeric_df.columns).head())

            st.write("Participant 3 Share")
            st.dataframe(pd.DataFrame(share3, columns=numeric_df.columns).head())

            reconstructed = share1 + share2 + share3

            st.subheader("4. Verification")

            if np.allclose(data, reconstructed):
                st.success("Verification successful: shares reconstruct the original numeric data.")
            else:
                st.error("Verification failed.")

            st.download_button(
                "Download Participant 1 Share",
                pd.DataFrame(share1, columns=numeric_df.columns).to_csv(index=False),
                "participant1_share.csv",
                "text/csv"
            )

            st.download_button(
                "Download Participant 2 Share",
                pd.DataFrame(share2, columns=numeric_df.columns).to_csv(index=False),
                "participant2_share.csv",
                "text/csv"
            )

            st.download_button(
                "Download Participant 3 Share",
                pd.DataFrame(share3, columns=numeric_df.columns).to_csv(index=False),
                "participant3_share.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")

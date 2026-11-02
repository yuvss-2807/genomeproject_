import streamlit as st
import pandas as pd
import numpy as np
import secrets

# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Genomic Data MPC",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genomic Data Sharing & MPC")
st.write(
    "Privacy-aware genomic data analysis "
    "using Three-Party Computation."
)

st.info(
    "Educational demonstration only. "
    "This application is not production-grade secure MPC."
)

PRIME = 2**61 - 1
SCALE = 1000

# ==========================================
# 2. CSV DATA UPLOAD
# ==========================================

st.header("1. Upload Genomic Data")

uploaded_file = st.file_uploader(
    "Upload genomic CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("Uploaded CSV is empty.")
            st.stop()

        st.success("CSV uploaded successfully!")

        # ==================================
        # 3. DATA VALIDATION
        # ==================================

        st.header("2. Data Validation")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Rows", len(df))
        col2.metric("Total Columns", len(df.columns))
        col3.metric("Missing Values", int(df.isna().sum().sum()))

        st.dataframe(df.head(10), use_container_width=True)

        # ==================================
        # 4. PRIVACY SCREENING
        # ==================================

        st.header("3. Privacy Screening")

        sensitive_words = [
            "name",
            "email",
            "phone",
            "address",
            "patient_id",
            "dob",
            "birth",
            "identifier",
            "ssn"
        ]

        flagged = [
            col for col in df.columns
            if any(word in col.lower()
                   for word in sensitive_words)
        ]

        if flagged:
            st.warning("Potentially sensitive columns detected.")
            st.write(flagged)
        else:
            st.success(
                "No obvious personal identifier column names detected."
            )

        st.warning(
            "Column-name screening cannot guarantee that "
            "the dataset is anonymous."
        )

        # ==================================
        # 5. NUMERIC DATA PREPARATION
        # ==================================

        st.header("4. Prepare Data for MPC")

        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            st.error("No numeric columns found.")
            st.stop()

        numeric_df = numeric_df.replace(
            [np.inf, -np.inf], np.nan
        )

        valid_counts = numeric_df.count()

        usable_columns = valid_counts[
            valid_counts > 0
        ].index.tolist()

        numeric_df = numeric_df[usable_columns]

        if numeric_df.empty:
            st.error("No usable numeric data found.")
            st.stop()

        st.write("Numeric columns:")
        st.write(usable_columns)

        st.dataframe(
            numeric_df.head(),
            use_container_width=True
        )

        # ==================================
        # 6. THREE PARTY SERVER SIMULATION
        # ==================================

        st.header("5. Three-Party MPC Simulation")

        st.write(
            "The numeric data is converted into "
            "fixed-point integers and divided into "
            "three additive shares."
        )

        st.caption(
            "Missing values are represented as zero in shares "
            "and excluded from the corresponding column mean."
        )

        data = np.rint(
            numeric_df.fillna(0).to_numpy(dtype=float) * SCALE
        )

        if not np.isfinite(data).all():
            st.error("Invalid numeric values detected.")
            st.stop()

        if np.max(np.abs(data), initial=0) >= PRIME // (
            4 * max(len(data), 1)
        ):
            st.error(
                "Values are too large for this demo's "
                "safe modular range."
            )
            st.stop()

        data = data.astype(np.int64)

        rows, cols = data.shape

        # ==================================
        # 7. ADDITIVE SECRET SHARING
        # ==================================

        st.header("6. Additive Secret Sharing")

        share1 = np.empty((rows, cols), dtype=np.int64)
        share2 = np.empty((rows, cols), dtype=np.int64)
        share3 = np.empty((rows, cols), dtype=np.int64)

        for i in range(rows):

            for j in range(cols):

                value = int(data[i, j])

                s1 = secrets.randbelow(PRIME)
                s2 = secrets.randbelow(PRIME)

                s3 = (value - s1 - s2) % PRIME

                share1[i, j] = s1
                share2[i, j] = s2
                share3[i, j] = s3

        st.success(
            "Data successfully divided into three shares."
        )

        # ==================================
        # 8. THREE SERVER SIMULATION
        # ==================================

        st.header("7. Three MPC Servers")

        server1, server2, server3 = st.columns(3)

        with server1:
            st.subheader("🖥️ Server A")
            st.success("Share 1 Created")
            st.metric("Rows", rows)

        with server2:
            st.subheader("🖥️ Server B")
            st.success("Share 2 Created")
            st.metric("Rows", rows)

        with server3:
            st.subheader("🖥️ Server C")
            st.success("Share 3 Created")
            st.metric("Rows", rows)

        # ==================================
        # 9. SECURE AGGREGATION
        # ==================================

        st.header("8. Secure Aggregation")

        if st.button("Run MPC Computation"):

            # Each server computes its local column sums.
            local_sum1 = np.sum(
                share1.astype(object), axis=0
            ) % PRIME

            local_sum2 = np.sum(
                share2.astype(object), axis=0
            ) % PRIME

            local_sum3 = np.sum(
                share3.astype(object), axis=0
            ) % PRIME

            # Combine only aggregate shares.
            aggregate = (
                local_sum1 + local_sum2 + local_sum3
            ) % PRIME

            # Decode signed aggregate.
            signed_sum = np.array([
                int(x) if int(x) <= PRIME // 2
                else int(x) - PRIME
                for x in aggregate
            ], dtype=object)

            total_sum = np.array(
                signed_sum, dtype=float
            ) / SCALE

            counts = valid_counts.to_numpy()

            means = np.divide(
                total_sum,
                counts,
                out=np.zeros(len(counts), dtype=float),
                where=counts > 0
            )

            results = pd.DataFrame({
                "Feature": usable_columns,
                "Aggregate Sum": total_sum,
                "Aggregate Mean": means,
                "Valid Records": counts
            })

            st.session_state["mpc_results"] = results
            st.session_state["aggregate"] = aggregate

            st.success(
                "Three-party aggregate computation completed!"
            )

        # ==================================
        # 10. MPC RESULTS DASHBOARD
        # ==================================

        if "mpc_results" in st.session_state:

            st.header("9. MPC Results Dashboard")

            results = st.session_state["mpc_results"]

            st.dataframe(
                results,
                use_container_width=True
            )

            a, b = st.columns(2)

            with a:
                st.subheader("Aggregate Sum")

                st.bar_chart(
                    results.set_index("Feature")[
                        "Aggregate Sum"
                    ]
                )

            with b:
                st.subheader("Aggregate Mean")

                st.bar_chart(
                    results.set_index("Feature")[
                        "Aggregate Mean"
                    ]
                )

            # ==================================
            # 11. RECONSTRUCTION VERIFICATION
            # ==================================

            st.header("10. Share Reconstruction Verification")

            st.warning(
                "Reconstruction reveals the original numeric data. "
                "Use this only with synthetic demonstration data."
            )

            if st.button("Verify All Three Shares"):

                reconstructed = (
                    share1.astype(object)
                    + share2.astype(object)
                    + share3.astype(object)
                ) % PRIME

                original = data.astype(object) % PRIME

                verified = np.array_equal(
                    reconstructed,
                    original
                )

                if verified:
                    st.success(
                        "Verification successful! "
                        "All three shares reconstruct the encoded data."
                    )
                else:
                    st.error("Verification failed!")

            # ==================================
            # 12. DOWNLOAD SERVER SHARES
            # ==================================

            st.header("11. Download Server Shares")

            def share_csv(share):

                share_df = pd.DataFrame(
                    share,
                    columns=usable_columns
                )

                return share_df.to_csv(
                    index=False
                ).encode("utf-8")

            d1, d2, d3 = st.columns(3)

            with d1:
                st.download_button(
                    "Download Server A Share",
                    share_csv(share1),
                    "server_A_share.csv",
                    "text/csv"
                )

            with d2:
                st.download_button(
                    "Download Server B Share",
                    share_csv(share2),
                    "server_B_share.csv",
                    "text/csv"
                )

            with d3:
                st.download_button(
                    "Download Server C Share",
                    share_csv(share3),
                    "server_C_share.csv",
                    "text/csv"
                )

            st.download_button(
                "Download MPC Results",
                results.to_csv(index=False).encode("utf-8"),
                "mpc_results.csv",
                "text/csv"
            )

            # ==================================
            # 13. FINAL STATUS
            # ==================================

            st.header("12. Project Status")

            st.success(
                "Genome Data MPC Demonstration Completed!"
            )

            st.write("✔ CSV Upload")
            st.write("✔ Privacy Screening")
            st.write("✔ Three Server Simulation")
            st.write("✔ Additive Secret Sharing")
            st.write("✔ Secure Aggregate Sum")
            st.write("✔ Secure Aggregate Mean")
            st.write("✔ Reconstruction Verification")
            st.write("✔ MPC Dashboard")

    except Exception as e:

        st.error(f"Error processing file: {e}")

else:

    st.info(
        "Please upload a CSV file to start genomic data analysis."
    )

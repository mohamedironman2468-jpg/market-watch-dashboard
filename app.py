
import os

import streamlit as st
import pandas as pd
from groq import Groq

st.set_page_config(
    page_title="Market Watch",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Market Watch")
st.caption(
    "Read-only market monitor | Sample data for testing"
)

st.info(
    "DEMO MODE: These are fictional prices, not live market data. "
    "This app cannot place trades."
)

# Sample data to test the dashboard.
# We will replace this with live public prices later.
data = pd.DataFrame({
    "Market": ["Market A", "Market B", "Market C"],
    "Price": [100.00, 100.40, 99.85],
})

st.subheader("Market overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Markets monitored", len(data))

with col2:
    st.metric("Highest sample price", f"${data['Price'].max():.2f}")

with col3:
    st.metric("Lowest sample price", f"${data['Price'].min():.2f}")

st.subheader("Prices")

st.bar_chart(
    data,
    x="Market",
    y="Price",
)

st.dataframe(data, hide_index=True, use_container_width=True)

# Compare the highest and lowest sample prices.
highest = data.loc[data["Price"].idxmax()]
lowest = data.loc[data["Price"].idxmin()]

difference = highest["Price"] - lowest["Price"]
percentage = difference / lowest["Price"] * 100

st.subheader("Price difference scanner")

threshold = st.slider(
    "Alert threshold (%)",
    min_value=0.05,
    max_value=2.0,
    value=0.30,
    step=0.05,
)

st.metric(
    "Difference between highest and lowest",
    f"{percentage:.3f}%",
    help="A raw price difference, before costs. Not proof of a profitable trade.",
)

if percentage >= threshold:
    st.warning(
        f"Difference exceeds your {threshold:.2f}% threshold. "
        "This is a signal to investigate, not a trading recommendation."
    )
else:
    st.success("The sample difference is below your alert threshold.")

st.subheader("AI explanation")

st.write(
    "Ask Groq to explain the sample results in plain English. "
    "AI explanations are optional."
)

if st.button("Explain these results with AI"):
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        st.warning(
            "Groq is not configured yet. Follow Step 5 to add your "
            "API key securely in Streamlit Secrets."
        )
    else:
        try:
            client = Groq(api_key=api_key)

            with st.spinner("Preparing explanation..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You explain market data for beginners. "
                                "Use only the supplied data. Explain "
                                "uncertainty and costs. Never claim a "
                                "profit is guaranteed or recommend "
                                "placing a trade. The data may be fictional."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Sample market prices: "
                                f"{data.to_dict(orient='records')}. "
                                f"Highest-lowest difference: "
                                f"{percentage:.3f}%. "
                                f"Alert threshold: {threshold:.2f}%. "
                                "Explain what these figures mean."
                            ),
                        },
                    ],
                    temperature=0.2,
                    max_tokens=300,
                )

            st.markdown(
                response.choices[0].message.content
            )

        except Exception as error:
            st.error(
                "The AI request failed. Check your Groq configuration "
                "and try again. Technical details: "
                + str(error)
            )

st.divider()
st.caption(
    "Version 1 | Sample data only | No trading or order execution"
)


import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="Market Watch",
    page_icon="📈",
    layout="wide",
)

API_URL = "https://data-api.binance.vision/api/v3/ticker/price"

st.title("📈 Market Watch")
st.caption("Live public market data · Read-only · No trading")

st.info(
    "This dashboard displays exchange price quotes for monitoring "
    "and learning. It does not execute trades or guarantee that "
    "price differences can be exploited."
)

# Refresh the page periodically.
st.markdown(
    """
    <meta http-equiv="refresh" content="15">
    """,
    unsafe_allow_html=True,
)

symbols = ["BTCUSDT", "BTCUSDC", "ETHUSDT"]

@st.cache_data(ttl=10, show_spinner=False)
def fetch_prices():
    results = []
    errors = []

    for symbol in symbols:
        try:
            response = requests.get(
                API_URL,
                params={"symbol": symbol},
                timeout=8,
            )
            response.raise_for_status()
            item = response.json()

            results.append({
                "Symbol": item["symbol"],
                "Price": float(item["price"]),
            })

        except (requests.RequestException, ValueError, KeyError) as exc:
            errors.append(f"{symbol}: {exc}")

    return results, errors


with st.spinner("Fetching public market prices..."):
    results, errors = fetch_prices()

if errors:
    st.warning(
        "Some prices could not be retrieved. "
        "Check your connection or try again later.\n\n"
        + "\n".join(errors)
    )

if not results:
    st.error(
        "No market prices were retrieved. "
        "The dashboard cannot show a live comparison right now."
    )
    st.stop()

data = pd.DataFrame(results)

updated_at = datetime.now(
    ZoneInfo("UTC")
).strftime("%Y-%m-%d %H:%M:%S UTC")

st.caption(f"Last successful page fetch: {updated_at}")

st.subheader("Current prices")

cols = st.columns(len(data))

for col, (_, row) in zip(cols, data.iterrows()):
    with col:
        st.metric(
            label=row["Symbol"],
            value=f"${row['Price']:,.2f}",
        )

st.dataframe(
    data,
    hide_index=True,
    use_container_width=True,
)

st.subheader("Price comparison")

st.bar_chart(
    data,
    x="Symbol",
    y="Price",
)

st.caption(
    "Quotes are from different trading pairs. USDT and USDC "
    "are separate assets, and ETH is not the same asset as BTC. "
    "Do not interpret these prices as a direct arbitrage opportunity."
)

st.subheader("Compare the same asset")

btc_quotes = data[
    data["Symbol"].isin(["BTCUSDT", "BTCUSDC"])
]

if len(btc_quotes) == 2:
    prices = dict(
        zip(btc_quotes["Symbol"], btc_quotes["Price"])
    )

    usdt = prices["BTCUSDT"]
    usdc = prices["BTCUSDC"]

    difference = usdc - usdt
    pct = difference / usdt * 100 if usdt else 0

    c1, c2, c3 = st.columns(3)

    c1.metric("BTC/USDT", f"${usdt:,.2f}")
    c2.metric("BTC/USDC", f"${usdc:,.2f}")
    c3.metric("Quote difference", f"{pct:+.4f}%")

    st.write(
        "This is a raw quote comparison, not a confirmed profit. "
        "Stablecoin prices can differ, and the quote alone does "
        "not account for liquidity, fees or execution."
    )
else:
    st.warning(
        "Both BTC trading pairs must load before comparing them."
    )

st.subheader("AI explanation")

if st.button("Explain the current readings"):
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        st.warning(
            "Add GROQ_API_KEY under your Streamlit app's Secrets "
            "settings to enable AI explanations."
        )
    else:
        try:
            client = Groq(api_key=api_key)

            with st.spinner("Explaining the readings..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Explain supplied market data for a "
                                "beginner. Do not invent facts, prices "
                                "or timestamps. Distinguish raw price "
                                "differences from actual profit. "
                                "Mention uncertainty and relevant "
                                "costs. Do not recommend trades."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Explain these public market quotes "
                                "in plain English:\n"
                                + data.to_json(orient="records")
                                + "\nExplain what can and cannot be "
                                "concluded from this snapshot."
                            ),
                        },
                    ],
                    temperature=0.2,
                    max_tokens=300,
                )

            st.write(response.choices[0].message.content)

        except Exception as exc:
            st.error(f"AI explanation failed: {exc}")

st.divider()
st.caption(
    "Read-only educational monitor. No account access, "
    "private keys, or order execution."
)

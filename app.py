
import os
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st
from groq import Groq

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Market Watch",
    page_icon="📈",
    layout="wide",
)

API_URL = "https://data-api.binance.vision/api/v3/depth"

st.title("📈 Market Watch")
st.caption(
    "Live public market data · Simulation only · No trading"
)

st.info(
    "This dashboard uses public order-book quotes. "
    "All cycle calculations are hypothetical estimates, "
    "not proof of a profitable trade. No wallet is connected."
)

# --------------------------------------------------
# USER SETTINGS
# --------------------------------------------------

st.sidebar.header("Scanner settings")

refresh_seconds = st.sidebar.selectbox(
    "Refresh interval",
    options=[15, 30, 60],
    index=0,
    format_func=lambda x: f"Every {x} seconds",
)

fee_percent = st.sidebar.number_input(
    "Estimated fee per market leg (%)",
    min_value=0.0,
    max_value=2.0,
    value=0.10,
    step=0.01,
    help="Illustrative fee assumption. This is not a verified account fee.",
)

simulated_amount = st.sidebar.number_input(
    "Hypothetical starting amount (USDT)",
    min_value=10.0,
    max_value=10000.0,
    value=100.0,
    step=10.0,
)

alert_threshold = st.sidebar.number_input(
    "Alert when estimated net return exceeds (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.10,
    step=0.05,
)

st.sidebar.caption(
    "The starting amount is simulated. No money is moved."
)

fee_rate = fee_percent / 100.0

# --------------------------------------------------
# PUBLIC ORDER BOOK
# --------------------------------------------------

@st.cache_data(ttl=8, show_spinner=False)
def fetch_order_book(symbol):
    """Fetch public bid and ask levels. No API key is used."""
    response = requests.get(
        API_URL,
        params={"symbol": symbol, "limit": 20},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()

    bids = [
        (float(price), float(quantity))
        for price, quantity in payload["bids"]
    ]
    asks = [
        (float(price), float(quantity))
        for price, quantity in payload["asks"]
    ]

    if not bids or not asks:
        raise ValueError(f"No usable order-book levels for {symbol}")

    return {
        "bids": bids,
        "asks": asks,
        "fetched_at": datetime.now(timezone.utc).strftime(
            "%H:%M:%S UTC"
        ),
    }


# --------------------------------------------------
# SIMULATION HELPERS
# --------------------------------------------------

def buy_base_with_quote(asks, quote_budget):
    """
    Simulate buying an asset by walking through ask levels.
    Returns: base quantity, quote spent, quote left unfilled.
    """
    remaining = quote_budget
    base_bought = 0.0
    quote_spent = 0.0

    for price, quantity in asks:
        level_cost = price * quantity
        spend = min(remaining, level_cost)

        if spend <= 0:
            break

        base_bought += spend / price
        quote_spent += spend
        remaining -= spend

        if remaining <= max(1e-8, quote_budget * 1e-9):
            remaining = 0.0
            break

    return base_bought, quote_spent, max(0.0, remaining)


def sell_base_for_quote(bids, base_quantity):
    """
    Simulate selling an asset by walking through bid levels.
    Returns: quote received, base sold, base left unsold.
    """
    remaining = base_quantity
    quote_received = 0.0
    base_sold = 0.0

    for price, quantity in bids:
        amount = min(remaining, quantity)

        if amount <= 0:
            break

        quote_received += amount * price
        base_sold += amount
        remaining -= amount

        if remaining <= max(1e-12, base_quantity * 1e-9):
            remaining = 0.0
            break

    return quote_received, base_sold, max(0.0, remaining)


def quote_is_unfilled(remaining, original_amount):
    tolerance = max(1e-8, abs(original_amount) * 1e-6)
    return remaining > tolerance


def base_is_unsold(remaining, original_amount):
    tolerance = max(1e-12, abs(original_amount) * 1e-6)
    return remaining > tolerance


def finish_route(route_name, start_amount, end_amount):
    """Package an estimate; this does not place an order."""
    net = end_amount - start_amount

    return {
        "route": route_name,
        "start": start_amount,
        "end": end_amount,
        "net": net,
        "net_pct": (net / start_amount) * 100.0,
    }


def simulate_route_a(books, start_usdt, fee):
    """
    Hypothetical route:
    USDT -> BTC -> USDC -> USDT

    Fee is approximated as a haircut to each leg's output.
    """
    btc, _, leftover = buy_base_with_quote(
        books["BTCUSDT"]["asks"], start_usdt
    )
    if quote_is_unfilled(leftover, start_usdt):
        return None, "Insufficient BTC/USDT ask depth."

    btc *= 1.0 - fee

    usdc, _, leftover_btc = sell_base_for_quote(
        books["BTCUSDC"]["bids"], btc
    )
    if base_is_unsold(leftover_btc, btc):
        return None, "Insufficient BTC/USDC bid depth."

    usdc *= 1.0 - fee

    usdt, _, leftover_usdc = sell_base_for_quote(
        books["USDCUSDT"]["bids"], usdc
    )
    if base_is_unsold(leftover_usdc, usdc):
        return None, "Insufficient USDC/USDT bid depth."

    usdt *= 1.0 - fee

    return finish_route(
        "USDT → BTC → USDC → USDT",
        start_usdt,
        usdt,
    ), None


def simulate_route_b(books, start_usdt, fee):
    """
    Hypothetical route:
    USDT -> USDC -> BTC -> USDT

    Fee is approximated as a haircut to each leg's output.
    """
    usdc, _, leftover = buy_base_with_quote(
        books["USDCUSDT"]["asks"], start_usdt
    )
    if quote_is_unfilled(leftover, start_usdt):
        return None, "Insufficient USDC/USDT ask depth."

    usdc *= 1.0 - fee

    btc, _, leftover_usdc = buy_base_with_quote(
        books["BTCUSDC"]["asks"], usdc
    )
    if quote_is_unfilled(leftover_usdc, usdc):
        return None, "Insufficient BTC/USDC ask depth."

    btc *= 1.0 - fee

    usdt, _, leftover_btc = sell_base_for_quote(
        books["BTCUSDT"]["bids"], btc
    )
    if base_is_unsold(leftover_btc, btc):
        return None, "Insufficient BTC/USDT bid depth."

    usdt *= 1.0 - fee

    return finish_route(
        "USDT → USDC → BTC → USDT",
        start_usdt,
        usdt,
    ), None


# --------------------------------------------------
# HISTORY FOR THIS DASHBOARD SESSION
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_books" not in st.session_state:
    st.session_state.latest_books = {}

if "latest_routes" not in st.session_state:
    st.session_state.latest_routes = {}

if "latest_errors" not in st.session_state:
    st.session_state.latest_errors = []


# --------------------------------------------------
# AUTO-UPDATING MONITOR
# --------------------------------------------------

@st.fragment(run_every=f"{refresh_seconds}s")
def live_monitor():
    books = {}
    errors = []

    for symbol in ["BTCUSDT", "BTCUSDC", "USDCUSDT"]:
        try:
            books[symbol] = fetch_order_book(symbol)
        except (requests.RequestException, ValueError, KeyError) as exc:
            errors.append(f"{symbol}: {exc}")

    st.session_state.latest_books = books
    st.session_state.latest_errors = errors

    if errors:
        st.warning(
            "Some market data could not be fetched. "
            "Unavailable markets will not be treated as zero prices."
        )
        for message in errors:
            st.caption(message)

    if len(books) != 3:
        st.error(
            "Waiting for all three order books. "
            "Check again after the next refresh."
        )
        st.session_state.latest_routes = {}
        return

    rows = []

    for symbol, book in books.items():
        bid_price, bid_qty = book["bids"][0]
        ask_price, ask_qty = book["asks"][0]
        midpoint = (bid_price + ask_price) / 2.0
        spread = ask_price - bid_price
        spread_pct = spread / midpoint * 100.0 if midpoint else 0.0

        rows.append({
            "Pair": symbol,
            "Best bid": bid_price,
            "Bid quantity": bid_qty,
            "Best ask": ask_price,
            "Ask quantity": ask_qty,
            "Bid-ask spread": spread,
            "Spread (%)": spread_pct,
            "Fetched (UTC)": book["fetched_at"],
        })

    st.subheader("1. Live order books")

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Best bid": st.column_config.NumberColumn(format="%.8f"),
            "Best ask": st.column_config.NumberColumn(format="%.8f"),
            "Bid-ask spread": st.column_config.NumberColumn(format="%.8f"),
            "Spread (%)": st.column_config.NumberColumn(format="%.4f%%"),
        },
    )

    st.caption(
        "A bid is the highest displayed buying price; an ask is the "
        "lowest displayed selling price. Quantities refer to the base "
        "asset at that price level."
    )

    # Midpoint prices for monitoring, not executable prices.
    btcusdt_mid = (
        books["BTCUSDT"]["bids"][0][0]
        + books["BTCUSDT"]["asks"][0][0]
    ) / 2.0

    btcusdc_mid = (
        books["BTCUSDC"]["bids"][0][0]
        + books["BTCUSDC"]["asks"][0][0]
    ) / 2.0

    usdcusdt_mid = (
        books["USDCUSDT"]["bids"][0][0]
        + books["USDCUSDT"]["asks"][0][0]
    ) / 2.0

    route_a, error_a = simulate_route_a(
        books, simulated_amount, fee_rate
    )
    route_b, error_b = simulate_route_b(
        books, simulated_amount, fee_rate
    )

    routes = {
        "Route A": route_a,
        "Route B": route_b,
    }
    st.session_state.latest_routes = routes

    # Store only in this Streamlit session.
    now = datetime.now(timezone.utc)
    hist_row = {
        "time": now.strftime("%H:%M:%S"),
        "BTC/USDT midpoint": btcusdt_mid,
        "BTC/USDC midpoint": btcusdc_mid,
        "USDC/USDT midpoint": usdcusdt_mid,
        "Route A net (%)": (
            route_a["net_pct"] if route_a else None
        ),
        "Route B net (%)": (
            route_b["net_pct"] if route_b else None
        ),
    }

    # Avoid duplicate history records within the same second.
    history = st.session_state.history
    if not history or history[-1]["time"] != hist_row["time"]:
        history.append(hist_row)

    # Keep the latest 240 points in memory.
    st.session_state.history = history[-240:]

    st.subheader("2. Live price history")

    history_df = pd.DataFrame(st.session_state.history)

    if len(history_df) >= 2:
        st.caption(
            "Prices collected during this browser session. "
            "History resets when the session is lost or the app restarts."
        )

        st.line_chart(
            history_df.set_index("time")[
                [
                    "BTC/USDT midpoint",
                    "BTC/USDC midpoint",
                ]
            ]
        )

        st.caption("Simulated route return over time (after estimated fees)")

        st.line_chart(
            history_df.set_index("time")[
                ["Route A net (%)", "Route B net (%)"]
            ]
        )
    else:
        st.info(
            "Collecting the first few data points. "
            "The charts will appear as the monitor refreshes."
        )

    st.subheader("3. Simulated cycle calculator")

    st.caption(
        f"Starting amount: {simulated_amount:,.2f} USDT · "
        f"Estimated fee per leg: {fee_percent:.3f}% · "
        "No orders are sent."
    )

    route_cols = st.columns(2)

    for col, label, route, error in [
        (route_cols[0], "Route A", route_a, error_a),
        (route_cols[1], "Route B", route_b, error_b),
    ]:
        with col:
            st.markdown(
                "**" + (
                    route["route"] if route else label
                ) + "**"
            )

            if route:
                st.metric(
                    "Hypothetical ending amount",
                    f"{route['end']:,.4f} USDT",
                    f"{route['net']:+,.4f} USDT",
                )
                st.write(
                    f"Estimated net return: **{route['net_pct']:+.4f}%**"
                )
            else:
                st.warning(error or "Route estimate is unavailable.")

    st.caption(
        "These calculations walk through up to 20 displayed levels per "
        "market. Fees are approximated as a percentage deducted from each "
        "leg's output. The estimate excludes latency, changing quotes, "
        "other costs, account-specific fees and any depth beyond the data "
        "returned. A simulation is not a guaranteed executable result."
    )

    st.subheader("4. Threshold alerts")

    qualifying = [
        route
        for route in routes.values()
        if route is not None
        and route["net_pct"] >= alert_threshold
    ]

    if qualifying:
        for route in qualifying:
            st.warning(
                f"Simulated alert: {route['route']} has an estimated "
                f"net return of {route['net_pct']:+.4f}%, above your "
                f"{alert_threshold:.3f}% threshold. This is a signal "
                "to inspect the data, not a recommendation to trade."
            )
    else:
        st.success(
            "No available route currently exceeds your selected "
            "estimated-return threshold."
        )


live_monitor()


# --------------------------------------------------
# GROQ EXPLANATION
# --------------------------------------------------

st.divider()
st.subheader("5. AI explanation")

st.write(
    "Ask Groq to explain the latest market readings and simulation. "
    "AI output is explanatory only and may be mistaken."
)

if st.button("Explain the current readings"):
    books = st.session_state.latest_books
    routes = st.session_state.latest_routes

    try:
        api_key = st.secrets.get(
            "GROQ_API_KEY",
            os.environ.get("GROQ_API_KEY", ""),
        )
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        st.warning(
            "Groq is not configured. Add GROQ_API_KEY in your "
            "Streamlit app's Secrets settings."
        )
    elif len(books) != 3:
        st.warning(
            "Wait for all three order books to load, then try again."
        )
    else:
        snapshot = {}

        for symbol, book in books.items():
            snapshot[symbol] = {
                "best_bid": book["bids"][0][0],
                "best_ask": book["asks"][0][0],
                "fetched_at_utc": book["fetched_at"],
            }

        route_summary = {}
        for label, route in routes.items():
            route_summary[label] = (
                {
                    "route": route["route"],
                    "ending_amount_usdt": round(route["end"], 6),
                    "net_change_usdt": round(route["net"], 6),
                    "net_return_percent": round(route["net_pct"], 6),
                }
                if route
                else "Unavailable due to insufficient order-book depth"
            )

        try:
            client = Groq(api_key=api_key)

            with st.spinner("Preparing explanation..."):
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You explain public market data to a beginner. "
                                "Use only the supplied values. Do not invent "
                                "prices or claim any result is guaranteed. "
                                "Distinguish a hypothetical estimate from an "
                                "executable trade. Mention that fees, latency, "
                                "depth and changing quotes can invalidate a "
                                "snapshot. Never instruct the user to place "
                                "a trade or connect a wallet."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Explain this read-only market snapshot and "
                                "simulated cycle estimate in plain English. "
                                f"Order books: {snapshot}. "
                                f"Simulation: {route_summary}. "
                                f"Starting amount: {simulated_amount} USDT. "
                                f"Estimated fee per leg: {fee_percent}%. "
                                "What can and cannot be concluded?"
                            ),
                        },
                    ],
                    temperature=0.2,
                    max_tokens=450,
                )

            st.markdown(response.choices[0].message.content)

        except Exception as exc:
            st.error(f"AI explanation failed: {exc}")


st.divider()
st.caption(
    "Read-only educational monitor. No wallet connection, private keys, "
    "real balances, or order execution. Prices and estimates can be stale "
    "or incomplete."
)

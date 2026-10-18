
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st
from groq import Groq

# ==================================================
# MARKET WATCH — PAPER TRADING ONLY
# ==================================================

st.set_page_config(
    page_title="Market Watch Paper Bot",
    page_icon="📈",
    layout="wide",
)

BASE_URL = "https://data-api.binance.vision"
SYMBOL = "BTCUSDT"
FEE_DEFAULT = 0.10  # Example fee percentage per side
POLL_SECONDS = 15

st.title("📈 Market Watch")
st.caption("Live market data · Simulated trades · No real money")

st.warning(
    "PAPER TRADING ONLY. This program has no trading credentials "
    "and cannot send real orders or connect to Phantom. All balances "
    "and trades shown here are simulated."
)


# ==================================================
# PUBLIC MARKET DATA
# ==================================================

def fetch_order_book():
    """Fetch public BTC/USDT order-book levels."""
    response = requests.get(
        f"{BASE_URL}/api/v3/depth",
        params={"symbol": SYMBOL, "limit": 100},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    bids = [
        (float(price), float(quantity))
        for price, quantity in data["bids"]
    ]
    asks = [
        (float(price), float(quantity))
        for price, quantity in data["asks"]
    ]

    if not bids or not asks:
        raise ValueError("The exchange returned an empty order book.")

    return {
        "bids": bids,
        "asks": asks,
        "time": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
    }


def fetch_signal():
    """
    Compute a 5-period versus 20-period simple moving average
    using closed one-minute candles.

    A bullish crossover creates a simulated buy signal.
    A bearish crossover creates a simulated exit signal.
    """
    response = requests.get(
        f"{BASE_URL}/api/v3/klines",
        params={
            "symbol": SYMBOL,
            "interval": "1m",
            "limit": 100,
        },
        timeout=10,
    )
    response.raise_for_status()
    raw = response.json()

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]

    candles = pd.DataFrame(raw, columns=columns)

    # Discard the last candle because it may still be forming.
    candles = candles.iloc[:-1].copy()
    candles["close"] = pd.to_numeric(candles["close"])

    candles["fast"] = candles["close"].rolling(5).mean()
    candles["slow"] = candles["close"].rolling(20).mean()

    valid = candles.dropna(subset=["fast", "slow"])

    if len(valid) < 2:
        raise ValueError("Not enough closed candles to calculate signals.")

    previous = valid.iloc[-2]
    latest = valid.iloc[-1]

    fast = float(latest["fast"])
    slow = float(latest["slow"])

    bullish = (
        float(previous["fast"]) <= float(previous["slow"])
        and fast > slow
    )

    bearish = (
        float(previous["fast"]) >= float(previous["slow"])
        and fast < slow
    )

    return {
        "fast": fast,
        "slow": slow,
        "bullish": bullish,
        "bearish": bearish,
        "trend": "Bullish" if fast > slow else "Bearish",
        "candle_key": int(latest["open_time"]),
        "close": float(latest["close"]),
    }


# ==================================================
# SIMULATED EXECUTION
# ==================================================

def buy_from_asks(asks, quote_budget):
    """
    Simulate a market buy by consuming ask levels.
    Returns (quantity, quote_spent), or None if depth is insufficient.
    """
    remaining = quote_budget
    quantity_bought = 0.0
    quote_spent = 0.0

    for price, available_qty in asks:
        spend = min(remaining, price * available_qty)

        if spend <= 0:
            continue

        quantity_bought += spend / price
        quote_spent += spend
        remaining -= spend

        if remaining <= max(1e-8, quote_budget * 1e-7):
            break

    if quantity_bought <= 0:
        return None

    if remaining > max(1e-8, quote_budget * 1e-6):
        return None

    return quantity_bought, quote_spent


def sell_to_bids(bids, quantity):
    """
    Simulate a market sell by consuming bid levels.
    Returns gross quote proceeds, or None if depth is insufficient.
    """
    remaining = quantity
    gross_proceeds = 0.0

    for price, available_qty in bids:
        sold = min(remaining, available_qty)

        if sold <= 0:
            continue

        gross_proceeds += sold * price
        remaining -= sold

        if remaining <= max(1e-12, quantity * 1e-7):
            break

    if remaining > max(1e-12, quantity * 1e-6):
        return None

    return gross_proceeds


def open_paper_position(paper, book, signal):
    """Buy BTC using only the simulated quote balance."""
    params = paper["params"]
    fee = params["fee_pct"] / 100.0

    # Reserve enough simulated cash to pay the estimated buy fee.
    budget = min(
        paper["cash"] / (1 + fee),
        paper["cash"] * params["allocation_pct"] / 100.0,
    )

    if budget <= 0:
        paper["note"] = "Insufficient simulated cash."
        return

    result = buy_from_asks(book["asks"], budget)

    if result is None:
        paper["note"] = "Not enough displayed ask depth to simulate the buy."
        return

    quantity, quote_spent = result
    buy_fee = quote_spent * fee
    total_cost = quote_spent + buy_fee

    if total_cost > paper["cash"] + 1e-7:
        paper["note"] = "Buy cancelled: simulated cash limit exceeded."
        return

    paper["cash"] = max(0.0, paper["cash"] - total_cost)

    paper["position"] = {
        "quantity": quantity,
        "entry_price": quote_spent / quantity,
        "entry_cost": total_cost,
        "entry_time": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "entry_candle": signal["candle_key"],
    }

    paper["note"] = (
        f"Simulated buy opened at approximately "
        f"${quote_spent / quantity:,.2f}."
    )


def close_paper_position(paper, book, reason):
    """Close a simulated position using current public bid levels."""
    position = paper["position"]

    if position is None:
        return True

    params = paper["params"]
    fee = params["fee_pct"] / 100.0

    gross = sell_to_bids(
        book["bids"],
        position["quantity"],
    )

    if gross is None:
        paper["note"] = (
            "Exit postponed: insufficient displayed bid depth. "
            "The simulated position remains open."
        )
        return False

    exit_fee = gross * fee
    net_proceeds = gross - exit_fee
    pnl = net_proceeds - position["entry_cost"]
    pnl_pct = pnl / position["entry_cost"] * 100.0

    paper["cash"] += net_proceeds

    paper["trades"].append({
        "Entry UTC": position["entry_time"],
        "Exit UTC": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "Quantity BTC": position["quantity"],
        "Entry price": position["entry_price"],
        "Exit price": gross / position["quantity"],
        "Entry cost incl. fee": position["entry_cost"],
        "Exit proceeds after fee": net_proceeds,
        "P/L USDT": pnl,
        "P/L %": pnl_pct,
        "Exit reason": reason,
    })

    paper["position"] = None
    paper["note"] = f"Position closed: {reason}."

    return True


# ==================================================
# BOT STATE
# ==================================================

def fresh_paper(balance, params):
    return {
        "running": True,
        "initial_balance": balance,
        "cash": balance,
        "started_at": time.time(),
        "params": params,
        "position": None,
        "trades": [],
        "history": [],
        "last_signal_key": None,
        "last_snapshot": None,
        "stop_reason": "",
        "note": "Waiting for the first market update.",
    }


if "paper" not in st.session_state:
    st.session_state.paper = fresh_paper(
        20.0,
        {
            "duration_minutes": 60,
            "fee_pct": FEE_DEFAULT,
            "allocation_pct": 100.0,
            "trade_tp_pct": 1.0,
            "trade_sl_pct": 0.75,
            "account_tp_pct": 5.0,
            "account_sl_pct": 5.0,
        },
    )
    st.session_state.paper["running"] = False

paper = st.session_state.paper


# ==================================================
# CONTROLS
# ==================================================

st.sidebar.title("Paper bot controls")

running = paper["running"]

balance_setting = st.sidebar.number_input(
    "Starting simulated balance (USDT)",
    min_value=1.0,
    max_value=10000.0,
    value=20.0,
    step=1.0,
    disabled=running,
)

allocation_setting = st.sidebar.slider(
    "Cash allocated per buy (%)",
    min_value=10,
    max_value=100,
    value=100,
    step=10,
    disabled=running,
)

fee_setting = st.sidebar.number_input(
    "Estimated fee per side (%)",
    min_value=0.0,
    max_value=2.0,
    value=FEE_DEFAULT,
    step=0.01,
    disabled=running,
    help="Illustrative input, not a verified exchange account fee.",
)

duration_setting = st.sidebar.number_input(
    "Maximum run time (minutes)",
    min_value=1,
    max_value=1440,
    value=60,
    step=5,
    disabled=running,
)

st.sidebar.subheader("Per-trade exits")

tp_setting = st.sidebar.number_input(
    "Take profit (%)",
    min_value=0.1,
    max_value=20.0,
    value=1.0,
    step=0.1,
    disabled=running,
)

sl_setting = st.sidebar.number_input(
    "Stop loss (%)",
    min_value=0.1,
    max_value=20.0,
    value=0.75,
    step=0.1,
    disabled=running,
)

st.sidebar.subheader("Whole-account limits")

account_tp_setting = st.sidebar.number_input(
    "Stop when account profit reaches (%)",
    min_value=0.1,
    max_value=100.0,
    value=5.0,
    step=0.5,
    disabled=running,
)

account_sl_setting = st.sidebar.number_input(
    "Stop when account loss reaches (%)",
    min_value=0.1,
    max_value=100.0,
    value=5.0,
    step=0.5,
    disabled=running,
)

start_clicked = st.sidebar.button(
    "Start paper bot",
    disabled=running,
    type="primary",
)

stop_clicked = st.sidebar.button(
    "Stop bot and close position",
    disabled=not running,
)

if start_clicked:
    params = {
        "duration_minutes": int(duration_setting),
        "fee_pct": float(fee_setting),
        "allocation_pct": float(allocation_setting),
        "trade_tp_pct": float(tp_setting),
        "trade_sl_pct": float(sl_setting),
        "account_tp_pct": float(account_tp_setting),
        "account_sl_pct": float(account_sl_setting),
    }

    st.session_state.paper = fresh_paper(
        float(balance_setting),
        params,
    )
    st.rerun()

if stop_clicked:
    paper = st.session_state.paper

    try:
        if paper["position"] is not None:
            latest_book = fetch_order_book()
            closed = close_paper_position(
                paper, latest_book, "Manual stop"
            )

            if not closed:
                st.rerun()

        paper["running"] = False
        paper["stop_reason"] = "Stopped manually."
        st.rerun()

    except (requests.RequestException, ValueError, KeyError) as exc:
        paper["note"] = (
            f"Could not obtain a quote to close the position: {exc}. "
            "The bot remains active; try stopping again."
        )
        st.rerun()


# ==================================================
# MARKET + TRADING LOOP
# ==================================================

run_every = "15s" if st.session_state.paper["running"] else None


@st.fragment(run_every=run_every)
def live_monitor():
    paper = st.session_state.paper
    was_running = paper["running"]

    book = None
    signal = None

    try:
        book = fetch_order_book()
        signal = fetch_signal()

        bid = book["bids"][0][0]
        ask = book["asks"][0][0]
        midpoint = (bid + ask) / 2.0
        spread_pct = (ask - bid) / midpoint * 100.0

        paper["last_snapshot"] = {
            "bid": bid,
            "ask": ask,
            "midpoint": midpoint,
            "spread_pct": spread_pct,
            "market_time": book["time"],
            "fast_sma": signal["fast"],
            "slow_sma": signal["slow"],
            "trend": signal["trend"],
        }

    except (requests.RequestException, ValueError, KeyError) as exc:
        st.error(
            f"Market data could not be updated: {exc}. "
            "The bot will not make a new trading decision on this failed update."
        )

    # No decisions are made unless both current data sources worked.
    if book is not None and signal is not None and paper["running"]:
        params = paper["params"]
        bid = book["bids"][0][0]
        ask = book["asks"][0][0]
        position = paper["position"]
        fee = params["fee_pct"] / 100.0

        # Mark the open position using estimated net liquidation value.
        equity = paper["cash"]

        if position is not None:
            equity += (
                position["quantity"] * bid * (1 - fee)
            )

        total_pnl = equity - paper["initial_balance"]
        elapsed = time.time() - paper["started_at"]

        # Whole-account targets and maximum run duration.
        stop_reason = None

        if total_pnl >= (
            paper["initial_balance"]
            * params["account_tp_pct"] / 100.0
        ):
            stop_reason = "Account take-profit target reached"

        elif total_pnl <= -(
            paper["initial_balance"]
            * params["account_sl_pct"] / 100.0
        ):
            stop_reason = "Account stop-loss limit reached"

        elif elapsed >= params["duration_minutes"] * 60:
            stop_reason = "Maximum run time reached"

        if stop_reason:
            if paper["position"] is not None:
                completed = close_paper_position(
                    paper, book, stop_reason
                )
                if not completed:
                    # Keep running and retry on a later update.
                    pass
                else:
                    paper["running"] = False
                    paper["stop_reason"] = stop_reason
            else:
                paper["running"] = False
                paper["stop_reason"] = stop_reason

        elif paper["position"] is not None:
            position = paper["position"]

            # Estimated profit/loss after both entry and hypothetical exit fees.
            estimated_exit = (
                position["quantity"] * bid * (1 - fee)
            )

            trade_pnl_pct = (
                (estimated_exit - position["entry_cost"])
                / position["entry_cost"]
                * 100.0
            )

            exit_reason = None

            if trade_pnl_pct >= params["trade_tp_pct"]:
                exit_reason = "Per-trade take profit"

            elif trade_pnl_pct <= -params["trade_sl_pct"]:
                exit_reason = "Per-trade stop loss"

            elif signal["bearish"]:
                exit_reason = "Bearish moving-average crossover"

            if exit_reason:
                close_paper_position(paper, book, exit_reason)

        else:
            # Only act once on each closed-candle crossover.
            key = signal["candle_key"]

            if (
                signal["bullish"]
                and key != paper["last_signal_key"]
            ):
                paper["last_signal_key"] = key
                open_paper_position(paper, book, signal)

        # Record account value for this active monitoring session.
        position = paper["position"]
        equity = paper["cash"]

        if position is not None:
            equity += (
                position["quantity"]
                * bid
                * (1 - params["fee_pct"] / 100.0)
            )

        paper["history"].append({
            "Time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "BTC/USDT midpoint": midpoint,
            "Account equity (USDT)": equity,
            "Account P/L (USDT)": equity - paper["initial_balance"],
        })

        paper["history"] = paper["history"][-500:]

    # Display current market data.
    st.subheader("Live BTC/USDT market")

    if paper["last_snapshot"] is not None:
        snapshot = paper["last_snapshot"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Best bid", f"${snapshot['bid']:,.2f}")
        c2.metric("Best ask", f"${snapshot['ask']:,.2f}")
        c3.metric("Bid-ask spread", f"{snapshot['spread_pct']:.4f}%")

        st.caption(
            f"Last successful market fetch: {snapshot['market_time']}"
        )

        st.subheader("Moving-average strategy")

        c1, c2 = st.columns(2)
        c1.metric("5-candle SMA", f"${snapshot['fast_sma']:,.2f}")
        c2.metric("20-candle SMA", f"${snapshot['slow_sma']:,.2f}")

        st.write(f"Current trend: **{snapshot['trend']}**")
        st.caption(
            "A simulated buy requires a fresh upward crossover. "
            "The bot may wait without opening a position."
        )

    # Portfolio statistics.
    paper = st.session_state.paper
    position = paper["position"]
    fee = paper["params"]["fee_pct"] / 100.0

    equity = paper["cash"]
    unrealised = 0.0

    if position is not None and paper["last_snapshot"] is not None:
        bid = paper["last_snapshot"]["bid"]
        estimated_exit = position["quantity"] * bid * (1 - fee)
        unrealised = estimated_exit - position["entry_cost"]
        equity += estimated_exit

    total_pnl = equity - paper["initial_balance"]
    pnl_pct = (
        total_pnl / paper["initial_balance"] * 100.0
        if paper["initial_balance"]
        else 0
    )

    st.subheader("Simulated account")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Starting balance", f"${paper['initial_balance']:.2f}")
    c2.metric("Estimated equity", f"${equity:.4f}")
    c3.metric("Net P/L", f"${total_pnl:+.4f}", f"{pnl_pct:+.3f}%")
    c4.metric("Closed trades", len(paper["trades"]))

    st.write(
        "**Bot status:** "
        + ("RUNNING" if paper["running"] else "STOPPED")
    )

    if position is not None:
        st.write(
            f"**Open simulated BTC position:** {position['quantity']:.8f} BTC "
            f"at an entry price of ${position['entry_price']:,.2f}"
        )
        st.write(
            f"Estimated open P/L after exit fee: ${unrealised:+.4f}"
        )
    else:
        st.write("**Open position:** None")

    if paper["stop_reason"]:
        st.info(f"Stop reason: {paper['stop_reason']}")

    if paper["note"]:
        st.caption(paper["note"])

    st.caption(
        "Estimated equity uses the best displayed bid and estimated sell fee. "
        "It is not a guaranteed liquidation value. Strategy decisions use "
        "closed candles, so signals are not instantaneous."
    )

    # History chart.
    st.subheader("Account performance")

    if paper["history"]:
        hist = pd.DataFrame(paper["history"])

        st.line_chart(
            hist.set_index("Time")[["Account equity (USDT)"]]
        )

        st.line_chart(
            hist.set_index("Time")[["BTC/USDT midpoint"]]
        )
    else:
        st.info("Start the bot to begin collecting performance history.")

    # Trade log.
    st.subheader("Paper trade history")

    if paper["trades"]:
        trades_df = pd.DataFrame(paper["trades"])
        st.dataframe(
            trades_df,
            hide_index=True,
            use_container_width=True,
        )

        total_closed_pnl = sum(
            row["P/L USDT"] for row in paper["trades"]
        )

        st.metric("Realised P/L from closed trades", f"${total_closed_pnl:+.4f}")
    else:
        st.info("No simulated trades have closed yet.")

    if was_running and not paper["running"]:
        # Rebuild the fragment with automatic refreshing disabled.
        st.rerun()


live_monitor()


# ==================================================
# OPTIONAL GROQ EXPLANATION
# ==================================================

st.divider()
st.subheader("AI explanation")

if st.button("Explain bot performance"):
    try:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = os.environ.get("GROQ_API_KEY", "")

        if not api_key:
            st.warning(
                "Add GROQ_API_KEY under your Streamlit app's Secrets settings."
            )
        else:
            paper = st.session_state.paper

            snapshot = paper["last_snapshot"]
            trades = paper["trades"][-10:]

            client = Groq(api_key=api_key)

            with st.spinner("Explaining the simulation..."):
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Explain paper-trading results to a beginner. "
                                "Use only supplied data. Do not invent "
                                "prices or claim returns are guaranteed. "
                                "Distinguish simulation from live execution. "
                                "Mention fees, spread, slippage and delays. "
                                "Do not recommend real-money trades."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Latest market snapshot: {snapshot}\n"
                                f"Initial simulated balance: "
                                f"{paper['initial_balance']}\n"
                                f"Current cash: {paper['cash']}\n"
                                f"Closed trades: {trades}\n"
                                f"Bot running: {paper['running']}\n"
                                "Explain the results and limitations."
                            ),
                        },
                    ],
                    temperature=0.2,
                    max_tokens=400,
                )

            st.write(response.choices[0].message.content)

    except Exception as exc:
        st.error(f"AI explanation failed: {exc}")


st.caption(
    "Educational paper simulation only. No real orders, exchange credentials, "
    "wallet connection or private keys are used."
)

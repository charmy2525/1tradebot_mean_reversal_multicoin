"""Futures Fibonacci + RSI based trading bot.

This module implements a trading routine for Binance USDT perpetual futures
contracts using four hour candles, RSI(14) swing detection and Fibonacci
extension rules as described in the task instructions.  The core logic is
contained in the :class:`FuturesFibRsiBot` class which can be reused inside the
project or executed directly as a standalone script.

The bot supports the BTCUSDT, ETHUSDT and SOLUSDT symbols and assumes one-way
mode on the futures account.  Orders can be simulated by running the bot in
``dry_run`` mode which keeps all order placements in memory while still
producing detailed logs.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from binance.client import Client
from binance.exceptions import BinanceAPIException


def rsi(closes: Sequence[float], period: int = 14) -> List[Optional[float]]:
    """Return a list containing the RSI for every close in ``closes``.

    The first ``period`` values are ``None`` because an RSI value cannot be
    computed for them.  Afterwards the classical Wilder smoothing method is
    used to calculate the indicator.
    """

    if len(closes) < period + 1:
        return [None for _ in closes]

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(delta, 0.0) for delta in deltas]
    losses = [abs(min(delta, 0.0)) for delta in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    rsis: List[Optional[float]] = [None] * period
    rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
    rsis.append(100.0 - (100.0 / (1.0 + rs)))

    for i in range(period, len(deltas)):
        gain = gains[i]
        loss = losses[i]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsis.append(100.0 - 100.0 / (1.0 + rs))

    rsis.insert(0, None)
    return rsis


def _find_fractal(
    values: Sequence[float],
    start_index: int,
    fractal_type: str,
    lookback: int = 2,
) -> int:
    """Locate a fractal (swing high/low) before ``start_index``.

    The function searches backwards for the first index where the value is
    strictly greater/less than the surrounding ``lookback`` candles depending
    on ``fractal_type``.
    """

    assert fractal_type in {"high", "low"}

    end = len(values) - lookback
    for idx in range(start_index, lookback - 1, -1):
        if idx >= end:
            continue
        window = values[idx - lookback : idx + lookback + 1]
        if len(window) < lookback * 2 + 1:
            continue

        center = values[idx]
        if fractal_type == "high" and center == max(window) and window.count(center) == 1:
            return idx
        if fractal_type == "low" and center == min(window) and window.count(center) == 1:
            return idx

    return max(start_index, 0)


def _slice_extremum(
    values: Sequence[float],
    start: int,
    end: int,
    find: str,
) -> Tuple[int, float]:
    """Return ``(index, value)`` for an extremum of ``values[start:end]``."""

    assert find in {"min", "max"}
    if end <= start:
        end = start + 1

    segment = list(enumerate(values[start:end], start))
    if find == "min":
        return min(segment, key=lambda v: v[1])
    return max(segment, key=lambda v: v[1])


def _fib_price(anchor: float, range_size: float, percent: float) -> float:
    """Return the Fibonacci price for ``percent``.

    ``anchor`` is the 0% level, ``range_size`` the distance between 0% and
    ±100%, ``percent`` is the requested level.
    """

    return anchor + range_size * (percent / 100.0)


DOWNTREND_LEVELS: Tuple[float, ...] = (
    -23.6,
    -38.2,
    -50.0,
    -61.8,
    -78.6,
    -100.0,
    -127.2,
    -138.2,
    -150.0,
    -161.8,
    -200.0,
    -261.8,
)

UPTREND_LEVELS: Tuple[float, ...] = (
    23.6,
    38.2,
    50.0,
    61.8,
    78.6,
    100.0,
    127.2,
    138.2,
    150.0,
    161.8,
    200.0,
    261.8,
)

DOWN_STOP_STEPS: Tuple[Tuple[float, float], ...] = (
    (-38.2, -12.8),
    (-50.0, -30.0),
    (-61.8, -38.2),
    (-78.6, -45.0),
    (-100.0, -60.0),
    (-127.2, -75.0),
    (-138.2, -98.0),
    (-150.0, -125.0),
    (-161.8, -135.0),
    (-200.0, -148.0),
    (-261.8, -198.0),
)

UP_STOP_STEPS: Tuple[Tuple[float, float], ...] = (
    (38.2, 12.8),
    (50.0, 30.0),
    (61.8, 38.2),
    (78.6, 45.0),
    (100.0, 60.0),
    (127.2, 75.0),
    (138.2, 98.0),
    (150.0, 125.0),
    (161.8, 135.0),
    (200.0, 148.0),
    (261.8, 198.0),
)


@dataclass
class PositionState:
    """Keeps track of the currently managed position for a symbol."""

    trend: str
    fib_zero: float
    fib_range: float
    entry_level: float
    stop_price: float
    filled_price: Optional[float] = None
    last_update_price: Optional[float] = None
    active: bool = False
    def level_price(self, percent: float) -> float:
        return _fib_price(self.fib_zero, self.fib_range, percent)


class FuturesFibRsiBot:
    """RSI/Fibonacci based futures trading logic."""

    SUPPORTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

    def __init__(
        self,
        client: Client,
        symbols: Iterable[str] = SUPPORTED_SYMBOLS,
        leverage: int = 3,
        risk_per_trade: float = 0.01,
        poll_interval: int = 60,
        dry_run: bool = True,
    ) -> None:
        self.client = client
        self.symbols = tuple(symbol.upper() for symbol in symbols if symbol.upper() in self.SUPPORTED_SYMBOLS)
        if not self.symbols:
            raise ValueError("At least one supported symbol must be provided")
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self.logger = logging.getLogger(self.__class__.__name__)
        self.positions: Dict[str, PositionState] = {}

    # ---------------------------------------------------------------------
    # Market data utilities
    # ---------------------------------------------------------------------
    def _fetch_klines(self, symbol: str, limit: int = 500) -> List[Dict[str, float]]:
        raw = self.client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_4HOUR, limit=limit)
        klines = []
        for entry in raw:
            klines.append(
                {
                    "open_time": datetime.fromtimestamp(entry[0] / 1000.0, tz=timezone.utc),
                    "open": float(entry[1]),
                    "high": float(entry[2]),
                    "low": float(entry[3]),
                    "close": float(entry[4]),
                    "volume": float(entry[5]),
                }
            )
        return klines

    def _determine_trend(self, rsi_values: Sequence[Optional[float]]) -> Tuple[Optional[str], Optional[int]]:
        last_overbought = None
        last_oversold = None
        for idx, value in enumerate(rsi_values):
            if value is None:
                continue
            if value >= 70.0:
                last_overbought = idx
            if value <= 30.0:
                last_oversold = idx

        if last_overbought is None and last_oversold is None:
            return None, None

        if last_overbought is not None and (last_oversold is None or last_overbought > last_oversold):
            return "downtrend", last_overbought
        if last_oversold is not None and (last_overbought is None or last_oversold > last_overbought):
            return "uptrend", last_oversold

        return None, None

    def _downtrend_context(
        self, highs: Sequence[float], lows: Sequence[float], trend_index: int
    ) -> Optional[Tuple[float, float]]:
        fractal_index = _find_fractal(highs, trend_index, "high")
        low_index, low_value = _slice_extremum(lows, fractal_index, trend_index + 1, "min")
        _, high_value = _slice_extremum(highs, low_index, len(highs), "max")

        if high_value <= low_value:
            return None

        return high_value, high_value - low_value

    def _uptrend_context(
        self, highs: Sequence[float], lows: Sequence[float], trend_index: int
    ) -> Optional[Tuple[float, float]]:
        fractal_index = _find_fractal(lows, trend_index, "low")
        high_index, high_value = _slice_extremum(highs, fractal_index, trend_index + 1, "max")
        _, low_value = _slice_extremum(lows, high_index, len(lows), "min")

        if high_value <= low_value:
            return None

        return low_value, high_value - low_value

    # ------------------------------------------------------------------
    # Trading utilities
    # ------------------------------------------------------------------
    def _ensure_leverage(self, symbol: str) -> None:
        if self.dry_run:
            return
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=self.leverage)
        except BinanceAPIException as exc:  # pragma: no cover - network dependent
            self.logger.warning("Unable to change leverage for %s: %s", symbol, exc)

    def _account_balance(self) -> float:
        balances = self.client.futures_account_balance()
        for entry in balances:
            if entry["asset"] == "USDT":
                return float(entry["balance"])
        raise RuntimeError("Unable to locate USDT balance")

    def _position_information(self, symbol: str) -> Optional[Dict[str, float]]:
        positions = self.client.futures_position_information(symbol=symbol)
        if not positions:
            return None
        position = positions[0]
        if abs(float(position["positionAmt"])) < 1e-8:
            return None
        return position

    def _place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = Client.FUTURE_ORDER_TYPE_MARKET,
        **params,
    ) -> Optional[Dict[str, object]]:
        self.logger.info("Placing order %s %s qty=%s params=%s", side, symbol, quantity, params)
        if self.dry_run:
            return {
                "symbol": symbol,
                "side": side,
                "executedQty": quantity,
                "updateTime": datetime.utcnow().isoformat(),
                "price": params.get("price"),
            }
        return self.client.futures_create_order(symbol=symbol, side=side, type=order_type, quantity=quantity, **params)

    def _place_stop(self, symbol: str, side: str, stop_price: float) -> None:
        self.logger.info("Updating stop for %s at %.2f", symbol, stop_price)
        if self.dry_run:
            return
        self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type=Client.FUTURE_ORDER_TYPE_STOP_MARKET,
            stopPrice=stop_price,
            closePosition=True,
            workingType="CONTRACT_PRICE",
        )

    def _calculate_order_size(self, symbol: str, stop_price: float, entry_price: float) -> float:
        balance = self._account_balance()
        risk_amount = balance * self.risk_per_trade
        tick_size = self._symbol_tick_size(symbol)
        contract_value = abs(entry_price - stop_price)
        if contract_value == 0:
            return 0.0
        quantity = risk_amount / contract_value * entry_price / self.leverage
        quantity = max(round(quantity / tick_size) * tick_size, tick_size)
        return quantity

    def _symbol_tick_size(self, symbol: str) -> float:
        info = self.client.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        return float(f["stepSize"])
        raise RuntimeError(f"Unable to determine tick size for {symbol}")

    # ------------------------------------------------------------------
    # Strategy execution
    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        while True:
            for symbol in self.symbols:
                try:
                    self.run_once(symbol)
                except Exception as exc:  # pragma: no cover - defensive measure
                    self.logger.exception("Error while processing %s: %s", symbol, exc)
            time.sleep(self.poll_interval)

    def run_once(self, symbol: str) -> None:
        klines = self._fetch_klines(symbol)
        closes = [candle["close"] for candle in klines]
        highs = [candle["high"] for candle in klines]
        lows = [candle["low"] for candle in klines]
        indicator = rsi(closes)

        trend, trend_index = self._determine_trend(indicator)
        if trend is None or trend_index is None:
            self.logger.info("%s - no actionable trend", symbol)
            return

        state = self.positions.get(symbol)

        if trend == "downtrend":
            context = self._downtrend_context(highs, lows, trend_index)
            if context is None:
                self.logger.info("%s - unable to determine downtrend structure", symbol)
                return
            zero_level, range_size = context
            self._execute_downtrend(symbol, klines, zero_level, range_size, state)
        else:
            context = self._uptrend_context(highs, lows, trend_index)
            if context is None:
                self.logger.info("%s - unable to determine uptrend structure", symbol)
                return
            zero_level, range_size = context
            self._execute_uptrend(symbol, klines, zero_level, range_size, state)

    def _execute_downtrend(
        self,
        symbol: str,
        klines: Sequence[Dict[str, float]],
        zero_level: float,
        range_size: float,
        state: Optional[PositionState],
    ) -> None:
        levels = {percent: _fib_price(zero_level, range_size, percent) for percent in DOWNTREND_LEVELS}

        last_close = klines[-1]["close"]
        last_high = klines[-1]["high"]
        prev_close = klines[-2]["close"]

        entry_level = levels[-23.6]
        crossed = prev_close >= entry_level and last_close < entry_level

        if state is None or not state.active:
            if not crossed:
                return
            stop_price = last_high
            position = PositionState(
                trend="downtrend",
                fib_zero=zero_level,
                fib_range=range_size,
                entry_level=-23.6,
                stop_price=stop_price,
                active=True,
            )
            quantity = self._calculate_order_size(symbol, stop_price, last_close)
            if quantity == 0:
                self.logger.warning("%s - calculated quantity is zero, skipping trade", symbol)
                return
            self._ensure_leverage(symbol)
            order = self._place_order(symbol, side=Client.SIDE_SELL, quantity=quantity)
            position.filled_price = last_close if order is None else float(order.get("price") or last_close)
            position.last_update_price = last_close
            self._place_stop(symbol, Client.SIDE_BUY, stop_price)
            self.positions[symbol] = position
            self.logger.info("%s - entered short at %.2f (stop %.2f)", symbol, last_close, stop_price)
            return

        self._trail_downtrend_stop(symbol, state, last_close)

    def _trail_downtrend_stop(self, symbol: str, state: PositionState, last_price: float) -> None:
        updated = False
        for trigger, new_stop in DOWN_STOP_STEPS:
            trigger_price = state.level_price(trigger)
            if last_price <= trigger_price:
                stop_candidate = state.level_price(new_stop)
                if stop_candidate > state.stop_price:
                    state.stop_price = stop_candidate
                    updated = True
        if updated:
            self._place_stop(symbol, Client.SIDE_BUY, state.stop_price)
            state.last_update_price = last_price
            self.logger.info("%s - updated short stop to %.2f", symbol, state.stop_price)

    def _execute_uptrend(
        self,
        symbol: str,
        klines: Sequence[Dict[str, float]],
        zero_level: float,
        range_size: float,
        state: Optional[PositionState],
    ) -> None:
        levels = {percent: _fib_price(zero_level, range_size, percent) for percent in UPTREND_LEVELS}

        last_close = klines[-1]["close"]
        last_low = klines[-1]["low"]
        prev_close = klines[-2]["close"]

        entry_level = levels[23.6]
        crossed = prev_close <= entry_level and last_close > entry_level

        if state is None or not state.active:
            if not crossed:
                return
            stop_price = last_low
            position = PositionState(
                trend="uptrend",
                fib_zero=zero_level,
                fib_range=range_size,
                entry_level=23.6,
                stop_price=stop_price,
                active=True,
            )
            quantity = self._calculate_order_size(symbol, stop_price, last_close)
            if quantity == 0:
                self.logger.warning("%s - calculated quantity is zero, skipping trade", symbol)
                return
            self._ensure_leverage(symbol)
            order = self._place_order(symbol, side=Client.SIDE_BUY, quantity=quantity)
            position.filled_price = last_close if order is None else float(order.get("price") or last_close)
            position.last_update_price = last_close
            self._place_stop(symbol, Client.SIDE_SELL, stop_price)
            self.positions[symbol] = position
            self.logger.info("%s - entered long at %.2f (stop %.2f)", symbol, last_close, stop_price)
            return

        self._trail_uptrend_stop(symbol, state, last_close)

    def _trail_uptrend_stop(self, symbol: str, state: PositionState, last_price: float) -> None:
        updated = False
        for trigger, new_stop in UP_STOP_STEPS:
            trigger_price = state.level_price(trigger)
            if last_price >= trigger_price:
                stop_candidate = state.level_price(new_stop)
                if stop_candidate < state.stop_price:
                    state.stop_price = stop_candidate
                    updated = True
        if updated:
            self._place_stop(symbol, Client.SIDE_SELL, state.stop_price)
            state.last_update_price = last_price
            self.logger.info("%s - updated long stop to %.2f", symbol, state.stop_price)


def _build_client_from_env() -> Client:
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise EnvironmentError("BINANCE_API_KEY and BINANCE_API_SECRET must be set")
    return Client(api_key, api_secret)


def main() -> None:  # pragma: no cover - entry point convenience
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    client = _build_client_from_env()
    symbols = os.environ.get("PERP_SYMBOLS")
    selected = symbols.split(",") if symbols else FuturesFibRsiBot.SUPPORTED_SYMBOLS
    dry_run = os.environ.get("PERP_DRY_RUN", "true").lower() in {"1", "true", "yes"}
    leverage = int(os.environ.get("PERP_LEVERAGE", "3"))
    risk = float(os.environ.get("PERP_RISK", "0.01"))
    poll = int(os.environ.get("PERP_POLL", "60"))

    bot = FuturesFibRsiBot(
        client=client,
        symbols=[symbol.strip() for symbol in selected],
        leverage=leverage,
        risk_per_trade=risk,
        poll_interval=poll,
        dry_run=dry_run,
    )
    bot.run_forever()


if __name__ == "__main__":  # pragma: no cover - script execution
    main()


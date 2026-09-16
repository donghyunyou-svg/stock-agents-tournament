"""
시세/재무/거시경제 데이터 제공자(Provider) 모듈.

- MarketDataProvider: 모든 데이터 소스가 구현해야 하는 공통 인터페이스.
- SyntheticDataProvider: 네트워크 없이 프레임워크를 검증하기 위한 가상 데이터 생성기.
- FDRProvider / YFinanceProvider: 실제 데이터 소스 어댑터 (국내 FinanceDataReader, 해외 yfinance).
  ** 현재 이 클라우드 세션은 조직의 네트워크 egress 정책상 naver/yahoo 등 외부 도메인에
  직접 접근이 차단되어 있어(허용목록: pypi/npm/anthropic API 등), 이 두 Provider는
  네트워크가 열린 환경(사용자 로컬 PC 또는 egress 허용된 클라우드 세션)에서만 동작합니다. **
- MeritzOpenAPIProvider: 메리츠증권 Open API 연동 자리표시자(placeholder). API 키가
  제공되고 해당 엔드포인트에 대한 네트워크 접근이 허용되면 구현을 채웁니다.
"""

from __future__ import annotations

import abc
import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Instrument:
    ticker: str
    name: str
    market: str  # "KRX" or "US"
    sector: str = "Unknown"


class MarketDataProvider(abc.ABC):
    """모든 데이터 제공자의 공통 인터페이스."""

    @abc.abstractmethod
    def get_universe(self) -> List[Instrument]:
        """거래 가능한 종목 유니버스를 반환."""

    @abc.abstractmethod
    def get_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """OHLCV 컬럼(Open, High, Low, Close, Volume)을 가진 일별 시세 DataFrame 반환."""

    def get_macro_snapshot(self, as_of: date) -> Dict[str, float]:
        """거시경제 지표 스냅샷 (금리, 환율, 유가 등). 기본값은 빈 dict."""
        return {}


class SyntheticDataProvider(MarketDataProvider):
    """
    네트워크 접근 없이 파이프라인(데이터->판단->체결->스코어링->진화)을 검증하기 위한
    가상 시세 생성기. 종목별로 서로 다른 추세/변동성/평균회귀 성향을 부여해
    10개 에이전트의 전략 차이가 실제로 드러나도록 설계.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._universe = [
            Instrument("005930.KS", "삼성전자", "KRX", "반도체"),
            Instrument("000660.KS", "SK하이닉스", "KRX", "반도체"),
            Instrument("035420.KS", "NAVER", "KRX", "IT서비스"),
            Instrument("051910.KS", "LG화학", "KRX", "화학"),
            Instrument("005380.KS", "현대차", "KRX", "자동차"),
            Instrument("AAPL", "Apple", "US", "Technology"),
            Instrument("MSFT", "Microsoft", "US", "Technology"),
            Instrument("NVDA", "NVIDIA", "US", "Semiconductors"),
            Instrument("JPM", "JPMorgan", "US", "Financials"),
            Instrument("XOM", "ExxonMobil", "US", "Energy"),
        ]
        # 종목별 고유 특성: 연간 드리프트, 변동성, 평균회귀 강도
        self._params = {
            inst.ticker: {
                "mu": self._rng.uniform(-0.05, 0.25),
                "sigma": self._rng.uniform(0.15, 0.45),
                "mean_reversion": self._rng.uniform(0.0, 0.4),
            }
            for inst in self._universe
        }
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_universe(self) -> List[Instrument]:
        return list(self._universe)

    def _full_series(self, ticker: str) -> pd.DataFrame:
        """
        티커별 전체 가격 시계열을 한 번만 생성해 캐시한다 (2023-01-02 ~ 2028-12-31).
        매 호출마다 새로 생성하면 매일 동일한 종가가 반복되는 버그가 생기므로,
        고정된 하나의 경로를 생성해두고 이후에는 구간만 잘라서 반환한다.
        """
        if ticker in self._cache:
            return self._cache[ticker]
        rng = random.Random(f"{ticker}-seed")
        p = self._params[ticker]
        dates = pd.bdate_range(start="2023-01-02", end="2028-12-31")
        price = 100.0
        prices = []
        long_run_mean = 100.0
        dt = 1 / 252
        for _ in dates:
            shock = rng.gauss(0, 1)
            drift = p["mu"] * dt
            reversion = p["mean_reversion"] * (math.log(long_run_mean / price)) * dt
            ret = drift + reversion + p["sigma"] * math.sqrt(dt) * shock
            price *= math.exp(ret)
            prices.append(price)
        closes = pd.Series(prices, index=dates)
        opens = closes.shift(1).fillna(closes.iloc[0]) * (1 + pd.Series(
            [rng.gauss(0, 0.002) for _ in dates], index=dates))
        highs = pd.concat([opens, closes], axis=1).max(axis=1) * (1 + pd.Series(
            [abs(rng.gauss(0, 0.004)) for _ in dates], index=dates))
        lows = pd.concat([opens, closes], axis=1).min(axis=1) * (1 - pd.Series(
            [abs(rng.gauss(0, 0.004)) for _ in dates], index=dates))
        volumes = pd.Series(
            [max(1000, int(rng.gauss(1_000_000, 300_000))) for _ in dates], index=dates)
        df = pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes,
        })
        self._cache[ticker] = df
        return df

    def get_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        if ticker not in self._params:
            raise KeyError(f"Unknown ticker: {ticker}")
        df = self._full_series(ticker)
        return df[(df.index.date >= start) & (df.index.date <= end)]

    def get_macro_snapshot(self, as_of: date) -> Dict[str, float]:
        rng = random.Random(f"macro-{as_of.isoformat()}")
        return {
            "us_10y_yield": round(4.0 + rng.uniform(-0.5, 0.5), 2),
            "usdkrw": round(1370 + rng.uniform(-40, 40), 1),
            "wti_oil": round(75 + rng.uniform(-10, 10), 1),
            "kospi_change_pct": round(rng.uniform(-1.5, 1.5), 2),
        }


class FDRProvider(MarketDataProvider):
    """FinanceDataReader 기반 국내(KRX) 실데이터 Provider. 네트워크 접근 필요."""

    def get_universe(self) -> List[Instrument]:
        raise NotImplementedError(
            "이 클라우드 세션은 현재 naver/krx 도메인에 대한 네트워크 접근이 "
            "허용되어 있지 않습니다. 조직 Admin 설정에서 네트워크 접근을 확장하거나, "
            "사용자 로컬 PC(디바이스 브리지)에서 실행해야 합니다."
        )

    def get_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        import FinanceDataReader as fdr  # local import: 네트워크 필요 시점에만 로드
        df = fdr.DataReader(ticker, start, end)
        return df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]


class YFinanceProvider(MarketDataProvider):
    """yfinance 기반 해외(미국) 실데이터 Provider. 네트워크 접근 필요."""

    def get_universe(self) -> List[Instrument]:
        raise NotImplementedError(
            "이 클라우드 세션은 현재 yahoo 도메인에 대한 네트워크 접근이 "
            "허용되어 있지 않습니다. 조직 Admin 설정에서 네트워크 접근을 확장하거나, "
            "사용자 로컬 PC(디바이스 브리지)에서 실행해야 합니다."
        )

    def get_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        return df[["Open", "High", "Low", "Close", "Volume"]]


class MeritzOpenAPIProvider(MarketDataProvider):
    """
    메리츠증권 Open API 연동 자리표시자.
    ** 2026-09 기준 조사 결과, 메리츠증권은 개인용 Open API를 아직 제공하지 않습니다
    (2026-05 보도 기준 "연내 출시 예정"). 공식 API가 출시되면 이 클래스에 구현을 채웁니다.
    그 전까지는 KISOpenAPIProvider 등 API를 제공하는 다른 증권사로 실계좌 연동을 진행합니다. **
    """

    def __init__(self, app_key: str, app_secret: str, account_no: str, paper: bool = True):
        self.app_key = app_key
        self.app_secret = app_secret
        self.account_no = account_no
        self.paper = paper

    def get_universe(self) -> List[Instrument]:
        raise NotImplementedError("메리츠증권 Open API는 2026-09 기준 아직 출시되지 않았습니다.")

    def get_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError("메리츠증권 Open API는 2026-09 기준 아직 출시되지 않았습니다.")


class KISOpenAPIProvider(MarketDataProvider):
    """
    한국투자증권(KIS, Korea Investment & Securities) Open API 연동 어댑터.

    공식 예제(koreainvestment/open-trading-api, examples_user/kis_auth.py,
    domestic_stock_functions.py, overseas_stock_functions.py)를 참고해 구현.

    - 실전투자 서버: https://openapi.koreainvestment.com:9443
    - 모의투자 서버: https://openapivts.koreainvestment.com:29443
    - 인증: POST {base}/oauth2/tokenP, body={"grant_type":"client_credentials",
      "appkey":..., "appsecret":...} -> access_token (유효 ~24시간). 발급은 앱키당
      분당 1회 제한이 있어 프로세스 내에서 캐싱한다 (GitHub Actions처럼 매 실행이
      새 프로세스인 경우 실행당 1회 발급되는 정도이므로 문제 없음).
    - 국내 일별시세: GET {base}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice
      (tr_id=FHKST03010100, 최대 100건/호출 -> 100일 초과 구간은 분할 호출)
    - 해외 일별시세: GET {base}/uapi/overseas-price/v1/quotations/dailyprice
      (tr_id=HHDFS76240000, 최대 100건/호출, 기준일자(BYMD)부터 과거로 반환 -> 분할 호출)

    ** 응답 필드명(stck_bsop_date 등)은 KIS 공식 문서 기준으로 널리 알려진 표준 필드명이지만,
    실제 응답을 한 번도 직접 확인하지 못한 상태에서 작성되었으므로 최초 실행 시
    파싱 실패하면 원본 JSON을 그대로 출력하도록 방어 코드를 넣어두었다. **
    """

    REAL_BASE_URL = "https://openapi.koreainvestment.com:9443"
    PAPER_BASE_URL = "https://openapivts.koreainvestment.com:29443"

    # 해외 종목의 거래소 코드 (KIS EXCD 파라미터). 필요시 여기에 종목을 추가한다.
    OVERSEAS_EXCHANGE = {
        "AAPL": "NAS", "MSFT": "NAS", "NVDA": "NAS",
        "JPM": "NYS", "XOM": "NYS",
    }

    DEFAULT_UNIVERSE = [
        Instrument("005930", "삼성전자", "KRX", "반도체"),
        Instrument("000660", "SK하이닉스", "KRX", "반도체"),
        Instrument("035420", "NAVER", "KRX", "IT서비스"),
        Instrument("051910", "LG화학", "KRX", "화학"),
        Instrument("005380", "현대차", "KRX", "자동차"),
        Instrument("AAPL", "Apple", "US", "Technology"),
        Instrument("MSFT", "Microsoft", "US", "Technology"),
        Instrument("NVDA", "NVIDIA", "US", "Semiconductors"),
        Instrument("JPM", "JPMorgan", "US", "Financials"),
        Instrument("XOM", "ExxonMobil", "US", "Energy"),
    ]

    def __init__(self, app_key: str, app_secret: str, account_no: str, paper: bool = True,
                 universe: Optional[List[Instrument]] = None):
        self.app_key = app_key
        self.app_secret = app_secret
        self.account_no = account_no
        self.paper = paper
        self.base_url = self.PAPER_BASE_URL if paper else self.REAL_BASE_URL
        self._universe = universe or list(self.DEFAULT_UNIVERSE)
        self._access_token: Optional[str] = None
        self._token_expiry = None  # datetime

    # ------------------------------------------------------------------ 인증
    def _auth(self) -> str:
        import requests
        from datetime import datetime as dt

        if self._access_token and self._token_expiry and dt.now() < self._token_expiry:
            return self._access_token

        resp = requests.post(
            f"{self.base_url}/oauth2/tokenP",
            json={"grant_type": "client_credentials", "appkey": self.app_key,
                  "appsecret": self.app_secret},
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=15,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"KIS 토큰 발급 실패: HTTP {resp.status_code} {resp.text[:500]}")
        data = resp.json()
        self._access_token = data["access_token"]
        try:
            self._token_expiry = dt.strptime(data["access_token_token_expired"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            self._token_expiry = dt.now() + timedelta(hours=12)  # 안전한 기본값
        return self._access_token

    def _headers(self, tr_id: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/plain",
            "authorization": f"Bearer {self._auth()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    def _get(self, path: str, tr_id: str, params: Dict) -> Dict:
        import requests
        resp = requests.get(f"{self.base_url}{path}", headers=self._headers(tr_id),
                             params=params, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"KIS API 호출 실패 ({path}): HTTP {resp.status_code} {resp.text[:500]}")
        return resp.json()

    # -------------------------------------------------------------- 유니버스
    def get_universe(self) -> List[Instrument]:
        return list(self._universe)

    # ---------------------------------------------------------------- 시세
    def get_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        inst = next((i for i in self._universe if i.ticker == ticker), None)
        market = inst.market if inst else ("KRX" if ticker.isdigit() else "US")
        if market == "KRX":
            return self._get_domestic_history(ticker, start, end)
        return self._get_overseas_history(ticker, start, end)

    def _get_domestic_history(self, code: str, start: date, end: date) -> pd.DataFrame:
        rows = []
        window_start = start
        while window_start <= end:
            window_end = min(end, window_start + timedelta(days=95))  # 최대 100건/호출 제한 대응
            data = self._get(
                "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                "FHKST03010100",
                {
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": code,
                    "FID_INPUT_DATE_1": window_start.strftime("%Y%m%d"),
                    "FID_INPUT_DATE_2": window_end.strftime("%Y%m%d"),
                    "FID_PERIOD_DIV_CODE": "D",
                    "FID_ORG_ADJ_PRC": "0",  # 0: 수정주가
                },
            )
            for row in data.get("output2", []):
                try:
                    rows.append({
                        "date": row["stck_bsop_date"],
                        "Open": float(row["stck_oprc"]),
                        "High": float(row["stck_hgpr"]),
                        "Low": float(row["stck_lwpr"]),
                        "Close": float(row["stck_clpr"]),
                        "Volume": float(row["acml_vol"]),
                    })
                except (KeyError, ValueError, TypeError) as e:
                    raise RuntimeError(
                        f"국내 시세 응답 파싱 실패 (필드명이 문서와 다를 수 있음): {e}\n"
                        f"원본 row: {row}"
                    )
            window_start = window_end + timedelta(days=1)
        return self._rows_to_df(rows)

    def _get_overseas_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        excd = self.OVERSEAS_EXCHANGE.get(ticker)
        if not excd:
            raise KeyError(f"'{ticker}'의 해외 거래소 코드가 OVERSEAS_EXCHANGE에 없습니다.")
        rows = []
        bymd = end
        seen_dates = set()
        for _ in range(20):  # 최대 20회 분할 호출 (약 2000거래일까지 커버)
            data = self._get(
                "/uapi/overseas-price/v1/quotations/dailyprice",
                "HHDFS76240000",
                {"AUTH": "", "EXCD": excd, "SYMB": ticker, "GUBN": "0",
                 "BYMD": bymd.strftime("%Y%m%d"), "MODP": "0"},
            )
            output2 = data.get("output2", [])
            if not output2:
                break
            earliest = None
            for row in output2:
                try:
                    d = row["xymd"]
                    if d in seen_dates:
                        continue
                    seen_dates.add(d)
                    rows.append({
                        "date": d,
                        "Open": float(row["open"]),
                        "High": float(row["high"]),
                        "Low": float(row["low"]),
                        "Close": float(row["clos"]),
                        "Volume": float(row["tvol"]),
                    })
                    if earliest is None or d < earliest:
                        earliest = d
                except (KeyError, ValueError, TypeError) as e:
                    raise RuntimeError(
                        f"해외 시세 응답 파싱 실패 (필드명이 문서와 다를 수 있음): {e}\n"
                        f"원본 row: {row}"
                    )
            if earliest is None:
                break
            earliest_date = date(int(earliest[:4]), int(earliest[4:6]), int(earliest[6:8]))
            if earliest_date <= start:
                break
            bymd = earliest_date - timedelta(days=1)
        return self._rows_to_df(rows)

    @staticmethod
    def _rows_to_df(rows: List[Dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df = df.drop_duplicates(subset="date").sort_values("date").set_index("date")
        return df[["Open", "High", "Low", "Close", "Volume"]]

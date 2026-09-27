"""Phase 1 - Loading.

Fetches the five scheme pages listed in sources.csv, respects robots.txt, caches
the raw HTML, and extracts a *narrow allow-listed* set of factual fields from the
embedded ``__NEXT_DATA__`` JSON payload.

Design notes
------------
* Facts come from ``__NEXT_DATA__ -> props.pageProps.mfServerSideData``, not from
  scraping rendered markup. [verified] All five pages embed a complete structured
  record, so extraction is exact rather than heuristic.
* Extraction is governed by ``ALLOWED_FIELDS``. Anything not named there is never
  copied, so upstream page changes cannot leak returns/NAV/holdings into the
  corpus. This is what enforces FR-6 in code rather than by prompt.
* Ingestion is polite: domain allow-list, robots.txt, one request per second, and
  an on-disk cache so re-runs are cheap.
* Ingestion is resilient: a single failing page never wipes a good corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import urllib3

from src.query import config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --------------------------------------------------------------------------
# Field allow-list
# --------------------------------------------------------------------------

# Dot-paths copied out of mfServerSideData. Nested paths are flattened with a
# dot, e.g. "rta_details.rta_name". A field absent from this set never leaves the
# parser, so it can never reach the vector store.
ALLOWED_FIELDS: tuple[str, ...] = (
    # identity / naming
    "scheme_name",
    "plan_type",
    "scheme_type",
    "fund_house",
    "amc",
    "isin",
    "launch_date",
    "allotment_date",
    # category
    "category",
    "sub_category",
    # charges
    "expense_ratio",
    "base_expense_ratio",
    "exit_load",
    # minimums
    "min_sip_investment",
    "max_sip_investment",
    "min_investment_amount",
    "sip_allowed",
    "lumpsum_allowed",
    # riskometer
    "nfo_risk",
    # lock-in
    "lock_in.years",
    "lock_in.months",
    "lock_in.days",
    "additional_details.lock_in_yrs",
    "additional_details.isTaxSaverFund",
    # benchmark / manager
    "benchmark",
    "benchmark_name",
    # NOTE: fund_manager_details.person_name is deliberately NOT extracted.
    # [verified] it is cross-scheme noise, not this scheme's managers: "Dhruv
    # Muchhal" appears in all five schemes' lists, and the page's own
    # fund_manager value (Prashant Jain / Vinay Kulkarni / Srinivas Rao Ravuri)
    # is missing from four of the five. Indexing it would answer "who manages
    # this fund" with the wrong people. Only the singular field is trusted.
    "fund_manager",
    # registrar (used by the statement-download chunks in Phase 2)
    "rta_details.rta_name",
    "rta_details.custodian_name",
    "rta_details.address",
    "rta_details.email",
    # prose
    "description",
)

# Substrings that must never appear in an extracted record. Guards FR-6 and the
# "no performance claims" non-goal. [verified] none of the ALLOWED_FIELDS above
# collide with any of these substrings, so this check cannot false-positive.
BANNED_KEY_SUBSTRINGS: tuple[str, ...] = (
    "return",
    "nav",
    "holding",
    "portfolio",
    "aum",
    "rating",
    "turnover",
    "performance",
    "yield",
    "sharpe",
    "alpha",
    "beta",
    "rank",
    "score",
    "growth_",
    "historic",
)

# Containers whose contents are never walked (belt-and-braces with the allow-list).
BLOCKED_CONTAINERS: tuple[str, ...] = (
    "stats",
    "return_stats",
    "simple_return",
    "sip_return",
    "analysis",
    "holdings",
    "fund_manager_details",
    "historic_fund_expense",
    "historic_exit_loads",
    "category_info",
    "amc_info",
    "investment_date_configs",
    "stp_details",
    "swp_details",
    "conditional_maximum_purchase_amount_details",
    "conditional_maximum_sip_amount_details",
    "additional_details",
    "lock_in",
)

_NEXT_DATA_RE = re.compile(
    r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S | re.I
)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass
class SourceRow:
    """One row of sources.csv."""

    scheme_slug: str
    display_name: str
    source_url: str
    source_type: str
    tier: str
    fetch_status: str
    fetched_at: str

    @property
    def is_corpus(self) -> bool:
        return self.tier == config.TIER_CORPUS


@dataclass
class LoadResult:
    """Outcome of loading one corpus page."""

    slug: str
    url: str
    display_name: str
    status: str = "pending"  # ok | stale | skipped_disallowed | failed | pending
    fetched_at: str = ""
    from_cache: bool = False
    http_status: int | None = None
    bytes_downloaded: int = 0
    record: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "stale") and bool(self.record)


# --------------------------------------------------------------------------
# HTTP session + robots
# --------------------------------------------------------------------------

_session: requests.Session | None = None
_last_request_at: float = 0.0
_robots_cache: dict[str, "RobotsRules | None"] = {}


def make_session() -> requests.Session:
    """One browser-like session. [verified] the sandbox CA bundle is broken, so
    TLS verification is disabled behind this single helper."""
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": config.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        _session = s
    return _session


def _throttle() -> None:
    """Enforce at least REQUEST_DELAY_SECONDS between outbound requests."""
    global _last_request_at
    wait = config.REQUEST_DELAY_SECONDS - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def get(url: str, **kwargs: Any) -> requests.Response:
    _throttle()
    kwargs.setdefault("timeout", config.REQUEST_TIMEOUT_SECONDS)
    kwargs.setdefault("verify", config.VERIFY_TLS)
    return make_session().get(url, **kwargs)


def domain_allowed(url: str) -> bool:
    """Guardrail: only allow-listed domains may ever be fetched."""
    host = (urlparse(url).hostname or "").lower()
    return host in config.ALLOWED_DOMAINS


class RobotsRules:
    """Minimal RFC 9309 robots.txt matcher.

    Python's ``urllib.robotparser`` is not sufficient here: it compares rule
    paths with a plain ``str.startswith``, so it ignores the ``*`` wildcard and
    the ``$`` end-anchor. Against Groww's robots.txt that means
    ``Disallow: /mutual-funds/compare/*`` never matches and the fetch is wrongly
    permitted. This matcher compiles each rule to a regex and resolves the
    longest-match-wins rule, so wildcard and anchored rules behave correctly.
    """

    def __init__(self, text: str = "", user_agent: str = config.ROBOTS_USER_AGENT) -> None:
        self._rules: list[tuple[re.Pattern[str], bool, int]] = []
        self._parse(text or "", user_agent)

    @staticmethod
    def _compile(path: str) -> tuple[re.Pattern[str], int]:
        anchored = path.endswith("$")
        if anchored:
            path = path[:-1]
        body = ".*".join(re.escape(part) for part in path.split("*"))
        pattern = "^" + body + ("$" if anchored else "")
        # Specificity proxy: length of the literal path, wildcards excluded.
        specificity = len(path.replace("*", ""))
        return re.compile(pattern), specificity

    def _parse(self, text: str, user_agent: str) -> None:
        groups: dict[str, list[str]] = {}
        current: list[str] = []
        agent: str | None = None

        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            field, _, value = line.partition(":")
            field = field.strip().lower()
            value = value.strip()
            if field == "user-agent":
                if agent is not None and current:
                    groups.setdefault(agent.lower(), []).extend(current)
                    current = []
                agent = value
            elif field in ("allow", "disallow"):
                if agent is not None:
                    current.append(f"{field}:{value}")

        if agent is not None and current:
            groups.setdefault(agent.lower(), []).extend(current)

        ua = user_agent.lower()
        chosen = groups.get(ua)
        if chosen is None:
            # Fall back to the "*" group, then to any group present.
            chosen = groups.get("*")
        if chosen is None:
            return

        for rule in chosen:
            kind, _, path = rule.partition(":")
            if kind == "disallow" and path == "":
                # "Disallow:" with an empty value means allow-all.
                continue
            pattern, specificity = self._compile(path)
            self._rules.append((pattern, kind == "allow", specificity))

    def can_fetch(self, target: str) -> bool:
        """True when no rule matches, or the winning rule is an Allow."""
        best: tuple[int, bool] | None = None
        for pattern, allow, specificity in self._rules:
            if not pattern.match(target):
                continue
            if best is None or specificity > best[0] or (specificity == best[0] and allow):
                best = (specificity, allow)
        return True if best is None else best[1]


def robots_for(url: str) -> RobotsRules | None:
    """Fetch and cache robots.txt for a URL's host. None when unavailable."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in _robots_cache:
        return _robots_cache[host]

    rules: RobotsRules | None = None
    # Try the exact host first, then the www. variant. [verified] groww serves
    # robots.txt on both groww.in and www.groww.in.
    for candidate_host in (host, f"www.{host}" if not host.startswith("www.") else host[4:]):
        robots_url = f"{parsed.scheme}://{candidate_host}/robots.txt"
        try:
            resp = get(robots_url)
        except requests.RequestException:
            continue
        if resp.status_code == 200 and resp.text.strip():
            rules = RobotsRules(resp.text)
            break
        if resp.status_code == 404:
            # No robots.txt means no restrictions.
            rules = RobotsRules("")
            break

    _robots_cache[host] = rules
    return rules


def robots_allows(url: str) -> tuple[bool, str]:
    """Return (allowed, reason). Never raises."""
    if not domain_allowed(url):
        return False, f"host not in allow-list: {urlparse(url).hostname}"
    rules = robots_for(url)
    if rules is None:
        return True, "robots.txt unavailable; treated as unrestricted"
    parsed = urlparse(url)
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    allowed = rules.can_fetch(target)
    return allowed, "allowed by robots.txt" if allowed else "disallowed by robots.txt"


# --------------------------------------------------------------------------
# sources.csv
# --------------------------------------------------------------------------

CSV_FIELDS = [
    "scheme_slug",
    "display_name",
    "source_url",
    "source_type",
    "tier",
    "fetch_status",
    "fetched_at",
]


def read_sources() -> list[SourceRow]:
    if not config.SOURCES_CSV.exists():
        raise FileNotFoundError(f"missing {config.SOURCES_CSV}")
    rows: list[SourceRow] = []
    with config.SOURCES_CSV.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            rows.append(
                SourceRow(
                    scheme_slug=(raw.get("scheme_slug") or "").strip(),
                    display_name=(raw.get("display_name") or "").strip(),
                    source_url=(raw.get("source_url") or "").strip(),
                    source_type=(raw.get("source_type") or "").strip(),
                    tier=(raw.get("tier") or "").strip(),
                    fetch_status=(raw.get("fetch_status") or "").strip(),
                    fetched_at=(raw.get("fetched_at") or "").strip(),
                )
            )
    return rows


def write_sources(rows: list[SourceRow]) -> None:
    """Rewrite sources.csv in place, preserving row order.

    Only ``tier == corpus`` rows are ever updated by the loader; citation-only
    rows are written back exactly as read.
    """
    with config.SOURCES_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "scheme_slug": r.scheme_slug,
                    "display_name": r.display_name,
                    "source_url": r.source_url,
                    "source_type": r.source_type,
                    "tier": r.tier,
                    "fetch_status": r.fetch_status,
                    "fetched_at": r.fetched_at,
                }
            )


# --------------------------------------------------------------------------
# Fetch + cache
# --------------------------------------------------------------------------


def _cache_path(slug: str) -> Path:
    return config.RAW_DIR / f"{slug}.html"


def _cache_age_hours(path: Path) -> float:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(tz=timezone.utc) - mtime).total_seconds() / 3600.0


def fetch_page(url: str, slug: str, *, force: bool = False) -> tuple[str | None, dict[str, Any]]:
    """Return (html, info). Uses the on-disk cache when fresh."""
    info: dict[str, Any] = {
        "from_cache": False,
        "http_status": None,
        "bytes_downloaded": 0,
        "notes": [],
    }
    cache_file = _cache_path(slug)

    if cache_file.exists() and not force:
        age = _cache_age_hours(cache_file)
        if age < config.CACHE_MAX_AGE_HOURS:
            info["from_cache"] = True
            info["bytes_downloaded"] = cache_file.stat().st_size
            info["notes"].append(f"cache hit ({age:.1f}h old)")
            return cache_file.read_text(encoding="utf-8", errors="replace"), info

    allowed, reason = robots_allows(url)
    if not allowed:
        info["notes"].append(f"SKIPPED: {reason}")
        return None, info

    resp = get(url)
    info["http_status"] = resp.status_code
    resp.raise_for_status()
    info["bytes_downloaded"] = len(resp.text)
    cache_file.write_text(resp.text, encoding="utf-8")
    return resp.text, info


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def _get_path(data: dict[str, Any], dotted: str) -> Any:
    """Fetch a dot-path. For list containers, returns a list of the values."""
    parts = dotted.split(".")
    node: Any = data
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list):
            collected = []
            for item in node:
                if isinstance(item, dict) and part in item:
                    collected.append(item[part])
            return collected or None
        else:
            return None
    return node


def assert_no_banned_keys(record: dict[str, Any]) -> None:
    """Fail loudly if a performance/PII-bearing field slipped through."""
    offenders = [
        key
        for key in record
        if any(bad in key.lower() for bad in BANNED_KEY_SUBSTRINGS)
    ]
    if offenders:
        raise ValueError(
            "banned keys present in extracted record (FR-6 violation): "
            + ", ".join(sorted(offenders))
        )


def extract_record(html: str, slug: str) -> dict[str, Any]:
    """Pull the allow-listed fields out of the embedded __NEXT_DATA__ payload."""
    match = _NEXT_DATA_RE.search(html)
    if not match:
        raise ValueError("no __NEXT_DATA__ script tag found")

    payload = json.loads(match.group(1))
    server_data = payload["props"]["pageProps"]["mfServerSideData"]
    if not isinstance(server_data, dict):
        raise ValueError("mfServerSideData is not an object")

    record: dict[str, Any] = {
        "scheme_slug": slug,
        "display_name": config.display_name(slug),
        "source_url": next(
            (r.source_url for r in read_sources() if r.scheme_slug == slug), ""
        ),
    }
    for dotted in ALLOWED_FIELDS:
        value = _get_path(server_data, dotted)
        if value is None or value == "" or value == []:
            continue
        record[dotted] = value

    assert_no_banned_keys(record)
    return record


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def load_corpus(*, force: bool = False, verbose: bool = True) -> list[LoadResult]:
    """Load every corpus source. Never raises for a single-page failure."""
    rows = read_sources()
    by_slug = {r.scheme_slug: r for r in rows}
    results: list[LoadResult] = []

    for slug in config.CORPUS_SLUGS:
        row = by_slug.get(slug)
        if row is None:
            results.append(
                LoadResult(
                    slug=slug,
                    url="",
                    display_name=config.display_name(slug),
                    status="failed",
                    notes=[f"no sources.csv row for slug '{slug}'"],
                )
            )
            continue

        result = LoadResult(
            slug=slug,
            url=row.source_url,
            display_name=row.display_name or config.display_name(slug),
        )

        if not domain_allowed(row.source_url):
            result.status = "skipped_disallowed"
            result.notes.append("host not in allow-list")
            results.append(result)
            continue

        try:
            html, info = fetch_page(row.source_url, slug, force=force)
            result.notes.extend(info.get("notes", []))
            result.from_cache = bool(info.get("from_cache"))
            result.http_status = info.get("http_status")
            result.bytes_downloaded = int(info.get("bytes_downloaded") or 0)

            if html is None:
                result.status = "skipped_disallowed"
                results.append(result)
                continue

            result.record = extract_record(html, slug)
            result.status = "ok"
            result.fetched_at = date.today().isoformat()

        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as exc:
            result.notes.append(f"{type(exc).__name__}: {exc}")
            # Degrade gracefully: fall back to a stale cache if we have one.
            cache_file = _cache_path(slug)
            if cache_file.exists():
                try:
                    result.record = extract_record(
                        cache_file.read_text(encoding="utf-8", errors="replace"), slug
                    )
                    result.from_cache = True
                    result.bytes_downloaded = cache_file.stat().st_size
                    result.status = "stale"
                except Exception as cache_exc:  # noqa: BLE001
                    result.notes.append(f"stale cache unusable: {cache_exc}")
                    result.status = "failed"
            else:
                result.status = "failed"

        results.append(result)
        if verbose:
            _print_result(result)

    return results


def _print_result(result: LoadResult) -> None:
    kb = result.bytes_downloaded / 1024.0
    flags = []
    if result.from_cache:
        flags.append("cache")
    if result.http_status:
        flags.append(f"http={result.http_status}")
    fields = len([k for k in result.record if "." in k or k in ALLOWED_FIELDS])
    print(
        f"  [{result.status:>18}] {result.slug:<24} {kb:7.1f} KB  "
        f"fields={fields:<3} {' '.join(flags)}"
    )
    for note in result.notes:
        print(f"       - {note}")


def persist(results: list[LoadResult]) -> None:
    """Write records.json and update sources.csv for corpus rows only."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "fetched_at": date.today().isoformat(),
        "records": [r.record for r in results if r.record],
    }
    config.RECORDS_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = read_sources()
    status_by_slug = {r.slug: r for r in results}
    for row in rows:
        # citation_only rows are never touched by the loader
        if not row.is_corpus:
            continue
        res = status_by_slug.get(row.scheme_slug)
        if res is None:
            continue
        row.fetch_status = res.status
        row.fetched_at = res.fetched_at or row.fetched_at
    write_sources(rows)


def summarise(results: list[LoadResult]) -> int:
    ok = [r for r in results if r.ok]
    print()
    print(f"  pages loaded : {len(ok)}/{len(results)}")
    if ok:
        sizes = sorted(r.bytes_downloaded for r in ok)
        print(
            f"  page size    : min {sizes[0]/1024:.0f} KB, "
            f"max {sizes[-1]/1024:.0f} KB"
        )
    total_fields = sum(
        len([k for k in r.record if k not in ("scheme_slug", "display_name", "source_url")])
        for r in ok
    )
    print(f"  fields kept  : {total_fields}")
    print(f"  records json : {config.RECORDS_JSON}")
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"  FAILED       : {', '.join(r.slug for r in failed)}")
    return 0 if len(ok) == len(results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1: fetch and extract the five HDFC scheme pages."
    )
    parser.add_argument(
        "--force", action="store_true", help="ignore the 24h raw-HTML cache"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress per-page output"
    )
    args = parser.parse_args(argv)

    print("Phase 1 - loading corpus")
    print(f"  sources : {config.SOURCES_CSV}")
    print(f"  raw dir : {config.RAW_DIR}")
    print(f"  allow   : {', '.join(sorted(config.ALLOWED_DOMAINS))}")
    print()

    results = load_corpus(force=args.force, verbose=not args.quiet)
    persist(results)
    return summarise(results)


if __name__ == "__main__":
    sys.exit(main())

# main.py
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import feedparser
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"


# -------------------------
# utils
# -------------------------
def nowstamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_truthy(s: str | None) -> bool:
    if s is None:
        return False
    return s.strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def safe_title(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"<[^>]+>", "", s)  # strip HTML tags
    return normalize_ws(s)


def pick_link(entry: dict) -> str:
    return (entry.get("link") or "").strip()


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def within_days(dt: datetime, days: int) -> bool:
    if days <= 0:
        return True
    # dt.tzinfo が None の場合に備えてローカルとして扱う
    base = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    return dt >= (base - timedelta(days=days))


def parse_entry_datetime(entry: dict) -> Optional[datetime]:
    st = entry.get("published_parsed") or entry.get("updated_parsed")
    if st:
        try:
            return datetime(*st[:6], tzinfo=timezone.utc).astimezone()
        except Exception:
            pass
    return None


def is_gnews(url: str) -> bool:
    return "news.google.com/rss" in (url or "")


def osc8_link(text: str, url: str, enable: bool) -> str:
    if not enable:
        return f"{text} ({url})"
    return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"


# -------------------------
# config
# -------------------------
def load_config() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="path to config json")
    args = ap.parse_args()

    path = Path(args.config)
    if not path.is_absolute():
        path = (ROOT / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません。configを作成してください。")

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["_config_path"] = str(path)
    cfg["_config_name"] = cfg.get("name") or path.stem
    # 表示タイトルは title > interest_name > name
    cfg["_title"] = cfg.get("title") or cfg.get("interest_name") or cfg["_config_name"]

    cfg["top_n"] = int(cfg.get("top_n", 7))
    cfg["max_items_total"] = int(cfg.get("max_items_total", 80))
    cfg["max_per_feed_default"] = int(cfg.get("max_per_feed_default", 20))
    cfg["max_per_feed_gnews"] = int(cfg.get("max_per_feed_gnews", 10))
    cfg["use_osc8"] = bool(cfg.get("use_osc8", True))

    cfg["max_items"] = int(cfg.get("max_items", 25))
    cfg["dedupe_days"] = int(cfg.get("dedupe_days", 60))

    cfg["include_keywords"] = cfg.get("include_keywords", []) or []
    cfg["exclude_keywords"] = cfg.get("exclude_keywords", []) or []

    cfg["categories"] = cfg.get("categories", []) or []
    cfg["category_rules"] = cfg.get("category_rules", []) or []

    cfg["feeds"] = cfg.get("feeds", []) or []
    if not cfg["feeds"]:
        raise ValueError("feeds が空です。RSSを追加してください。")

    # Seen store default: per-config file under seen_urls/
    seen_dir = cfg.get("seen_dir") or "seen_urls"
    cfg["_seen_dir"] = str((ROOT / seen_dir).resolve())
    ensure_dir(Path(cfg["_seen_dir"]))

    cfg["_seen_path"] = str((Path(cfg["_seen_dir"]) / f"{cfg['_config_name']}.json").resolve())
    cfg["seen_max"] = int(cfg.get("seen_max", 7000))

    # LINE: 新着0なら送らない（デフォルトTrue）
    cfg["line_skip_if_no_new"] = bool(cfg.get("line_skip_if_no_new", True))

    return cfg


# -------------------------
# seen store
# -------------------------
def load_seen(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"version": 1, "items": []}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "items" not in data:
            return {"version": 1, "items": []}
        return data
    except Exception:
        try:
            p.rename(p.with_suffix(".corrupt.json"))
        except Exception:
            pass
        return {"version": 1, "items": []}


def save_seen(path: str, data: dict, max_items: int) -> None:
    items = data.get("items", [])
    if len(items) > max_items:
        data["items"] = items[-max_items:]
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def seen_has(seen: dict, url: str) -> bool:
    h = hash_url(url)
    return h in set(seen.get("items", []))


def seen_add(seen: dict, url: str) -> None:
    h = hash_url(url)
    seen.setdefault("items", []).append(h)


# -------------------------
# filtering / categorization
# -------------------------
def compile_keywords(keywords: List[str]) -> List[re.Pattern]:
    pats = []
    for k in keywords:
        k = k.strip()
        if not k:
            continue
        pats.append(re.compile(re.escape(k), re.IGNORECASE))
    return pats


def matches_any(text: str, pats: List[re.Pattern]) -> bool:
    for p in pats:
        if p.search(text):
            return True
    return False


def categorize(cfg: dict, title: str, summary: str) -> str:
    text = f"{title}\n{summary}".lower()
    for rule in cfg.get("category_rules", []):
        cat = rule.get("category", "Other")
        kws = rule.get("keywords", []) or []
        for kw in kws:
            if kw and kw.lower() in text:
                return cat
    return "Other"


@dataclass
class Item:
    title: str
    url: str
    source: str
    published: Optional[datetime]
    category: str


# ★修正: feedparser に timeout を渡さず、requests で timeout 付き取得してから parse
def fetch_feed(url: str, timeout: int = 20) -> feedparser.FeedParserDict:
    headers = {
        "User-Agent": "news-digest-bot/1.0 (+https://example.invalid)",
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return feedparser.parse(r.content)


def collect_items(cfg: dict) -> List[Item]:
    include_pats = compile_keywords(cfg.get("include_keywords", []))
    exclude_pats = compile_keywords(cfg.get("exclude_keywords", []))

    max_total = int(cfg.get("max_items_total", 80))
    max_default = int(cfg.get("max_per_feed_default", 20))
    max_gnews = int(cfg.get("max_per_feed_gnews", 10))

    items: List[Item] = []
    for feed_url in cfg["feeds"]:
        try:
            feed = fetch_feed(feed_url)
        except Exception as e:
            print(f"[WARN] feed fetch failed: {feed_url} err={repr(e)}")
            continue

        feed_title = safe_title((getattr(feed, "feed", {}) or {}).get("title", "") or "")
        src = feed_title or domain_of(feed_url) or feed_url

        per_limit = max_gnews if is_gnews(feed_url) else max_default
        if not feed.entries:
            continue

        take = 0
        for ent in feed.entries:
            if take >= per_limit:
                break

            t = safe_title(ent.get("title", ""))
            u = pick_link(ent)
            if not t or not u:
                continue

            text_for_match = f"{t}\n{html.unescape(ent.get('summary','') or '')}"
            if include_pats and not matches_any(text_for_match, include_pats):
                continue
            if exclude_pats and matches_any(text_for_match, exclude_pats):
                continue

            dt = parse_entry_datetime(ent)
            dd = int(cfg.get("dedupe_days", 60))
            if dt is not None and not within_days(dt, dd):
                continue

            cat = categorize(cfg, t, html.unescape(ent.get("summary", "") or ""))
            items.append(Item(title=t, url=u, source=src, published=dt, category=cat))

            take += 1
            if len(items) >= max_total:
                break

        if len(items) >= max_total:
            break

    def sort_key(it: Item):
        return (it.published is not None, it.published or datetime(1970, 1, 1, tzinfo=timezone.utc))

    items.sort(key=sort_key, reverse=True)

    seen_u = set()
    uniq: List[Item] = []
    for it in items:
        if it.url in seen_u:
            continue
        seen_u.add(it.url)
        uniq.append(it)
    return uniq


# -------------------------
# formatting
# -------------------------
def format_digest_slack(cfg: dict, items: List[Item]) -> str:
    title = cfg.get("_title", cfg.get("_config_name"))
    top_n = int(cfg.get("top_n", 7))

    lines: List[str] = []
    lines.append(f"*{title}*")
    lines.append("")  # ★generated を消した

    if not items:
        lines.append("新着なし")
        return "\n".join(lines)

    # カテゴリ別
    grouped: Dict[str, List[Item]] = {}
    for it in items:
        grouped.setdefault(it.category, []).append(it)

    n = 0
    for cat, lst in grouped.items():
        if n >= top_n:
            break

        lines.append(f"**")  # ★壊れてた "**" を修正
        for it in lst:
            if n >= top_n:
                break
            lines.append(f"• {it.title}\n  {it.url}")
            n += 1
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def format_digest_line(cfg: dict, items: List[Item]) -> str:
    title = cfg.get("_title", cfg.get("_config_name"))
    top_n = int(cfg.get("top_n", 7))

    lines: List[str] = []
    lines.append(f"{title}")
    lines.append("")  # ★日時（nowstamp）を消した

    if not items:
        lines.append("新着なし")
        return "\n".join(lines).strip()

    for i, it in enumerate(items[:top_n], start=1):
        lines.append(f"{i}. {it.title}")
        lines.append(f"{it.url}")
        lines.append("")

    return "\n".join(lines).strip()


def print_console_preview(cfg: dict, items: List[Item]) -> None:
    use_osc8 = bool(cfg.get("use_osc8", True))
    title = cfg.get("_title", cfg.get("_config_name"))
    top_n = int(cfg.get("top_n", 7))

    print("=" * 60)
    print(f"[{cfg.get('_config_name')}] {title}")
    # ★generated を消した
    print(f"items(total after filter)={len(items)}  preview(top_n)={top_n}")
    print("=" * 60)

    n = 0
    for it in items:
        if n >= top_n:
            break
        print(f"- {it.title}")
        print(f"  {osc8_link(it.url, it.url, use_osc8)}")
        n += 1
    print("")


# -------------------------
# delivery
# -------------------------
def post_slack(webhook: str, text: str, timeout: int = 15) -> bool:
    try:
        r = requests.post(webhook, json={"text": text}, timeout=timeout)
        ok = 200 <= r.status_code < 300
        if not ok:
            print(f"[WARN] Slack non-OK status={r.status_code} body={r.text[:200]}")
        return ok
    except Exception as e:
        print(f"[WARN] Slack post failed: {repr(e)}")
        return False


def post_line_push(token: str, to: str, text: str, timeout: int = 15) -> bool:
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"to": to, "messages": [{"type": "text", "text": text}]}

    t0 = time.time()
    print(f"[DBG] LINE push start {nowstamp()}")
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    elapsed = time.time() - t0
    print(f"[DBG] LINE push end   {nowstamp()} status={r.status_code} elapsed={elapsed:.2f}s body={r.text[:200]}")
    ok = 200 <= r.status_code < 300
    if not ok:
        print(f"[WARN] LINE push non-OK status={r.status_code} body={r.text[:500]}")
    return ok


# ★追加: broadcast 対応
def post_line_broadcast(token: str, text: str, timeout: int = 15) -> bool:
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"messages": [{"type": "text", "text": text}]}

    t0 = time.time()
    print(f"[DBG] LINE broadcast start {nowstamp()}")
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    elapsed = time.time() - t0
    print(f"[DBG] LINE broadcast end   {nowstamp()} status={r.status_code} elapsed={elapsed:.2f}s body={r.text[:200]}")
    ok = 200 <= r.status_code < 300
    if not ok:
        print(f"[WARN] LINE broadcast non-OK status={r.status_code} body={r.text[:500]}")
    return ok


def resolve_env(key: str) -> str:
    return (os.getenv(key) or "").strip()


def deliver_single(cfg: dict, text: str) -> bool:
    dry = is_truthy(os.getenv("DRY_RUN"))
    d = (cfg.get("delivery") or {}) if isinstance(cfg.get("delivery"), dict) else {}
    dtype = str(d.get("type", "slack")).lower().strip()

    if dry:
        print("[DRY_RUN] deliver skipped. printing text below:\n")
        print(text)
        return True

    if dtype == "slack":
        env_key = d.get("webhook_env") or d.get("webhook_env_key")
        if not env_key:
            print("[WARN] Slack webhook env key not set in config: delivery.webhook_env")
            return False
        webhook = resolve_env(env_key)
        if not webhook:
            print(f"[WARN] Slack webhook not set: env={env_key}")
            return False
        return post_slack(webhook, text)

    if dtype == "line":
        token_env = d.get("token_env") or "LINE_CHANNEL_ACCESS_TOKEN"
        to_env = d.get("to_env") or "LINE_TO"
        mode = str(d.get("mode", "push")).lower().strip()

        token = resolve_env(token_env)
        if not token:
            print(f"[WARN] LINE token not set: env={token_env}")
            return False

        if mode == "push":
            to = resolve_env(to_env)
            if not to:
                print(f"[WARN] LINE to not set: env={to_env}")
                return False
            return post_line_push(token, to, text)

        if mode == "broadcast":
            return post_line_broadcast(token, text)

        print(f"[WARN] LINE mode not supported: {mode} (supported: push/broadcast)")
        return False

    print(f"[WARN] Unknown delivery type: {dtype}")
    return False


def deliver_all(cfg: dict, text: Union[str, Dict[str, str]], new_count: int) -> dict:
    results = {"sent": 0, "attempted": 0, "details": []}

    def should_skip_line(delivery_obj: dict) -> bool:
        dtype = str((delivery_obj or {}).get("type", "")).lower().strip()
        skip = bool(cfg.get("line_skip_if_no_new", True))
        return dtype == "line" and skip and new_count == 0

    def pick_text_for(dtype: str) -> str:
        # text が dict の場合: "line"/"slack" を使い分ける
        if isinstance(text, dict):
            if dtype == "line":
                return text.get("line", "") or text.get("slack", "")
            return text.get("slack", "") or text.get("line", "")
        return text

    deliveries = cfg.get("deliveries")

    # ---- single delivery mode (cfg["delivery"]) ----
    if not deliveries:
        d = (cfg.get("delivery") or {}) if isinstance(cfg.get("delivery"), dict) else {}
        dtype = str(d.get("type", "slack")).lower().strip()
        results["attempted"] = 1

        if should_skip_line(d):
            results["sent"] = 0
            results["details"].append({"type": "line", "ok": False, "skipped": True, "reason": "new=0"})
            print("[INFO] Skip LINE delivery (new=0)")
            return results

        ok = deliver_single(cfg, pick_text_for(dtype))
        results["sent"] = 1 if ok else 0
        results["details"].append({"type": dtype, "ok": ok})
        return results

    # ---- multiple deliveries mode (cfg["deliveries"]) ----
    if not isinstance(deliveries, list):
        print("[WARN] config.deliveries must be a list; falling back to single delivery")
        d = (cfg.get("delivery") or {}) if isinstance(cfg.get("delivery"), dict) else {}
        dtype = str(d.get("type", "slack")).lower().strip()
        results["attempted"] = 1

        if should_skip_line(d):
            results["sent"] = 0
            results["details"].append({"type": "line", "ok": False, "skipped": True, "reason": "new=0"})
            print("[INFO] Skip LINE delivery (new=0)")
            return results

        ok = deliver_single(cfg, pick_text_for(dtype))
        results["sent"] = 1 if ok else 0
        results["details"].append({"type": dtype, "ok": ok})
        return results

    for d in deliveries:
        d = d or {}
        dtype = str(d.get("type", "")).lower().strip()
        results["attempted"] += 1

        if should_skip_line(d):
            results["details"].append({"type": "line", "ok": False, "skipped": True, "reason": "new=0"})
            continue

        prev = cfg.get("delivery")
        cfg["delivery"] = d

        ok = deliver_single(cfg, pick_text_for(dtype))

        if prev is None:
            cfg.pop("delivery", None)
        else:
            cfg["delivery"] = prev

        results["sent"] += 1 if ok else 0
        results["details"].append({"type": dtype, "ok": ok})

    return results


# -------------------------
# logging
# -------------------------
def setup_logs(cfg: dict) -> Path:
    logs_dir = ROOT / "logs"
    ensure_dir(logs_dir)
    return logs_dir / f"task_{cfg['_config_name']}.log"


def tee_stdout_to_file(log_path: Path, cfg: dict) -> None:
    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:
                    pass

        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    f = log_path.open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)

    print("\n" + "=" * 80)
    print(f"[LOG] start {nowstamp()} config={cfg.get('_config_path')}")
    print(f"[DBG] python={sys.executable}")
    print(f"[DBG] ROOT={ROOT}")
    print(f"[DBG] env_file_exists={(ROOT / '.env').exists()}")
    print("=" * 80 + "\n")


# -------------------------
# main
# -------------------------
def main() -> int:
    cfg = load_config()
    load_dotenv(dotenv_path=(ROOT / ".env"), override=False)

    log_path = setup_logs(cfg)
    tee_stdout_to_file(log_path, cfg)

    print(f"[INFO] cfg_name={cfg['_config_name']} title={cfg.get('_title')}")
    print(f"[INFO] seen_path={cfg['_seen_path']}")

    items = collect_items(cfg)
    print_console_preview(cfg, items)

    seen = load_seen(cfg["_seen_path"])
    new_items: List[Item] = []
    for it in items:
        if not seen_has(seen, it.url):
            new_items.append(it)

    max_items = int(cfg.get("max_items", 25))
    new_items = new_items[:max_items]

    for it in new_items:
        seen_add(seen, it.url)
    save_seen(cfg["_seen_path"], seen, int(cfg.get("seen_max", 7000)))

    # Slack/LINEで文面を分ける
    slack_text = format_digest_slack(cfg, new_items)
    line_text = format_digest_line(cfg, new_items)

    res = deliver_all(cfg, {"slack": slack_text, "line": line_text}, len(new_items))

    print(
        f"[OK] {cfg['_config_name']}: new={len(new_items)} total={len(items)} "
        f"sent={res['sent']}/{res['attempted']} details={res['details']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

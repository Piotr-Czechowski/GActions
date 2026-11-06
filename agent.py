#!/usr/bin/env python3
"""
A3C Watcher - Twój pierwszy agent.

Agent działa w prostym cyklu:
  1. OBSERWUJ    - pobierz listę eksperymentów (runów) z projektu W&B
  2. PORÓWNAJ    - znajdź runy, których jeszcze nie widział (plik state.json)
  3. ZROZUM      - wyciągnij kluczowe cechy: liczba workerów, średnia nagroda,
                   czas trwania, kto uruchomil. Opcjonalnie poproś Claude'a
                   o interpretację, gdy nazwy pól są nietypowe.
  4. DZIAŁAJ     - wyślij maila z podsumowaniem i linkami
  5. ZAPAMIĘTAJ  - zapisz ID runów jako widziane

Użycie:
  python agent.py                  # normalne sprawdzenie (to uruchamia harmonogram)
  python agent.py --check          # test połączenia z W&B + podglad pól ostatnich runów
  python agent.py --test-email     # wyślij testowego maila
  python agent.py --dry-run        # pokaż maila w konsoli, nic nie wysyłaj i nie zapisuj
  python agent.py --notify-existing  # przy pierwszym uruchomieniu zgłoś też stare runy
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import logging
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

STATE_FILE = HERE / "state.json"
LOG_FILE = HERE / "agent.log"

# --- Konfiguracja (z pliku .env) ---------------------------------------------
ENTITY = (os.getenv("WANDB_ENTITY") or "rl_testing")
PROJECT = (os.getenv("WANDB_PROJECT") or "A3C_ACCELERATION")
UI_BASE = (os.getenv("UI_BASE_URL") or "https://forge.coreweave.com/wandb").rstrip("/")
EMAIL_TO = (os.getenv("EMAIL_TO") or "phdpcz1997@gmail.com")
GMAIL_USER = (os.getenv("GMAIL_USER") or "")
GMAIL_APP_PASSWORD = (os.getenv("GMAIL_APP_PASSWORD") or "").replace(" ", "")
ANTHROPIC_API_KEY = (os.getenv("ANTHROPIC_API_KEY") or "")
CLAUDE_MODEL = (os.getenv("CLAUDE_MODEL") or "claude-haiku-5-5")
MAX_RUNS = int((os.getenv("MAX_RUNS") or "200"))


def _keys_from_env(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [k.strip() for k in value.split(",") if k.strip()]


# Nazwy pól, pod którymi agent szuka liczby workerów (w config runa)
WORKER_KEYS = _keys_from_env("WORKERS_KEYS", [
    "num_workers", "n_workers", "workers", "nworkers", "num_processes",
    "n_processes", "num_procs", "n_procs", "num_actors", "n_actors",
    "num_threads", "world_size", "num_envs", "n_envs",
])
# Nazwy pól, pod którymi agent szuka średniej nagrody (w summary, potem w config)
REWARD_KEYS = _keys_from_env("REWARD_KEYS", [
    "mean_reward", "avg_reward", "average_reward", "reward_mean",
    "episode_reward_mean", "mean_episode_reward", "ep_rew_mean",
    "rollout/ep_rew_mean", "eval/mean_reward", "reward/mean",
    "episode_reward", "reward",
])

log = logging.getLogger("a3c_watcher")


def setup_logging() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(LOG_FILE, encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)


# --- 1. OBSERWUJ ----------------------------------------------------------------
def fetch_runs() -> list:
    """Pobiera najnowsze runy z projektu (od najnowszego)."""
    import wandb  # import tutaj, żeby --help działał bez wandb

    api = wandb.Api(timeout=60)  # klucz z WANDB_API_KEY, host z WANDB_BASE_URL
    runs = api.runs(f"{ENTITY}/{PROJECT}", order="-created_at", per_page=50)
    result = []
    for i, run in enumerate(runs):
        if i >= MAX_RUNS:
            break
        result.append(run)
    return result


# --- 2. PORÓWNAJ ---------------------------------------------------------------
def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def save_state(seen: set[str]) -> None:
    STATE_FILE.write_text(json.dumps({
        "seen_run_ids": sorted(seen),
        "last_check": dt.datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")


# --- 3. ZROZUM -----------------------------------------------------------------
def flatten(d, prefix: str = "") -> dict:
    """Spłaszcza zagnieżdżone słowniki: {"a": {"b": 1}} -> {"a.b": 1}."""
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        k = str(k)
        if k.startswith("_wandb"):
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            if "_type" in v:  # histogramy, tabele, obrazki - pomijamy
                continue
            if "value" in v and set(v) <= {"value", "desc"}:
                out[key] = v["value"]
            else:
                out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _norm(key: str) -> str:
    return key.lower().replace("/", ".").replace("-", "_").replace(" ", "_")


def find_value(flat: dict, candidates: list[str]):
    """Zwraca (nazwa_pola, wartosc) dla pierwszego pasującego pola albo (None, None)."""
    normed = {_norm(k): (k, v) for k, v in flat.items()}
    for cand in candidates:                       # dokładne dopasowanie
        if _norm(cand) in normed:
            return normed[_norm(cand)]
    for cand in candidates:                       # dopasowanie końcówki, np. "train.mean_reward"
        nc = "." + _norm(cand)
        for nk, kv in normed.items():
            if nk.endswith(nc):
                return kv
    return None, None


def get_summary(run) -> dict:
    try:
        return dict(run.summary._json_dict)
    except Exception:
        try:
            return {k: run.summary[k] for k in run.summary.keys()}
        except Exception:
            return {}


def _parse_ts(value):
    if not value:
        return None
    try:
        ts = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def duration_seconds(run, summary: dict):
    runtime = summary.get("_runtime")
    if isinstance(runtime, (int, float)):
        return float(runtime)
    start = _parse_ts(getattr(run, "created_at", None))
    end = _parse_ts(getattr(run, "heartbeat_at", None))
    if start and end:
        return max(0.0, (end - start).total_seconds())
    return None


def format_duration(seconds) -> str:
    if seconds is None:
        return "brak danych"
    seconds = int(seconds)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days} d {hours} h {minutes} min"
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def run_author(run) -> str:
    try:
        user = run.user
        name = getattr(user, "name", None)
        username = getattr(user, "username", None)
        if name and username and name != username:
            return f"{name} ({username})"
        if name or username:
            return name or username
    except Exception:
        pass
    return "nieznany"


def format_number(value) -> str:
    if value is None:
        return "brak danych"
    if isinstance(value, float):
        return f"{value:,.3f}".replace(",", " ")
    return str(value)


def run_link(run) -> str:
    return f"{UI_BASE}/{ENTITY}/{PROJECT}/runs/{run.id}"


def ask_claude(name: str, config: dict, summary: dict) -> dict | None:
    """'Mózg' agenta: Claude interpretuje nietypowe nazwy pol i wskazuje ciekawe cechy.

    Działa tylko, gdy w .env jest ANTHROPIC_API_KEY. Bez niego agent korzysta
    wyłącznie z reguł (WORKER_KEYS / REWARD_KEYS).
    """
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("Brak pakietu 'anthropic' - pomijam analizę przez Claude'a.")
        return None

    def trim(d: dict) -> dict:
        return {k: str(v)[:200] for k, v in list(d.items())[:150]}

    prompt = (
        "Analizujesz eksperyment uczenia ze wzmocnieniem (implementacje A2C/A3C "
        "uruchamiane na zasobach HPC), zapisany w Weights & Biases.\n\n"
        f"Nazwa runa: {name}\n"
        f"CONFIG:\n{json.dumps(trim(config), ensure_ascii=False, indent=1)}\n\n"
        f"SUMMARY (ostatnie wartosci metryk):\n{json.dumps(trim(summary), ensure_ascii=False, indent=1)}\n\n"
        "Zwroc WYLACZNIE obiekt JSON o kluczach:\n"
        '  "workers": liczba workerów/procesow/aktorow (liczba) lub null,\n'
        '  "workers_key": nazwa pola, z ktorego ja wziales, lub null,\n'
        '  "mean_reward": średnia nagroda (liczba) lub null,\n'
        '  "mean_reward_key": nazwa pola, z ktorego ja wziales, lub null,\n'
        '  "highlights": lista max 4 krotkich zdan po polsku o innych istotnych '
        "cechach (np. algorytm A2C/A3C, srodowisko, liczba wezlow/GPU, learning rate, "
        "liczba krokow, przyspieszenie).\n"
        "Korzystaj tylko z podanych danych - nie zgaduj. Brak danych = null."
    )
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(b, "text", "") for b in message.content)
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1])
    except Exception as exc:
        log.warning("Analiza przez Claude'a nie powiodła się: %s", exc)
        return None


def describe_run(run) -> dict:
    """Zbiera najważniejsze cechy jednego eksperymentu."""
    config = flatten(dict(run.config or {}))
    summary = flatten(get_summary(run))

    workers_key, workers = find_value(config, WORKER_KEYS)
    reward_key, reward = find_value(summary, REWARD_KEYS)
    if reward_key is None:
        reward_key, reward = find_value(config, REWARD_KEYS)

    info = {
        "id": run.id,
        "name": getattr(run, "name", run.id),
        "state": getattr(run, "state", "?"),
        "author": run_author(run),
        "created_at": str(getattr(run, "created_at", "")),
        "workers": workers,
        "workers_key": workers_key,
        "mean_reward": reward,
        "mean_reward_key": reward_key,
        "duration": format_duration(duration_seconds(run, get_summary(run))),
        "url": run_link(run),
        "highlights": [],
    }

    ai = ask_claude(info["name"], config, summary)
    if ai:
        if info["workers"] is None and ai.get("workers") is not None:
            info["workers"], info["workers_key"] = ai["workers"], f"{ai.get('workers_key')} (Claude)"
        if info["mean_reward"] is None and ai.get("mean_reward") is not None:
            info["mean_reward"], info["mean_reward_key"] = ai["mean_reward"], f"{ai.get('mean_reward_key')} (Claude)"
        info["highlights"] = [str(h) for h in (ai.get("highlights") or [])][:4]
    return info


# --- 4. DZIAŁAJ ----------------------------------------------------------------
STATE_PL = {"running": "w toku", "finished": "zakończony", "crashed": "awaria",
            "failed": "błąd", "killed": "przerwany", "preempted": "wywłaszczony"}


def build_email(runs_info: list[dict]) -> EmailMessage:
    n = len(runs_info)
    noun = "nowy eksperyment" if n == 1 else "nowe eksperymenty" if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else "nowych eksperymentów"
    subject = f"[A3C Watcher] {n} {noun} w {PROJECT}"

    text_parts, html_parts = [], []
    for r in runs_info:
        state = STATE_PL.get(r["state"], r["state"])
        rows = [
            ("Kto uruchomił", r["author"]),
            ("Liczba workerów", format_number(r["workers"])),
            ("Średnia nagroda", format_number(r["mean_reward"])),
            ("Czas trwania", f"{r['duration']} ({state})"),
            ("Utworzono (UTC)", r["created_at"].replace("T", " ")),
        ]
        text_parts.append(
            f"{r['name']}\n{r['url']}\n"
            + "\n".join(f"  {k}: {v}" for k, v in rows)
            + "".join(f"\n  - {h}" for h in r["highlights"])
        )
        table = "".join(
            f"<tr><td style='padding:3px 12px 3px 0;color:#666'>{html.escape(k)}</td>"
            f"<td style='padding:3px 0'><b>{html.escape(str(v))}</b></td></tr>"
            for k, v in rows
        )
        hl = ""
        if r["highlights"]:
            hl = "<ul style='margin:6px 0 0 18px;padding:0'>" + "".join(
                f"<li>{html.escape(h)}</li>" for h in r["highlights"]) + "</ul>"
        html_parts.append(
            "<div style='border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:12px 0'>"
            f"<div style='font-size:16px'><a href='{html.escape(r['url'])}'>{html.escape(r['name'])}</a></div>"
            f"<table style='margin-top:8px;font-size:14px'>{table}</table>{hl}</div>"
        )

    project_url = f"{UI_BASE}/{ENTITY}/{PROJECT}"
    heading = f"{n} {noun} w {PROJECT}"
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER or EMAIL_TO
    msg["To"] = EMAIL_TO
    msg.set_content("\n\n".join(text_parts) + f"\n\nProjekt: {project_url}\n")
    msg.add_alternative(
        "<div style='font-family:Arial,sans-serif;max-width:640px'>"
        f"<h2 style='font-size:18px'>{html.escape(heading)}</h2>"
        + "".join(html_parts)
        + f"<p style='color:#888;font-size:12px'>Projekt: <a href='{project_url}'>{project_url}</a></p></div>",
        subtype="html",
    )
    return msg


def send_email(msg: EmailMessage) -> None:
    if not (GMAIL_USER and GMAIL_APP_PASSWORD):
        raise RuntimeError("Uzupełnij GMAIL_USER i GMAIL_APP_PASSWORD w pliku .env")
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        smtp.send_message(msg)


def send_error_email(error: Exception) -> None:
    msg = EmailMessage()
    msg["Subject"] = "[A3C Watcher] Błąd podczas sprawdzania eksperymentów"
    msg["From"] = GMAIL_USER or EMAIL_TO
    msg["To"] = EMAIL_TO
    msg.set_content(f"Agent nie mógł sprawdzić projektu {ENTITY}/{PROJECT}.\n\n"
                    f"Błąd: {error!r}\n\nSzczegóły w pliku agent.log.")
    try:
        send_email(msg)
    except Exception as exc:
        log.error("Nie udało się wysłać maila o błędzie: %s", exc)


# --- Tryby uruchomienia ---------------------------------------------------------
def cmd_check() -> None:
    log.info("Łączę się z W&B: %s/%s", ENTITY, PROJECT)
    runs = fetch_runs()
    log.info("OK - znaleziono %d runów.", len(runs))
    for run in runs[:5]:
        info = describe_run(run)
        print(f"\n=== {info['name']}  [{info['id']}]  {info['state']}")
        print(f"  autor:     {info['author']}")
        print(f"  workery:   {info['workers']}   (pole: {info['workers_key']})")
        print(f"  nagroda:   {info['mean_reward']}   (pole: {info['mean_reward_key']})")
        print(f"  czas:      {info['duration']}")
        print(f"  link:      {info['url']}")
        for h in info["highlights"]:
            print(f"  - {h}")
    if runs:
        latest = runs[0]
        print("\nPola CONFIG najnowszego runa:\n  " + ", ".join(sorted(flatten(dict(latest.config or {})))))
        print("\nPola SUMMARY najnowszego runa:\n  " + ", ".join(sorted(flatten(get_summary(latest)))))
        print("\nJeśli workery/nagroda = None, dopisz właściwą nazwę pola do WORKERS_KEYS / REWARD_KEYS w .env")


def cmd_test_email() -> None:
    sample = [{
        "id": "test123", "name": "a3c-test-16workers", "state": "finished",
        "author": "Jan Kowalski (jkowalski)", "created_at": "2026-11-06T08:00:00",
        "workers": 16, "workers_key": "num_workers", "mean_reward": 187.42,
        "mean_reward_key": "mean_reward", "duration": "1 h 12 min",
        "url": f"{UI_BASE}/{ENTITY}/{PROJECT}", "highlights": ["To jest wiadomość testowa."],
    }]
    send_email(build_email(sample))
    log.info("Wysłano testowego maila na %s", EMAIL_TO)


def cmd_run(dry_run: bool, notify_existing: bool) -> None:
    runs = fetch_runs()                                   # 1. OBSERWUJ
    state = load_state()
    current_ids = {r.id for r in runs}

    if state is None and not notify_existing:             # pierwsze uruchomienie
        log.info("Pierwsze uruchomienie: zapamiętuję %d istniejących runów, "
                 "kolejne nowe będą zgłaszane mailem.", len(runs))
        if not dry_run:
            save_state(current_ids)
        return

    seen = set(state["seen_run_ids"]) if state else set()
    new_runs = [r for r in runs if r.id not in seen]      # 2. PORÓWNAJ
    log.info("Runów w projekcie: %d, nowych: %d", len(runs), len(new_runs))
    if not new_runs:
        if not dry_run:
            save_state(seen | current_ids)
        return

    new_runs.sort(key=lambda r: str(getattr(r, "created_at", "")))
    infos = [describe_run(r) for r in new_runs]           # 3. ZROZUM
    msg = build_email(infos)

    if dry_run:
        print("\n----- PODGLĄD MAILA (nie wysłano) -----")
        print("Temat:", msg["Subject"])
        print(msg.get_body(preferencelist=("plain",)).get_content())
        return

    send_email(msg)                                       # 4. DZIAŁAJ
    log.info("Wysłano powiadomienie o %d runach na %s", len(infos), EMAIL_TO)
    save_state(seen | current_ids)                        # 5. ZAPAMIĘTAJ


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent pilnujacy nowych eksperymentów A2C/A3C w W&B")
    parser.add_argument("--check", action="store_true", help="test połączenia i podgląd pól")
    parser.add_argument("--test-email", action="store_true", help="wyślij testowego maila")
    parser.add_argument("--dry-run", action="store_true", help="nic nie wysyłaj i nie zapisuj")
    parser.add_argument("--notify-existing", action="store_true",
                        help="przy pierwszym uruchomieniu zgłoś też istniejące runy")
    args = parser.parse_args()
    setup_logging()

    try:
        if args.check:
            cmd_check()
        elif args.test_email:
            cmd_test_email()
        else:
            cmd_run(args.dry_run, args.notify_existing)
        return 0
    except Exception as exc:
        log.exception("Błąd: %s", exc)
        if not (args.check or args.test_email or args.dry_run):
            send_error_email(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

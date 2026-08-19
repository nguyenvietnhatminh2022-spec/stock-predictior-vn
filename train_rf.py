#!/usr/bin/env python3
"""
Background training runner for the Random Forest signal predictor.

Usage:
    python train_rf.py                    # full run with auto-tuning
    python train_rf.py --quick            # faster run with fewer params
    python train_rf.py --no-tune          # skip tuning
    python train_rf.py --no-charts        # skip chart generation
    python train_rf.py --no-save          # don't persist artifacts
    python train_rf.py --daemon           # run in background (nohup)
    python train_rf.py --config config.yaml  # load args from config file
    python train_rf.py --notify           # send notification on completion

Config file (YAML) example:
    no_tune: false
    quick: false
    no_charts: false
    no_save: false
    feature_select: true
"""

import os
import sys
import time
import subprocess
import argparse
import shlex
from datetime import datetime
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_config(path: str) -> dict:
    """Load configuration from YAML file."""
    if not HAS_YAML:
        raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_command(args: argparse.Namespace, script_dir: str) -> list[str]:
    """Build the command to execute."""
    venv_python = os.path.expandvars(r"%USERPROFILE%\.venv\Scripts\python.exe")
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    target_script = os.path.join(script_dir, "stock_signal_random_forest.py")
    cmd = [venv_python, "-u", target_script]

    # Map args to CLI flags
    flag_map = {
        "no_tune": "--no-tune",
        "quick": "--quick",
        "no_save": "--no-save",
        "no_charts": "--no-charts",
        "feature_select": "--feature-select",
    }
    for attr, flag in flag_map.items():
        if getattr(args, attr, False):
            cmd.append(flag)
    return cmd


def run_daemon(cmd: list[str], log_file: str) -> int:
    """Run command in background (nohup-style) and return PID."""
    # Use start /B on Windows, nohup on Unix
    if os.name == "nt":
        # Windows: use start /B to run in background
        full_cmd = f'start /B "" {subprocess.list2cmdline(cmd)}'
        proc = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        # Unix: nohup
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                ["nohup"] + cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    return proc.pid


def run_interactive(cmd: list[str], log_file: str) -> int:
    """Run command interactively, streaming output to console and log."""
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"Command: {' '.join(shlex.quote(c) for c in cmd)}\n")
        lf.write(f"Started: {datetime.now().isoformat()}\n\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        proc.wait()
    return proc.returncode


def send_notification(title: str, message: str):
    """Send desktop notification (cross-platform best effort)."""
    try:
        if os.name == "nt":
            # Windows: use PowerShell toast
            ps_cmd = (
                f'powershell -Command "'
                f'[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; '
                f'$template = [Windows.UI.Notifications.ToastTemplateType]::ToastText02; '
                f'$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($template); '
                f'$xml.GetElementsByTagName(\"text\")[0].AppendChild($xml.CreateTextNode(\"{title}\")); '
                f'$xml.GetElementsByTagName(\"text\")[1].AppendChild($xml.CreateTextNode(\"{message}\")); '
                f'$toast = [Windows.UI.Notifications.ToastNotification]::new($xml); '
                f'[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier(\"RF Trainer\").Show($toast)"'
            )
            subprocess.run(ps_cmd, shell=True, capture_output=True)
        else:
            # Linux/macOS: notify-send or osascript
            subprocess.run(["notify-send", title, message], capture_output=True)
    except Exception:
        pass  # Notifications are best-effort


def main():
    ap = argparse.ArgumentParser(
        description="VN30 Random Forest background trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument("--no-tune", action="store_true", help="skip hyper-parameter search")
    ap.add_argument("--quick", action="store_true", help="fast demo mode (fewer features/estimators)")
    ap.add_argument("--no-save", action="store_true", help="do not persist the model")
    ap.add_argument("--no-charts", action="store_true", help="skip PNG visualisations")
    ap.add_argument("--feature-select", action="store_true", help="run permutation feature selection")
    ap.add_argument("--daemon", action="store_true", help="run in background (detached)")
    ap.add_argument("--config", type=str, help="path to YAML config file")
    ap.add_argument("--notify", action="store_true", help="send desktop notification on completion")
    args = ap.parse_args()

    # Load config file if provided
    if args.config:
        if not os.path.exists(args.config):
            sys.exit(f"Config file not found: {args.config}")
        cfg = load_config(args.config)
        for k, v in cfg.items():
            if hasattr(args, k) and isinstance(v, bool):
                setattr(args, k, v)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(
        script_dir,
        f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )

    cmd = build_command(args, script_dir)

    print("=" * 60)
    print("  RF Trainer — Background Runner")
    print("=" * 60)
    print(f"  Python: {cmd[0]}")
    print(f"  Script: {cmd[1]}")
    print(f"  Args:   {' '.join(cmd[2:])}")
    print(f"  Log:    {log_file}")
    print(f"  Mode:   {'Daemon (background)' if args.daemon else 'Interactive'}")
    print("=" * 60)

    start = time.time()

    if args.daemon:
        pid = run_daemon(cmd, log_file)
        print(f"\n[DAEMON] Started in background with PID {pid}")
        print(f"         Output logged to: {log_file}")
        print(f"         Check status with: tail -f {log_file}")
        if args.notify:
            send_notification("RF Training Started", f"PID {pid} - check log for progress")
        return 0
    else:
        returncode = run_interactive(cmd, log_file)
        elapsed = time.time() - start
        status = "SUCCESS" if returncode == 0 else f"FAILED (exit code {returncode})"
        print(f"\n[DONE] {status} in {elapsed:.1f}s. Log saved to {log_file}")
        if args.notify:
            send_notification("RF Training Complete", f"{status} in {elapsed:.1f}s")
        return returncode


if __name__ == "__main__":
    sys.exit(main())
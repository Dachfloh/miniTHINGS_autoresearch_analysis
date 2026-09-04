#!/usr/bin/env python3
"""
Runs the results-monitoring loop below every 5 seconds, watch-style:
clears the screen and shows a header (interval / timestamp) each refresh.
Ctrl+C to quit.
"""

import csv
import shutil
import sys
import time
from datetime import datetime

INTERVAL = 5.0

files = [
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent1/results.tsv",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent2/results.tsv",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent3/results.tsv",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent4/results.tsv",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent5/results.tsv",
]


def run_once():
    for filepath in files:
        try:
            with open(filepath) as f:
                print(filepath, '\nExperiments completed: ', (sum(1 for _ in f) - 2))
            with open(filepath, newline='') as f:
                reader = csv.DictReader(f, delimiter='\t')
                rows = list(reader)          # read once, reuse below
                if rows:
                    max_value = max(float(row['test_acc']) for row in rows)
                    print(f"Max accuracy: {max_value}\n")
                else:
                    print("starting ...")
        except FileNotFoundError:
            print(filepath, "\n(not created yet)\n")
        except Exception as e:
            print(filepath, f"\n(error reading file: {e})\n")


def clear_screen():
    # ANSI: move cursor home + clear screen (same trick `watch` uses)
    sys.stdout.write("\x1b[H\x1b[2J")
    sys.stdout.flush()


def make_header(columns):
    now = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    left = f"Every {INTERVAL:.1f}s: monitor.py"
    padding = max(columns - len(left) - len(now), 2)
    header = left + " " * padding + now
    return header[:columns] if columns > 0 else header


def main():
    try:
        while True:
            columns, _ = shutil.get_terminal_size(fallback=(80, 24))
            clear_screen()
            sys.stdout.write(make_header(columns) + "\n\n")
            run_once()
            sys.stdout.flush()
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        clear_screen()
        sys.exit(0)


if __name__ == "__main__":
    main()

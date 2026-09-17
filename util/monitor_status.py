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
import re


INTERVAL = 5.0

files = [
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent1/",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent2/",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent3/",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent4/",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent5/",
]

def get_n_epochs(filepath):
    pattern = re.compile(r"'n_epochs':\s*(\d+)")
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return int(match.group(1))
    return None  # not found

def count_word(filepath, word="EPOCH"):
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            count += line.split().count(word)
    return count


def run_once():
    for dirpath in files:
        filepath = dirpath + "results.tsv"
        try:
            with open(filepath) as f:
                print(filepath, '\nExperiments completed: ', (sum(1 for _ in f) - 2))
            with open(filepath, newline='') as f:
                reader = csv.DictReader(f, delimiter='\t')
                rows = list(reader)          # read once, reuse below
                if rows:
                    if rows[-1]['status'] == 'final':
                        max_value = max(float(row['val_acc']) for row in rows[:-1])
                        print(f"Max val accuracy: {max_value}")
                        print(f"Test accuracy: {rows[-1]['val_acc']}\n")
                    else:
                        max_value = max(float(row['val_acc']) for row in rows)
                        print(f"Max val accuracy: {max_value}")
		        
                        epoch_count = count_word(dirpath + 'run.log')
                        epoch_max = get_n_epochs(dirpath + 'run.log')
                        filled = int(40 * epoch_count / epoch_max)
                        bar = ("=" * filled)[:-1] + ">" if filled > 0 else ""
                        print(f"|{bar:<40}| {epoch_count}/{epoch_max}\n")
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

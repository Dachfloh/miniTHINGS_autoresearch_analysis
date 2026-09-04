import csv

files = [
  "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent1/results.tsv",
  "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent2/results.tsv",
  "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent3/results.tsv",
  "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent4/results.tsv",
  "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent5/results.tsv",
]

for filepath in files:
  with open(filepath) as f:
    print(filepath, '\nExperiments completed: ', (sum(1 for _ in f) - 2))

  with open(filepath, newline='') as f:
    reader = csv.DictReader(f, delimiter='\t')
    if list(reader):
        max_value = max(float(row['test_acc']) for row in reader)
        print(f"Max accuracy: {max_value}\n")
    else: 
        print("starting ...")

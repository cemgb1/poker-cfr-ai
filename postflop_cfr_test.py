import random
import csv
import time
from collections import defaultdict

# === CONFIG ===
MAX_ITERATIONS = 300
TEST_BOARDS = [
    ['2C', '7D', 'KD'],
    ['AH', '4S', '9H'],
    ['8C', '9C', 'TC'],
    ['5S', '5D', '5C'],
    ['3H', '3C', '8S']
]
ACTIONS = ['fold', 'call', 'raise']

# === MOCK SCENARIO SETUP ===
def generate_scenarios():
    scenarios = []
    for board in TEST_BOARDS:
        for hand in ['AK', 'QJ', '77', 'T9', '32']:
            for position in ['SB', 'BB']:
                scenarios.append((hand, tuple(board), position, ''))
    return scenarios

# === MAIN TRAINING ===
strategy_map = defaultdict(lambda: defaultdict(float))
scenario_visits = defaultdict(int)
scenarios = generate_scenarios()

start_time = time.time()

for iteration in range(MAX_ITERATIONS):
    for hand, board, position, history in scenarios:
        info_set = f"{hand}|{''.join(board)}|{position}|{history}"
        strategy = {a: random.random() for a in ACTIONS}
        total = sum(strategy.values())
        for a in strategy:
            strategy[a] /= total  # Normalize
        strategy_map[info_set] = strategy
        scenario_visits[info_set] += 1

    if (iteration + 1) % 50 == 0 or iteration == MAX_ITERATIONS - 1:
        print(f"✅ Completed iteration {iteration + 1} of {MAX_ITERATIONS}")

end_time = time.time()
print(f"\n🕒 Finished training in {round(end_time - start_time, 2)} seconds.")

# === CSV OUTPUT ===
def parse_info_set(info_set):
    hand, board, position, history = info_set.split('|')
    return hand, board, position, history

with open("postflop_strategy_output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Hand", "Board", "Position", "History", "Action", "Probability", "Visits"])
    
    for info_set, strategy in strategy_map.items():
        hand, board, position, history = parse_info_set(info_set)
        visits = scenario_visits.get(info_set, 0)
        for action, prob in strategy.items():
            writer.writerow([hand, board, position, history, action, round(prob, 4), visits])

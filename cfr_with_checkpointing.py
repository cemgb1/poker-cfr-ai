# Add the missing methods to cfr_with_checkpointing.py
cat >> cfr_with_checkpointing.py << 'EOF'

    def generate_all_169_hands(self):
        """Generate all 169 possible starting hands"""
        hands = []
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        
        for i, rank1 in enumerate(ranks):
            for j, rank2 in enumerate(ranks):
                if i < j:
                    hands.append(rank1 + rank2 + 's')
                    hands.append(rank1 + rank2 + 'o')
                elif i == j:
                    hands.append(rank1 + rank2)
        return hands

    def create_hand_groups(self):
        """Create hand grouping system"""
        groups = {}
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

        for i, rank1 in enumerate(ranks):
            for j, rank2 in enumerate(ranks):
                if i <= j:
                    if rank1 == rank2:
                        groups[rank1 + rank2] = self.categorize_pair(rank1)
                    else:
                        suited_hand = rank1 + rank2 + 's'
                        offsuit_hand = rank1 + rank2 + 'o'
                        groups[suited_hand] = self.categorize_suited(rank1, rank2)
                        groups[offsuit_hand] = self.categorize_offsuit(rank1, rank2)
        return groups

    def categorize_pair(self, rank):
        """Categorize pocket pairs"""
        if rank in ['A', 'K', 'Q']:
            return 'premium_pairs'
        elif rank in ['J', 'T', '9']:
            return 'medium_pairs'
        else:
            return 'small_pairs'

    def categorize_suited(self, rank1, rank2):
        """Categorize suited hands"""
        high_rank, low_rank = rank1, rank2

        if high_rank == 'A':
            if low_rank in ['K', 'Q', 'J']:
                return 'premium_aces_suited'
            elif low_rank in ['T', '9', '8']:
                return 'medium_aces_suited'
            else:
                return 'weak_aces_suited'
        elif high_rank == 'K':
            if low_rank in ['Q', 'J', 'T']:
                return 'premium_kings_suited'
            else:
                return 'medium_kings_suited'
        elif high_rank in ['Q', 'J', 'T']:
            if self.is_connected(rank1, rank2):
                return 'suited_connectors_high'
            else:
                return 'suited_broadways'
        elif self.is_connected(rank1, rank2):
            return 'suited_connectors_low'
        else:
            return 'suited_trash'

    def categorize_offsuit(self, rank1, rank2):
        """Categorize offsuit hands"""
        high_rank, low_rank = rank1, rank2

        if high_rank == 'A':
            if low_rank in ['K', 'Q', 'J']:
                return 'premium_aces_offsuit'
            elif low_rank in ['T', '9']:
                return 'medium_aces_offsuit'
            else:
                return 'weak_aces_offsuit'
        elif high_rank == 'K':
            if low_rank in ['Q', 'J']:
                return 'premium_kings_offsuit'
            else:
                return 'weak_kings_offsuit'
        elif high_rank in ['Q', 'J', 'T'] and low_rank in ['J', 'T', '9']:
            return 'offsuit_broadways'
        else:
            return 'offsuit_trash'

    def is_connected(self, rank1, rank2):
        """Check if ranks are connected"""
        rank_order = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        try:
            pos1 = rank_order.index(rank1)
            pos2 = rank_order.index(rank2)
            return abs(pos1 - pos2) == 1
        except:
            return False

    def generate_all_preflop_scenarios(self):
        """Generate ALL possible preflop training scenarios"""
        scenarios = []
        hand_groups = set(self.hand_groups.values())
        positions = ["BTN", "BB"]
        histories = ["", "r", "rr", "cr", "rrr", "rrc", "crr"]
        stack_depths = ["short", "medium", "deep"]

        for group in hand_groups:
            for position in positions:
                for history in histories:
                    for stack_depth in stack_depths:
                        scenario = f"{group}|{position}|{history}|{stack_depth}"
                        scenarios.append(scenario)
        
        return scenarios

    def cfr_preflop(self, p0_hand, p1_hand, history, p0, p1, player, stack_bb):
        """CFR recursion for preflop play"""
        import random
        
        # Terminal check
        if self.is_terminal(history):
            return self.get_payoff(p0_hand, p1_hand, history, player, stack_bb)

        # Get information set
        current_hand = p0_hand if player == 0 else p1_hand
        hand_group = self.hand_groups[current_hand]
        position = "BTN" if player == 0 else "BB"
        stack_cat = "deep" if stack_bb > 75 else "medium" if stack_bb > 35 else "short"
        
        info_set = f"{hand_group}|{position}|{history}|{stack_cat}"
        
        # Get strategy
        strategy = self.get_strategy(info_set, p0 if player == 0 else p1)
        
        # Get valid actions
        valid_actions = self.get_valid_actions(history)
        
        # Calculate utilities for each action
        utilities = np.zeros(3)
        for action in valid_actions:
            new_history = history + ['f', 'c', 'r'][action]
            
            if player == 0:
                utilities[action] = -self.cfr_preflop(
                    p0_hand, p1_hand, new_history, p0 * strategy[action], p1, 1, stack_bb
                )
            else:
                utilities[action] = -self.cfr_preflop(
                    p0_hand, p1_hand, new_history, p0, p1 * strategy[action], 0, stack_bb
                )

        # Calculate node utility and update regrets
        node_util = np.sum(strategy * utilities)
        
        for action in valid_actions:
            regret = utilities[action] - node_util
            self.regret_sum[info_set][action] += (p1 if player == 0 else p0) * regret

        return node_util

    def is_terminal(self, history):
        """Check if game state is terminal"""
        return (
            len(history) > 0 and history[-1] == 'f' or
            history in ['cc', 'rc', 'crc'] or
            len(history) >= 4
        )

    def get_payoff(self, p0_hand, p1_hand, history, player, stack_bb):
        """Calculate payoff for terminal states"""
        if history and history[-1] == 'f':
            pot_size = 2 + history[:-1].count('r')
            folder = (len(history) - 1) % 2
            return pot_size if player != folder else -pot_size
        else:
            p0_strength = self.get_hand_strength(p0_hand)
            p1_strength = self.get_hand_strength(p1_hand)
            pot_size = 2 + history.count('r')

            if p0_strength > p1_strength:
                return pot_size if player == 0 else -pot_size
            elif p1_strength > p0_strength:
                return -pot_size if player == 0 else pot_size
            else:
                return 0

    def get_hand_strength(self, hand):
        """Calculate relative hand strength"""
        group = self.hand_groups[hand]
        group_strengths = {
            'premium_pairs': 95, 'medium_pairs': 75, 'small_pairs': 45,
            'premium_aces_suited': 90, 'premium_aces_offsuit': 85,
            'medium_aces_suited': 70, 'medium_aces_offsuit': 60,
            'weak_aces_suited': 50, 'weak_aces_offsuit': 35,
            'premium_kings_suited': 80, 'premium_kings_offsuit': 70,
            'medium_kings_suited': 55, 'suited_connectors_high': 65,
            'suited_connectors_low': 40, 'suited_broadways': 60,
            'offsuit_broadways': 50, 'suited_trash': 25, 'offsuit_trash': 15
        }
        return group_strengths.get(group, 30)

    def get_valid_actions(self, history):
        """Get valid actions for current game state"""
        if len(history) == 0:
            return [1, 2]  # check or raise (can't fold first)
        elif len(history) >= 3:
            return [0, 1]  # fold or call only (betting cap)
        else:
            return [0, 1, 2]  # fold, call, raise

    def get_strategy(self, info_set, reach_prob):
        """Get strategy using regret matching"""
        regrets = self.regret_sum[info_set]
        positive_regrets = np.maximum(regrets, 0)
        regret_sum = np.sum(positive_regrets)

        if regret_sum > 0:
            strategy = positive_regrets / regret_sum
        else:
            strategy = np.ones(3) / 3

        self.strategy_sum[info_set] += reach_prob * strategy
        return strategy

    def get_average_strategy(self, info_set):
        """Get average strategy over all iterations"""
        strategy_sum = self.strategy_sum[info_set]
        total = np.sum(strategy_sum)
        return strategy_sum / total if total > 0 else np.ones(3) / 3

    def get_strategy_for_hand(self, hand, position, history="", stack_depth="deep"):
        """Get optimal strategy for any hand in any situation"""
        if hand not in self.hand_groups:
            return None, f"Hand {hand} not recognized"

        hand_group = self.hand_groups[hand]
        info_set = f"{hand_group}|{position}|{history}|{stack_depth}"

        if info_set in self.strategy_sum and np.sum(self.strategy_sum[info_set]) > 0:
            strategy = self.get_average_strategy(info_set)
            best_action = ["FOLD", "CALL", "RAISE"][np.argmax(strategy)]

            return {
                'hand': hand,
                'group': hand_group,
                'position': position,
                'situation': history if history else "first_to_act",
                'stack_depth': stack_depth,
                'fold_prob': strategy[0],
                'call_prob': strategy[1],
                'raise_prob': strategy[2],
                'recommended_action': best_action,
                'confidence': np.max(strategy)
            }, None
        else:
            return None, f"No strategy learned for {info_set}"

EOF

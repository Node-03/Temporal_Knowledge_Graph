from collections import defaultdict
import random
import math
import numpy as np


class BeliefRLAgent:

    def __init__(self, env,
                 alpha=0.1,
                 gamma=0.95,
                 epsilon=0.1,
                 lambda_b=0.7,
                 beta=1.0,
                 k_max=3):

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_b = lambda_b
        self.beta = beta
        self.k_max = k_max

        self.Q = defaultdict(float)
        self.entities = self._extract_entities()

    # =========================
    # ENV
    # =========================
    def _extract_entities(self):
        entities = set()
        for h, r, t, time in self.env.train:
            entities.add(h)
            entities.add(t)
        return list(entities)

    def _get_actions(self, e, t):
        return [(r, t2) for (h, r, t2, time) in self.env.train if h == e and time == t]

    # =========================
    # belief
    # =========================
    def _init_belief(self):
        p = 1.0 / len(self.entities)
        return {e: p for e in self.entities}

    def _update_belief(self, b, next_e, query_r):
        b[next_e] = self.lambda_b * b.get(next_e, 0) + (1 - self.lambda_b)

        # relation bias
        for (h, r, t, time) in self.env.train:
            if r == query_r:
                b[t] = b.get(t, 0) + 0.001

        return self._normalize(b)

    def _normalize(self, b):
        total = sum(b.values())
        if total == 0:
            return b
        return {k: v / total for k, v in b.items()}

    def _entropy(self, b):
        return -sum(v * math.log(v + 1e-9) for v in b.values())

    # =========================
    # Q
    # =========================
    def _get_Q(self, state, action):
        return self.Q[(state, action)]

    def _set_Q(self, state, action, value):
        self.Q[(state, action)] = value

    # =========================
    # policy
    # =========================
    def _select_action(self, state, actions, belief):

        if random.random() < self.epsilon:
            return random.choice(actions)

        best_a = None
        best_score = -1e9

        for a in actions:
            _, next_e = a
            score = self._get_Q(state, a) + self.beta * belief.get(next_e, 0)

            if score > best_score:
                best_score = score
                best_a = a

        return best_a

    # =========================
    # training (with logging)
    # =========================
    def train(self, epochs=5, episodes_per_query=1):

        for epoch in range(epochs):

            td_errors = []
            success = 0
            total = 0
            entropies = []

            print(f"\nEpoch {epoch}")

            for i, query in enumerate(self.env.train):

                s, r, o, t = query

                for _ in range(episodes_per_query):

                    e = s
                    belief = self._init_belief()

                    for k in range(self.k_max):

                        state = (e, r, k)
                        actions = self._get_actions(e, t)

                        if not actions:
                            break

                        a = self._select_action(state, actions, belief)
                        _, next_e = a

                        reward = 1.0 if next_e == o else 0.0
                        reward -= 0.01 * k

                        next_state = (next_e, r, k + 1)
                        next_actions = self._get_actions(next_e, t)

                        max_next_Q = 0
                        if next_actions:
                            max_next_Q = max(self._get_Q(next_state, na) for na in next_actions)

                        old_Q = self._get_Q(state, a)
                        td_error = reward + self.gamma * max_next_Q - old_Q

                        new_Q = old_Q + self.alpha * td_error
                        self._set_Q(state, a, new_Q)

                        td_errors.append(abs(td_error))

                        belief = self._update_belief(belief, next_e, r)
                        entropies.append(self._entropy(belief))

                        e = next_e

                        if e == o:
                            success += 1
                            break

                total += 1

                # 中間觀察
                if i % 300 == 0:
                    pred, _ = self.infer(query)
                    print(f"  sample pred: {pred}, gt: {o}")

            # epoch summary
            print({
                "TD_error": float(np.mean(td_errors)),
                "train_acc": success / total,
                "entropy": float(np.mean(entropies))
            })

    # =========================
    # inference
    # =========================
    def infer(self, query, rollout=20):

        s, r, _, t = query
        final_belief = defaultdict(float)

        for _ in range(rollout):

            e = s
            belief = self._init_belief()

            for k in range(self.k_max):

                state = (e, r, k)
                actions = self._get_actions(e, t)

                if not actions:
                    break

                a = self._select_action(state, actions, belief)
                _, next_e = a

                belief = self._update_belief(belief, next_e, r)
                e = next_e

            for k, v in belief.items():
                final_belief[k] += v

        final_belief = self._normalize(final_belief)
        pred = max(final_belief, key=final_belief.get)

        return pred, final_belief
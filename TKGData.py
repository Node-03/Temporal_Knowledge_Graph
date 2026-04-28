from collections import defaultdict, Counter
import os
import random

# =========================================================
# 1. Dataset Layer（純資料，不含 split）
# =========================================================
class TKGDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.entity2id = {}
        self.relation2id = {}

        self.quads = self._read_all()
    def stats(self):
        entities = set()
        relations = set()
        times = []

        for s, r, o, t in self.quads:
            entities.add(s)
            entities.add(o)
            relations.add(r)
            times.append(t)

        return {
            "num_quads": len(self.quads),
            "num_entities": len(entities),
            "num_relations": len(relations),
            "time_range": (min(times), max(times)) if times else None
        }
    def _read_all(self):
        path = os.path.join(self.data_dir, "train.txt")

        quads = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue

                s, r, o, t = parts[:4]

                try:
                    t = int(t)
                except:
                    continue

                s_id = self._get_ent(s)
                o_id = self._get_ent(o)
                r_id = self._get_rel(r)

                quads.append((s_id, r_id, o_id, t))

        return quads

    def _get_ent(self, e):
        if e not in self.entity2id:
            self.entity2id[e] = len(self.entity2id)
        return self.entity2id[e]

    def _get_rel(self, r):
        if r not in self.relation2id:
            self.relation2id[r] = len(self.relation2id)
        return self.relation2id[r]


# =========================================================
# 2. Split Layer（完全獨立）
# =========================================================
class TKGSplitter:
    """
    split_mode 定義資料切分策略：
    random:
        隨機切分（i.i.d.），忽略時間順序。
    temporal:
        時間切分（temporal setting），
        train: t < target_time
        test : t = target_time
    split_ratios:
        僅適用於 random 模式。
    target_time:
        temporal 模式必要參數。
    """
    def __init__(self, quads, split_mode="random",
                 split_ratios=(0.8, 0.1, 0.1),
                 target_time=None):

        self.quads = quads
        self.split_mode = split_mode
        self.split_ratios = split_ratios
        self.target_time = target_time
        
    def stats(self, train, valid, test):
        def _stat(data):
            times = [t for _,_,_,t in data]
            return {
                "size": len(data),
                "time_range": (min(times), max(times)) if times else None
            }

        return {
            "train": _stat(train),
            "valid": _stat(valid),
            "test": _stat(test)
        }
    
    def split(self):
        
        if self.split_mode == "random":
            return self._random_split()

        elif self.split_mode == "temporal":
            return self._temporal_split()

        else:
            raise ValueError("Unknown split_mode")

    def _random_split(self):
        quads = self.quads[:]
        random.shuffle(quads)

        n = len(quads)
        n_train = int(n * self.split_ratios[0])
        n_valid = int(n * self.split_ratios[1])

        return (
            quads[:n_train],
            quads[n_train:n_train+n_valid],
            quads[n_train+n_valid:]
        )

    def _temporal_split(self):
        if self.target_time is None:
            raise ValueError("target_time required")

        train = [q for q in self.quads if q[3] < self.target_time]
        test  = [q for q in self.quads if q[3] == self.target_time]

        return train, [], test


# =========================================================
# 3. Graph Builder（環境）
# =========================================================
class TKGGraphBuilder:
    def __init__(self, quads, max_actions=20):
        self.quads = quads
        self.max_actions = max_actions

    def build(self):
        graph = defaultdict(list)

        for s, r, o, _ in self.quads:
            graph[s].append((r, o))

        for s in graph:
            if len(graph[s]) > self.max_actions:
                graph[s] = graph[s][:self.max_actions]

        return graph
# =========================================================
# 4. Diagnostics（統一評估）
# =========================================================
class TKGDiagnostics:

    # -------------------------
    # 基本分佈
    # -------------------------
    @staticmethod
    def time_distribution(quads):
        from collections import Counter
        return dict(Counter(t for _,_,_,t in quads))

    @staticmethod
    def relation_distribution(quads):
        from collections import Counter
        return dict(Counter(r for _,r,_,_ in quads))

    # -------------------------
    # Query 層觀測
    # -------------------------
    @staticmethod
    def query_overlap(train, test):
        train_set = set((s,r,t) for s,r,_,t in train)
        test_set  = set((s,r,t) for s,r,_,t in test)

        if not test_set:
            return 0

        return len(train_set & test_set) / len(test_set)

    @staticmethod
    def answer_existence(graph, queries):
        hit = 0

        for s,_,o,_ in queries:
            if any(o == dst for _,dst in graph.get(s, [])):
                hit += 1

        return hit / len(queries) if queries else 0

    @staticmethod
    def unreachable_ratio(graph, queries):
        unreachable = 0

        for s,_,_,_ in queries:
            if s not in graph:
                unreachable += 1

        return unreachable / len(queries) if queries else 0

    @staticmethod
    def graph_stats(graph):
        num_nodes = len(graph)
        degrees = [len(v) for v in graph.values()]

        total_edges = sum(degrees)
        avg_degree = total_edges / num_nodes if num_nodes else 0

        return {
            "nodes": num_nodes,
            "edges": total_edges,
            "avg_degree": avg_degree,
            "max_degree": max(degrees) if degrees else 0
        }

    @staticmethod
    def query_stats(graph, queries):
        total = len(queries)

        start = 0
        one_hop = 0

        for s, r, o, t in queries:
            if s in graph:
                start += 1

                if any(o == dst for _, dst in graph[s]):
                    one_hop += 1

        return {
            "total": total,
            "start_ratio": start / total if total else 0,
            "1hop_ratio": one_hop / total if total else 0
        }

    @staticmethod
    def multi_hop(graph, s, target, depth=3):
        visited = set([s])
        frontier = [s]

        for _ in range(depth):
            nxt = []
            for node in frontier:
                for _, nei in graph.get(node, []):
                    if nei == target:
                        return True
                    if nei not in visited:
                        visited.add(nei)
                        nxt.append(nei)
            frontier = nxt

        return False

    @classmethod
    def full(cls, graph, queries, depth=3):
        base = cls.query_stats(graph, queries)

        multi = 0
        for s, r, o, t in queries:
            if s in graph and cls.multi_hop(graph, s, o, depth):
                multi += 1

        base["multi_hop_ratio"] = multi / base["total"] if base["total"] else 0
        return base

    @classmethod
    def evaluate_environment(cls, graph, queries):
        return {
            "graph": cls.graph_stats(graph),
            "query": cls.full(graph, queries)
        }


# =========================================================
# 5. High-level Wrapper（你實驗用的入口）
# =========================================================
class TKGEnvironment:
    def __init__(self,
                 data_dir,
                 target_time,
                 history_len,
                 filter_mode="none",   # 預設不刪資料（重要）
                 save_env=False,
                 save_dir="./cache_env"):

        self.dataset = TKGDataset(data_dir)

        # 1. temporal split
        constructor = TKGTemporalConstructor(
            self.dataset.quads,
            target_time,
            history_len
        )
        self.train_raw, self.test_raw = constructor.split()

        # 2. graph
        if history_len == 0:
            builder = TKGGraphBuilder(self.test_raw)   # ← 關鍵
        else:
            builder = TKGGraphBuilder(self.train_raw)

        self.graph = builder.build()

        # 3. filter
        filter_fn = getattr(TKGFilter, filter_mode)
        self.train = filter_fn(self.graph, self.train_raw)
        self.test  = filter_fn(self.graph, self.test_raw)

        if save_env:
            self.save_environment(save_dir)

    # -------------------------
    # summary（簡潔版）
    # -------------------------
    def summary(self):
        return {
            "dataset": self.dataset.stats(),

            "temporal": {
                "train_size": len(self.train),
                "test_size": len(self.test)
            },

            "environment": {
                "train": TKGDiagnostics.evaluate_environment(
                    self.graph, self.train
                ),
                "test": TKGDiagnostics.evaluate_environment(
                    self.graph, self.test
                )
            }
        }

    # -------------------------
    # save（只存 final）
    # -------------------------
    def save_environment(self, save_dir):
        import os, json, pickle
        os.makedirs(save_dir, exist_ok=True)

        def _write(path, data):
            with open(path, "w") as f:
                for s, r, o, t in data:
                    f.write(f"{s}\t{r}\t{o}\t{t}\n")

        _write(os.path.join(save_dir, "train.txt"), self.train)
        _write(os.path.join(save_dir, "test.txt"), self.test)

        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            json.dump(self.summary(), f, indent=2)

        with open(os.path.join(save_dir, "graph.pkl"), "wb") as f:
            pickle.dump(self.graph, f)


class TKGFilter:
    @staticmethod
    def startable(graph, quads):
        return [q for q in quads if q[0] in graph]

    @staticmethod
    def answer_in_graph(graph, quads):
        return [
            (s,r,o,t)
            for s,r,o,t in quads
            if any(o == dst for _,dst in graph.get(s, []))
        ]

    @staticmethod
    def none(graph, quads):
        return quads
    


class TKGTemporalConstructor:
    def __init__(self, quads, target_time, history_len):
        self.quads = quads
        self.t = target_time
        self.L = history_len

    def split(self):
        # --------
        # case 1: no temporal constraint
        # --------
        if self.t is None:
            return self.quads, []

        # --------
        # case 2: no-history (L=0)
        # --------
        if self.L == 0:
            train = []
            test  = [q for q in self.quads if q[3] == self.t]
            return train, test

        # --------
        # case 3: normal temporal window
        # --------
        t_start = self.t - self.L

        train = [q for q in self.quads if t_start <= q[3] < self.t]
        test  = [q for q in self.quads if q[3] == self.t]

        return train, test
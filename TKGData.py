from collections import defaultdict, Counter
import os
import random

# =========================================================
# 1. Dataset Layer（一次性讀取全部，記錄子集）
# =========================================================
class TKGDataset:
    """
    讀取 data_dir 下的 train.txt / test.txt / valid.txt（若存在），
    合併為 self.quads（完整資料集），同時保留各子集供 file 模式使用。
    實體／關係編碼基於所有檔案建立。
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity2id = {}
        self.relation2id = {}

        # 預設會讀取的檔案
        file_names = ["train.txt", "test.txt", "valid.txt"]
        self.quads = []                # 完整四元組
        self.subsets = {}              # 各子集: dict of list

        for fname in file_names:
            path = os.path.join(data_dir, fname)
            if os.path.exists(path):
                quads = self._read_file(fname)
                self.subsets[fname] = quads
                self.quads.extend(quads)

    def stats(self):
        # 收集整體時間跨度（可選，仍可保留）
        all_times = [q[3] for q in self.quads]
        # 各子集大小
        file_sizes = {fname: len(quads) for fname, quads in self.subsets.items()}
        return {
            "num_entities": len(self.entity2id),
            "num_relations": len(self.relation2id),
            "time_range": (min(all_times), max(all_times)) if all_times else None,
            "file_sizes": file_sizes,                     # ← 取代 num_quads
            "total_quads": len(self.quads)                # 可選保留總數，或直接刪除
        }

    def _read_file(self, filename):
        """讀取單一檔案並轉換為 ID"""
        path = os.path.join(self.data_dir, filename)
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
# 2. Graph Builder（環境）
# =========================================================
class TKGGraphBuilder:
    def __init__(self, quads):
        self.quads = quads

    def build(self):
        graph = defaultdict(list)
        for s, r, o, _ in self.quads:
            graph[s].append((r, o))
        return graph


# =========================================================
# 3. Diagnostics（統一評估）
# =========================================================
class TKGDiagnostics:
    @staticmethod
    def time_distribution(quads):
        return dict(Counter(t for _, _, _, t in quads))

    @staticmethod
    def relation_distribution(quads):
        return dict(Counter(r for _, r, _, _ in quads))

    @staticmethod
    def query_overlap(train, test):
        train_set = set((s, r, t) for s, r, _, t in train)
        test_set = set((s, r, t) for s, r, _, t in test)
        if not test_set:
            return 0
        return len(train_set & test_set) / len(test_set)

    @staticmethod
    def answer_existence(graph, queries):
        hit = 0
        for s, _, o, _ in queries:
            if any(o == dst for _, dst in graph.get(s, [])):
                hit += 1
        return hit / len(queries) if queries else 0

    @staticmethod
    def unreachable_ratio(graph, queries):
        unreachable = 0
        for s, _, _, _ in queries:
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
        start = one_hop = 0
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
        visited = {s}
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
# 4. Filter 工具
# =========================================================
class TKGFilter:
    @staticmethod
    def startable(graph, quads):
        return [q for q in quads if q[0] in graph]

    @staticmethod
    def answer_in_graph(graph, quads):
        return [
            (s, r, o, t)
            for s, r, o, t in quads
            if any(o == dst for _, dst in graph.get(s, []))
        ]

    @staticmethod
    def none(graph, quads):
        return quads


# =========================================================
# 5. Temporal Window Splitter（輕量時間切割）
# =========================================================
class TemporalWindowSplitter:
    """
    從完整 quads 中切割 train / test。
    - history_len: 要回溯「額外」的時間步數。
        * 0 → train 只包含 target_time 的事實
        * 1 → train 包含 target_time 及前一個時間點的事實
        * n → train 包含 target_time 及最近 n 個歷史時間點的事實
    - test 一律是 target_time 的全部事實（與 train 可能重疊，這是合理的靜態圖評估設定）。
    """
    def __init__(self, quads, target_time, history_len):
        self.quads = quads
        self.t = target_time
        self.L = history_len

    def split(self):
        if self.t is None:
            return self.quads, []

        # 收集並排序所有時間點
        all_times = sorted(set(q[3] for q in self.quads))
        try:
            idx = all_times.index(self.t)
        except ValueError:
            return [], []               # 目標時間不存在

        # 決定歷史時間點（不包含 t 自己，歷史會在後面跟 t 合併）
        if self.L == 0 or idx == 0:
            hist_times = set()
        else:
            start = max(0, idx - self.L)
            hist_times = set(all_times[start:idx])

        # 訓練時間 = 歷史 + 當前 t
        train_times = hist_times | {self.t}

        train = [q for q in self.quads if q[3] in train_times]
        test  = [q for q in self.quads if q[3] == self.t]
        return train, test

# =========================================================
# 6. High‑level Wrapper（支援 file / temporal 模式）
# =========================================================
class TKGEnvironment:
    """
    mode = 'file'（預設）:
        直接從 data_dir 下讀取 train.txt / test.txt / valid.txt，
        分別當作 train / test / valid 分割。
    mode = 'temporal':
        將 data_dir 下所有檔案合併成完整資料集，再根據 target_time
        與 history_len 進行時間窗切割。
    filter_mode: 可選 "none", "startable", "answer_in_graph"
    """
    def __init__(self,
                 data_dir,
                 mode='file',
                 target_time=None,
                 history_len=None,
                 filter_mode="none",
                 save_env=False,
                 save_dir="./cache_env"):

        # ----- 1. 完整讀取並建立映射 -----
        self.dataset = TKGDataset(data_dir)

        # ----- 2. 根據模式取得原始分割 -----
        if mode == 'file':
            self.train_raw = self.dataset.subsets.get("train.txt", [])
            self.test_raw  = self.dataset.subsets.get("test.txt", [])
            self.valid_raw = self.dataset.subsets.get("valid.txt", [])

        elif mode == 'temporal':
            if target_time is None or history_len is None:
                raise ValueError("temporal mode 需要 target_time 與 history_len")
            splitter = TemporalWindowSplitter(self.dataset.quads,
                                              target_time, history_len)
            self.train_raw, _ = splitter.split()          # 只保留動態切割的訓練集
            self.test_raw = self.dataset.subsets.get("test.txt", [])   # 固定回歸原始測試集
            self.valid_raw = self.dataset.subsets.get("valid.txt", []) # 固定回歸原始驗證集

        else:
            raise ValueError("mode 必須為 'file' 或 'temporal'")

        # ----- 3. 建構查詢圖 -----
        if self.train_raw:
            builder = TKGGraphBuilder(self.train_raw)
        else:
            builder = TKGGraphBuilder(self.test_raw)
        self.graph = builder.build()

        # ----- 4. 過濾 -----
        filter_fn = getattr(TKGFilter, filter_mode)
        self.train = filter_fn(self.graph, self.train_raw)
        self.test  = filter_fn(self.graph, self.test_raw)
        self.valid = filter_fn(self.graph, self.valid_raw)

        # ----- 5. 可選保存 -----
        if save_env:
            self.save_environment(save_dir)

    def summary(self):
        return {
            "dataset": self.dataset.stats(),
            "split": {
                "train_size": len(self.train),
                "test_size": len(self.test),
                "valid_size": len(self.valid)
            },
            "environment": {
                "train": TKGDiagnostics.evaluate_environment(self.graph, self.train),
                "test": TKGDiagnostics.evaluate_environment(self.graph, self.test)
            }
        }

    def save_environment(self, save_dir):
        import json, pickle
        os.makedirs(save_dir, exist_ok=True)

        def _write(path, data):
            with open(path, "w") as f:
                for s, r, o, t in data:
                    f.write(f"{s}\t{r}\t{o}\t{t}\n")

        _write(os.path.join(save_dir, "train.txt"), self.train)
        _write(os.path.join(save_dir, "test.txt"), self.test)
        if self.valid:
            _write(os.path.join(save_dir, "valid.txt"), self.valid)

        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            json.dump(self.summary(), f, indent=2)

        with open(os.path.join(save_dir, "graph.pkl"), "wb") as f:
            pickle.dump(self.graph, f)
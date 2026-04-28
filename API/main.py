from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional, List, Dict, Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "../cache_env"

# ===== 基礎工具 =====

def parse_line(line: str) -> List[int]:
    """將一行資料轉為整數列表，保留所有欄位（包含時間）"""
    return [int(x) for x in line.split()]

def read_triples(filepath: str) -> List[List[int]]:
    """讀取四元組檔案，每筆為 [h, r, t, timestamp]"""
    triples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triples.append(parse_line(line))
    return triples

def read_mapping(filepath: str) -> Dict[int, str]:
    """讀取 id -> name 對應表"""
    mapping = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid line {i}: {line}")
            try:
                idx = int(parts[-1])
            except ValueError:
                raise ValueError(f"Invalid id at line {i}: {line}")
            name = " ".join(parts[:-1])
            mapping[idx] = name
    return mapping

# 預先載入映射表（只載一次，不隨請求重複讀取）── 注意：若檔案會更新，可改成每次讀取
entity_map = read_mapping(os.path.join(DATA_DIR, "entity2id.txt"))
relation_map = read_mapping(os.path.join(DATA_DIR, "relation2id.txt"))

# ===== API 端點 =====

@app.get("/timestamps")
def get_timestamps():
    """取得所有出現的時間戳（排序）"""
    triples = read_triples(os.path.join(DATA_DIR, "train.txt"))
    timestamps = sorted({row[3] for row in triples})
    return {"timestamps": timestamps}

@app.get("/data")
def get_graph(
    time: Optional[int] = Query(None, description="精確時間點，等同 start=end=time"),
    start: Optional[int] = Query(None, description="時間區間開始"),
    end: Optional[int] = Query(None, description="時間區間結束")
):
    """
    回傳節點與邊（支援時間過濾）。
    - 若無時間參數：回傳所有四元組。
    - 若提供 time：回傳 timestamp == time 的資料。
    - 若提供 start / end：回傳 timestamp 在 [start, end] 內的資料。
    """
    # 整理時間條件
    if time is not None:
        start = end = time
    # 讀取全部四元組
    all_triples = read_triples(os.path.join(DATA_DIR, "train.txt"))
    
    # 根據時間過濾
    if start is not None and end is not None:
        filtered_triples = [
            row for row in all_triples
            if start <= row[3] <= end
        ]
    else:
        filtered_triples = all_triples  # 無條件時回傳全部

    # 建立節點與邊
    nodes_dict: Dict[int, dict] = {}
    links: List[dict] = []

    for row in filtered_triples:
        h, r, t, ts = row[0], row[1], row[2], row[3]

        # 頭實體節點
        if h not in nodes_dict:
            nodes_dict[h] = {
                "id": str(h),
                "label": entity_map.get(h, str(h))
            }
        # 尾實體節點
        if t not in nodes_dict:
            nodes_dict[t] = {
                "id": str(t),
                "label": entity_map.get(t, str(t))
            }

        links.append({
            "source": str(h),
            "target": str(t),
            "relation": relation_map.get(r, str(r)),
            "timestamp": ts
        })

    return {
        "nodes": list(nodes_dict.values()),
        "links": links
    }
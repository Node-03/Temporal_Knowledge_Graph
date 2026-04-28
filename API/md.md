# Graph API 簡介 (Graph API Overview)

## 概述 (Overview)

此 API 提供知識圖譜 (Knowledge Graph) 的基本資料存取，將後端的 ID-based 三元組資料轉換為前端可直接使用的圖結構。

---

## Endpoint

### `GET /data`

回傳完整圖資料。

---

## 回傳格式 (Response Schema)

```json
{
  "nodes": [
    { "id": "21084", "label": "Make statement" }
  ],
  "links": [
    {
      "source": "21084",
      "target": "8010",
      "relation": "Use conventional military force"
    }
  ]
}
```

---

## 欄位說明 (Field Description)

### nodes

表示圖中的節點 (entities)

(a) `id`: 節點 ID（字串）
(b) `label`: 節點名稱（由 entity mapping 轉換）

---

### links

表示節點之間的關係 (edges)

(a) `source`: 起點節點 ID
(b) `target`: 終點節點 ID
(c) `relation`: 關係名稱（由 relation mapping 轉換）

---

## 資料來源 (Data Source)

API 從以下檔案讀取資料：

(a) `entity2id.txt`：實體名稱 → ID
(b) `relation2id.txt`：關係名稱 → ID
(c) `data.txt`：三元組資料（head, relation, tail）

---

## 備註 (Notes)

(a) 原始資料使用整數 ID 表示，API 已轉換為可讀 label
(b) 前端無需解析 `.txt` 檔案
(c) 回傳資料可直接用於 D3 或其他圖視覺化工具

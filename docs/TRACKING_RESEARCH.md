# 蝙蝠熱影像偵測與軌跡呈現研究

## 結論

目前系統最大的限制不是線條繪製，而是把單幀 YOLO 結果直接交給相鄰幀規則配對。
當目標只有約 10 x 11 像素、偶爾漏偵測、快速轉向或兩隻蝙蝠交會時，只調中心距離、
IoU 與 Kalman 門檻，必然會在「軌跡斷裂」和「錯誤接線」之間反覆擺盪。

本專案建議改成以下四層：

1. 高召回候選層：YOLO 切片偵測，加上多幀差分或背景模型產生移動候選。
2. 多幀確認層：候選必須在數幀內形成合理運動，才升級為正式蝙蝠。
3. 軌跡求解層：先以 OC-SORT 或 ByteTrack 建立 baseline，再改為短視窗全域圖配對。
4. 呈現與分析層：區分原始候選、確認框、暫時遺失與正式軌跡，另輸出事件報表與熱區圖。

## 專案現況

- 影片：800 x 600、9 FPS、15,672 幀，攝影機固定。
- 標註：126 張影像、230 個框。
- 標註框平均大小：約 10.7 x 11.4 像素。
- 現行偵測：單幀 YOLO，`conf=0.3`、`imgsz=960`。
- 現行追蹤：自製 Kalman + Hungarian，成本由中心距離、預測誤差、方向、IoU、
  框面積與信心組成。
- 現行誤判處理：偵測前與追蹤內各有一套原地停留刪除邏輯。

主要風險：

- 極小物體的單幀外觀資訊不足，YOLO 漏框會直接造成斷線。
- 相鄰幀貪婪配對一旦接錯，後續 Kalman 會延續錯誤方向。
- 兩隻蝙蝠接近或交會時，僅靠位置和框大小不足以避免 ID switch。
- 固定背景熱點、樹葉與壓縮雜訊會被低信心門檻放大。
- `Total Detected` 現在接近「曾建立的 ID 數」，不等於真正飛過的蝙蝠事件數。

## 研究對照

### ByteTrack

ByteTrack 的重點不是丟掉低信心框，而是先用高信心框配對，再用低信心框救回短暫漏檢。
這正好對應目前「有偵測跡象但線斷掉」的問題，且 Ultralytics 8.4.35 已內建，可快速建立
baseline。不過它主要依賴 IoU 與線性運動，對快速轉向、極小框和蝙蝠交會仍有限制。

### OC-SORT

OC-SORT 用實際觀測修正遮擋期間累積的 Kalman 誤差，並加入方向一致性與失聯後重新配對。
論文特別針對非線性運動，概念上比目前的自製規則更符合蝙蝠飛行。現有 Ultralytics
8.4.35 尚未內建，需要升級或獨立整合。

### BoT-SORT 與 ReID

BoT-SORT 結合運動、相機運動補償與可選的外觀特徵。此影片攝影機固定，因此相機運動補償
價值不高。蝙蝠框平均只有十幾個像素，且熱影像外觀相近，通用 ReID 很可能學到背景而非
個體差異，不應作為第一優先。若未來有更高解析度或多光譜影像，再評估專用 ReID。

### SAHI 切片偵測

SAHI 將畫面切成重疊區塊後放大推論，可提升小物體在 detector 輸入中的有效像素數。
研究在小物體資料集上顯示切片推論與切片微調均可提高 AP。此專案畫面只有 800 x 600，
可先測試 2 x 2 重疊切片，而不是無限制提高整張圖的 `imgsz`。切片邊界的重複框必須用
NMS 或 weighted box fusion 合併。

### 多幀紅外小目標偵測

紅外小目標研究普遍利用鄰近幀的時間資訊，因為單幀中的小亮點與背景雜訊難以區分。
可行做法從簡單到複雜包括：

- 時間中位數背景 + 三幀差分。
- 光流對齊後的時間特徵聚合。
- 以 5 至 9 幀影像堆疊訓練 3D CNN、ConvLSTM 或 temporal attention detector。
- 先產生低門檻候選，再要求候選形成連續、可解釋的運動才保留。

對目前只有 126 張標註影像的資料量，先做「背景差分候選 + YOLO 確認」比直接訓練大型
時序網路更實際。

### 全域資料關聯

目前 Hungarian assignment 每次只決定當前幀，錯誤發生後無法利用未來幀修正。影片是
離線分析，不必受限於純線上追蹤。可把 1 至 3 秒內的 detections 建成有向圖：

- 節點：每個候選框。
- 邊：允許跨 1 至數個漏檢幀連接。
- 邊成本：速度、加速度、轉向角、框尺度、熱強度 patch 相似度、YOLO confidence。
- 約束：一個 detection 只能屬於一條軌跡；同一軌跡同幀只能有一個 detection。
- 求解：min-cost flow、動態規劃或多假設追蹤。

短視窗全域求解可以同時比較多種接法，明顯比「最近的就是同一隻」更能避免兩隻蝙蝠被
接成同一條不存在的線。允許 0.5 至 1.5 秒輸出延遲即可兼顧接近即時的畫面。

## 建議新架構

```text
Video Reader
    |
    +-- Background / temporal-difference proposals
    |
    +-- Tiled YOLO detections
              |
        Candidate Fusion + NMS
              |
        Temporal Candidate Validator
        - 固定熱點刪除
        - 最短連續觀測
        - 速度範圍
        - 移動方向與加速度
              |
        Sliding-window Track Graph
        - 低信心框可救回既有軌跡
        - 允許短暫漏檢
        - 全域一對一約束
              |
        Track State Machine
        tentative -> confirmed -> lost -> ended/rejected
              |
        Renderer + Event Exporter
```

### 候選融合

YOLO 與運動候選不應互相取代：

- YOLO 高信心 + 有運動：高優先。
- YOLO 低信心 + 軌跡預測吻合：允許續接，但不能建立新軌跡。
- 無 YOLO + 強運動 + 前後幀都吻合：標成 provisional observation，可補一個漏框。
- YOLO 高信心 + 長期固定：降權或拒絕。
- 僅單幀出現：先顯示淺灰候選框，不計入正式數量。

### 防止錯接的硬性規則

- 同一 detection 不可分配給兩個 ID。
- 兩條 confirmed 軌跡交會後，不因最近距離立即交換 ID；比較交會前後的速度與加速度。
- 新軌跡不可在既有軌跡預測區內直接建立，除非同幀有兩個可分離 detection。
- 跨漏檢連線必須通過最大加速度與最大轉角檢查。
- 無法確定時寧可保留兩段短軌跡，不畫一條高風險的長線。
- `Total Bats` 應在軌跡結束且通過最短長度、位移、平均信心後才增加。

## 呈現方式

畫面上不要把所有狀態都畫成相同權重：

- 淺灰細框：單幀 detector 候選，尚未確認。
- 彩色實線框：confirmed track。
- 彩色虛線框：tracker 預測位置，該幀沒有 detector 支持。
- 軌跡線：最近 1.5 至 3 秒；越舊越淡，末端加方向箭頭。
- 預測補線：使用虛線，不能和真實觀測的實線混在一起。
- ID 標籤：顯示 ID、信心與狀態，不顯示沒有分析價值的長文字。
- 交會狀態：兩隻蝙蝠接近時，暫時減少線條亮度並保留各自預測方向。

另外輸出三種結果：

1. 標註影片：給人檢查 detection 與 ID。
2. 軌跡事件表：每個 ID 的開始/結束 frame、時間、持續秒數、位移、路徑長、
   平均/最低信心、最大漏檢間隔、結束原因。
3. 分析圖：軌跡總覽、穿越熱區圖、每分鐘事件數；不要把整段歷史線全部疊在主影片。

## 評估方式

沒有量化評估時，繼續調參只會依賴少數畫面的主觀印象。建議從原影片標註至少 5 段，
每段 10 至 30 秒，包含：

- 單隻清楚飛行。
- 多隻接近或交會。
- 短暫漏檢。
- 樹葉或固定熱點誤判。
- 快速移動與突然轉向。

標註需包含每幀 bbox 與跨幀 track ID。評估分開看：

- Detection：precision、recall、false positives per minute。
- Tracking：HOTA、DetA、AssA、IDF1、ID switches、fragmentations。
- 專案事件：真實飛行事件召回率、每分鐘錯誤事件數、錯誤長連線數。

HOTA 會把偵測與關聯拆開，能判斷問題到底是「沒框到」還是「框有了但 ID 接錯」。
TrackEval 可直接評估 MOTChallenge 格式。

## 實作順序

## 已落地的第一階段

- 新增 `detection_pipeline.py`：支援全圖與 2 x 2 重疊切片推論、跨切片 NMS、
  低信心候選的幀差運動驗證。
- 新增 `bytetrack_tracker.py`：提供官方 ByteTrack baseline，實測可確認它不適合直接
  處理本影片的大幅相鄰幀位移。
- 將原 motion tracker 改成 hybrid tracker：先配對高信心框，再用低信心框續接尚未
  配對的既有軌跡；低信心框不能建立新 ID。
- 正式軌跡必須至少 3 次觀測且淨位移超過固定點半徑，避免背景抖動被計數。
- 畫面分開顯示 Candidates、Active Bats 與 Confirmed Flights，並在軌跡末端顯示箭頭。
- 新增 MP4 輸出與 `--detector full|tiled`、`--tracker hybrid|bytetrack|motion`。
- 新增 bbox 面積 proxy 的相對遠近分級：輸出 `bbox_width`、`bbox_height`、`bbox_area`、
  `relative_distance_score = 1 / sqrt(area)` 與 `distance_bin`。此分級只能作為 near / mid / far
  的趨勢判讀，不代表實際公尺距離。

30 幀實測中，全圖 960 與 2 x 2 切片都確認 4 條軌跡；CPU 平均速度分別約為
1.39 FPS 與 0.68 FPS。因此目前預設使用全圖，切片模式留給漏檢嚴重片段或離線高召回分析。

### 第一階段：建立可比較 baseline

- 保留現行結果。
- 新增 Ultralytics ByteTrack 版本。
- 統一輸出 MOTChallenge 文字檔與事件表。
- 建立 5 段帶 track ID 的驗證集。
- 使用 HOTA、DetA、AssA、IDF1 比較，不再只看總數或單張預覽。

### 第二階段：改善小目標召回

- 測試 2 x 2 重疊切片 YOLO。
- 加入時間中位數背景與三幀差分候選。
- 低信心候選只能續接 confirmed track，不能直接增加總數。
- 移除主流程與 tracker 內重複的 stationary filter，集中為 candidate validator。

### 第三階段：改善軌跡正確性

- 先整合 OC-SORT 作為非線性 motion baseline。
- 再實作 9 至 27 幀 sliding-window graph association。
- 加入 acceleration、turn angle、intensity patch 與 gap penalty。
- 軌跡結束後做一次 forward-backward smoothing，再輸出最終線與總數。

### 第四階段：研究型提升

- 蒐集更多連續影片標註，而非只增加互相相似的單張截圖。
- 訓練 5 至 9 幀 temporal detector 或中心點 segmentation model。
- 若取得同步雙相機，改做立體幾何與 3D 軌跡，可大幅降低 2D 投影造成的交會歧義。

## 不建議優先採用

- 繼續增加更多手工 short-step / long-step 門檻。
- 對十幾像素的熱影像框直接套通用人物 ReID。
- 只降低 YOLO confidence 後把所有框建立成新 ID。
- 用畫線平滑掩蓋 ID switch；線變順不代表關聯正確。
- 以 `confirmed_total` 當成研究結論，卻沒有軌跡完成條件與 ground truth。

## 參考資料

- ByteTrack: <https://arxiv.org/abs/2110.06864>
- OC-SORT: <https://arxiv.org/abs/2203.14360>
- BoT-SORT: <https://arxiv.org/abs/2206.14651>
- Ultralytics tracking documentation:
  <https://docs.ultralytics.com/modes/track/>
- SAHI: <https://arxiv.org/abs/2202.06934>
- Flow-Guided Feature Aggregation:
  <https://arxiv.org/abs/1703.10025>
- Infrared small-target video pipeline:
  <https://arxiv.org/abs/2012.02579>
- Bidirectional temporal infrared small-target detection:
  <https://arxiv.org/abs/2508.15415>
- Global min-cost data association:
  <https://arxiv.org/abs/1703.10764>
- HOTA: <https://arxiv.org/abs/2009.07736>
- TrackEval: <https://github.com/JonathonLuiten/TrackEval>
- Bat trajectories from synchronized thermal cameras:
  <https://arxiv.org/abs/1303.3072>

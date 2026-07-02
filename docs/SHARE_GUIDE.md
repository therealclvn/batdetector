# 蝙蝠熱影像追蹤工具

此分享包包含程式、YOLO 模型、跨平台啟動腳本與範例圖片，不包含原始影片或 Python 虛擬環境。

## 系統需求

- Python 3.11
- 第一次安裝套件時需要網路
- macOS、Windows 或 Linux

## 快速啟動

### macOS

1. 解壓縮後執行 `run_mac.command`。
2. 將要分析的影片拖曳到終端機視窗。
3. 按 Enter 開始。

若 macOS 阻擋執行，可在終端機執行：

```bash
chmod +x run_mac.command
./run_mac.command "/path/to/video.mp4"
```

### Windows

1. 解壓縮後執行 `run_windows.bat`。
2. 輸入影片完整路徑。
3. 按 Enter 開始。

### 命令列

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py --video "/path/to/video.mp4"
```

Windows 請將 `.venv/bin/` 改為 `.venv\Scripts\`。

## 播放控制

- `Progress`：拖曳至指定 frame。
- `Speed`：`0.25x`、`0.5x`、`1x`、`2x`、`4x`。
- `Space`：暫停或繼續。
- `A` / `D`：前後移動 5 秒。
- `[` / `]`：切換倍速。
- `Q` 或 `Esc`：結束並輸出報表。

跳轉影片後會重設追蹤狀態，避免不同片段被錯誤連成同一條軌跡。`2x` 與 `4x` 會跳過部分 frame，正式完整分析請使用 `1x`。

## 輸出

預設 Excel 報表會建立在：

```text
outputs/reports/bat_detection_report.xlsx
```

輸出標註影片：

```bash
.venv/bin/python main.py \
  --video "/path/to/video.mp4" \
  --headless \
  --video-output outputs/bat_result.mp4 \
  --excel-output outputs/bat_result.xlsx
```

## 模型與模式

- 預設模型：`models/best3.pt`
- 預設 tracker：`hybrid`
- 預設 detector：`full`
- 漏檢嚴重時可加 `--detector tiled`，但速度較慢。

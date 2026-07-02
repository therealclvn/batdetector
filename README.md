# Bat Model 3

熱影像蝙蝠偵測、追蹤、軌跡繪製與 Excel 報表輸出。

## 目錄

```text
models/             目前使用的模型
models/archive/     舊模型
videos/             目前分析的影片
videos/archive/     其他或舊影片
data/               訓練與標註資料
outputs/previews/   正式預覽圖片與影片
outputs/reports/    正式 Excel 報表
docs/               研究與設計文件
```

## 執行

```bash
venv/bin/python main.py --video videos/bat_vid2.mkv
```

macOS 可直接雙擊 `啟動蝙蝠監測.command` 播放預設影片 `videos/bat_vid2.mkv`。也可以執行 `run_mac.command` 或 `run_windows.bat`，再輸入影片路徑。

預設會同時載入：

- 蝙蝠模型：`models/best3.pt`
- 風力發電風扇模型：`models/blade_best.pt`

風扇模型目前只負責在畫面上標示風扇／葉片位置，不會判斷撞擊風險。若只想跑蝙蝠偵測，可加入：

```bash
venv/bin/python main.py --video videos/bat_vid2.mkv --no-blade
```

若風扇模型放在其他位置，可指定：

```bash
venv/bin/python main.py --blade-model models/blade_best.pt
```

程式會在畫面中央建立紅色橢圓危險區。只要已確認的蝙蝠軌跡進入過危險區一次，就會計入危險蝙蝠數，並在數量統計下方顯示百分比：

```text
Impact Risk: 危險蝙蝠數 / 確認蝙蝠總數 * 100%
```

這裡的百分比是偵測到的蝙蝠進入危險區比例，不是經真實撞擊資料校準的物理撞擊機率。

不同影片若風扇位置不同，可用 `--danger-zone` 調整橢圓：

```bash
venv/bin/python main.py --danger-zone 400,240,160,100
```

格式為 `中心X,中心Y,半徑X,半徑Y`；也可加旋轉角度：`中心X,中心Y,半徑X,半徑Y,角度`。

## 播放控制

- `Progress`：拖曳到指定影片 frame；跳轉後會清除舊軌跡，避免跨片段誤接。
- `Speed`：選擇 `0.25x`、`0.5x`、`1x`、`2x`、`4x`。
- `Space`：暫停或繼續。
- `A` / `D`：向前或向後跳 5 秒。
- `[` / `]`：降低或提高倍速。
- `Q` 或 `Esc`：結束並輸出 Excel。

`2x` 與 `4x` 會跳過部分 frame，適合快速尋找片段；正式完整分析請使用 `1x`。

輸出影片：

```bash
venv/bin/python main.py --headless \
  --video-output outputs/previews/bat_result.mp4 \
  --excel-output outputs/reports/bat_result.xlsx
```

極小目標漏檢嚴重時可加入 `--detector tiled`，但處理速度較慢。

## 測試

```bash
venv/bin/python -m unittest discover -v
venv/bin/python main.py --headless --max-frames 3 --no-excel --tracker hybrid
venv/bin/python main.py --headless --max-frames 3 --no-excel --tracker motion
```

## 重建環境

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

建議使用 Python 3.11。首次安裝需要網路，蝙蝠模型已放在 `models/best3.pt`，風扇模型預設放在 `models/blade_best.pt`。

## GitHub 協作

GitHub 只放程式碼與文件，不放大型本機檔案。組員 clone 後請自行建立環境：

```bash
git clone https://github.com/CalvinKuo1213/batdetector.git
cd batdetector
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

模型與影片請另外分享後放回指定位置：

```text
models/best3.pt
models/blade_best.pt
videos/bat_vid2.mkv
```

被排除不上傳的內容包含 `venv/`、`models/*.pt`、`videos/*.mkv`、`videos/*.mp4`、`data/`、`outputs/`。

## 分享版本

分享壓縮檔不包含大型原始影片、訓練資料及虛擬環境。解壓縮後執行對應系統的啟動腳本，並指定自己的影片即可。

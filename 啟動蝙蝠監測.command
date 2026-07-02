#!/bin/bash
set -e

cd "$(dirname "$0")"

VIDEO_PATH="videos/bat_vid2.mkv"

if [ ! -f "$VIDEO_PATH" ]; then
  echo "找不到預設影片: $VIDEO_PATH"
  echo "請先把要播放的影片放到 videos/bat_vid2.mkv，或改用 run_mac.command 手動指定影片。"
  echo
  read -r -p "按 Enter 關閉視窗..."
  exit 1
fi

if [ -x "venv/bin/python" ]; then
  PYTHON_BIN="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  echo "找不到 Python 虛擬環境，正在建立 .venv..."
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
  PYTHON_BIN=".venv/bin/python"
fi

echo "啟動蝙蝠監測..."
echo "影片: $VIDEO_PATH"
echo

"$PYTHON_BIN" main.py --video "$VIDEO_PATH"

echo
read -r -p "程式已結束，按 Enter 關閉視窗..."

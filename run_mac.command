#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

VIDEO_PATH="$1"
if [ -z "$VIDEO_PATH" ]; then
  printf "請將影片拖曳到此視窗後按 Enter: "
  read -r VIDEO_PATH
fi

VIDEO_PATH="${VIDEO_PATH#\"}"
VIDEO_PATH="${VIDEO_PATH%\"}"
exec .venv/bin/python main.py --video "$VIDEO_PATH"

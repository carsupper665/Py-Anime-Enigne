Py Anime Engine — 圖片/動圖/影片去背

簡介
- 本專案提供桌面版去背工具，支援圖片、動圖與影片的前景擷取與輸出。
- 引擎：HSV（可調）、rembg、OpenVINO、魔術棒（wand）。
- 內建處理佇列（非阻塞）、LoadingToast 進度、可設定輸出格式與路徑。

系統需求
- Python 3.11+
- 依賴：`pip install -r requirements.txt`
- 必備：ffmpeg（抽幀/封裝/轉檔）
  - Linux/macOS/Windows 請將 ffmpeg 放入 PATH
  - 官網原始碼：https://git.ffmpeg.org/ffmpeg.git 或套件管理器安裝
- 影片播放（選用）：PyQt6 多媒體相依（缺少時仍可用 ffmpeg 擷取影格預覽）

快速開始
- 執行：`./run.sh` 或 `python main.py`
- 如果是首次啟動，可於 Settings 設定：
  - 輸出資料夾（預設 `./animes`）
  - HSV 預設值
  - OpenVINO 模型路徑（選用，`RMBG_MODEL_PATH` 亦可）

功能總覽
- 圖片/動圖/影片去背：以非阻塞方式運作，UI 持續可操作
- 影片 in/out：可於播放器設定入點/出點，僅處理此範圍
- 影片輸出：強制輸出 animated WebP（含透明、無音訊）
- 佇列：支援加入佇列、暫停、繼續、取消當前
- LoadingToast：顯示 extract → frame i/N → encode → done 進度
- 魔術棒（wand）：點選影像取樣 seed，自動生成遮罩範圍

使用教學
1) 圖片/動圖
- 拖放或「選擇檔案」載入
- 勾選「remove background」→ 選擇引擎與參數 →
  - 直接「儲存」或按「加入佇列」
- 未勾選時「儲存」：
  - 非支援直拷格式將轉為 WebP（靜態：WebP 無損；動圖：animated WebP）

2) 影片
- 載入影片後可使用：播放、設入點/出點（毫秒）
- 點「預覽影格」：
  - 彈出單幀預覽，支援 HSV 預覽疊色、重新擷取、點擊取樣 wand seed
  - 「套用到控制」可將預覽調整套回主介面控制
- 勾選「remove background」→ 儲存或加入佇列：
  - 工作會入列並開始處理（抽幀→逐幀去背→封裝 animated WebP）
  - 進度以 LoadingToast 顯示

3) 佇列操作
- Home 頁右側：
  - 「加入佇列」：加入當前設定的去背任務
  - 「暫停/繼續/取消當前」：控制佇列與目前任務

常見問題
- FFmpeg Not Found：請安裝 ffmpeg 並確保在 PATH
- 無多媒體相依（WSL 等）：影片仍可「預覽影格」（ffmpeg 擷取）與去背；但播放器無法預覽播放
- 魔術棒未取樣：engine=wand 需要先於圖片或「預覽影格」點擊以取得 seed
- OpenVINO 未設定：若選擇 openvino 引擎，請於 Settings 設定模型路徑或環境變數

OpenSpec 與規格連結
- Processing Queue：`openspec/changes/add-processing-queue/specs/processing-queue/spec.md`
- Queue + LoadingToast：`openspec/changes/update-queue-loading-toast/specs/processing-queue/spec.md`
- 非阻塞一律走佇列：`openspec/changes/update-queue-nonblocking-integration/specs/processing-queue/spec.md`
- 影片去背（animated WebP、in/out）：`openspec/changes/update-video-remove-bg/specs/video-remove-bg/spec.md`
- 影片預覽/取樣重構：`openspec/changes/refactor-video-remove-bg/specs/video-remove-bg/spec.md`
- 修正 Home 影片 UI：`openspec/changes/fix-home-video-ui/specs/home-ui/spec.md`

字型
- 思源黑體（Source Han Sans）：https://github.com/adobe-fonts/source-han-sans

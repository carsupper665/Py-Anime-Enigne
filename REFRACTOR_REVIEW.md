# Refactor Review Summary

整合先前審查提出的所有重構需求，依模組/職責分類列出現有問題、造成的影響與建議的調整方向，便於規劃後續工作拆解。

## `ui/main.py`
### `Main._on_rmbg`
- **問題**：單一方法負責 UI loading、環境變數注入、設定合併、輸出參數 fallback、時間區間解析等，超過 50 行並夾雜多層 `try/except`，資料流混亂。
- **影響**：難以掌握錯誤來源與資料流，測試覆蓋率低，日後加入新分支時容易造成回歸。
- **建議**：
  - 拆出專責 helper（例如 `build_export_params`、`parse_time_range`、`apply_engine_overrides`）。
  - 錯誤處理集中在真正可能失敗的區塊，並以明確的輸入/輸出資料結構（dataclass 或 NamedTuple）傳遞上下文。

## `ui/home_page.py`
### `HomePage._save_with_prompt`
- **問題**：同一方法同時處理對話框、設定檔更新、佇列排程、暫存檔、ffmpeg 轉檔與通知回報，依賴大量旗標與內部狀態。
- **影響**：流程難以閱讀與測試，排程與即時儲存彼此耦合，擴充新儲存策略成本高。
- **建議**：
  - 切成「收集輸入」「持久化設定」「建立佇列任務」「即時儲存執行」等函式。
  - 將儲存選項封裝成 dataclass，主流程只處理流程控制與錯誤轉譯。

### 儲存對話框預設副檔名
- **問題**：預設沿用原始副檔名（例如 `.mp4`），但後續流程會強制輸出 `animated-webp`。
- **影響**：使用者預期與實際輸出格式不一致，容易誤解為程式出錯。
- **建議**：預設副檔名改成 `.webp` 或在 UI 中明確提示輸出格式會被覆寫，並在儲存時同步調整路徑與訊息。

### 暫存檔命名策略
- **問題**：來源檔複製到 `./temp/<name>`，多個同名檔或同時處理時會互相覆寫，並可能留下髒檔。
- **影響**：佇列工作彼此干擾、檔案遺失或被覆蓋，造成難以追蹤的錯誤。
- **建議**：改用 `uuid` 或時間戳組成唯一暫存檔，並在工作完成後清理；或由背景任務自行管理其暫存目錄。

### 預覽與導出選項耦合
- **問題**：`_update_preview` 直接操作 numpy/OpenCV/PyQt 疊圖；`_apply_export_opts` 使用字串檢查拼裝 ffmpeg 參數，`_ffmpeg_trim`、`_ffmpeg_mute` 等 helper 深度耦合 UI。
- **影響**：UI 層過度了解媒體處理細節，難以重用或替換後端；錯誤處理與測試分散。
- **建議**：建立獨立 preview service，UI 只接收 `QPixmap`；導出選項封裝成 `ExportOptions` 物件，集中在 command builder 管理 ffmpeg 旗標。

### 影片工具列與 ffmpeg 呼叫
- **問題**：`_ffmpeg_trim`、`_ffmpeg_mute`、`_ask_output` 同時處理 UI 對話框與命令構建；`_apply_export_opts` 只支援 `libwebp_anim`。
- **影響**：測試困難、擴充新格式需改多個地方。
- **建議**：抽出 `VideoExportService` 或 command builder，統一命名規則、輸出路徑與 ffmpeg 選項，UI 只負責參數收集與回報。

## `ui/threads.py`
### `RmbgThread.remove_bg`
- **問題**：`_worker.finished` 被連結到 `self.finished` 兩次，任務完成時訊號發出兩次。
- **影響**：造成 UI 或佇列事件重複觸發（例如重複 reload 或進度更新）。
- **建議**：移除重複連結，或改在 thread 結束後統一轉發一次。

### `_RmbgWorker` 責任過載
- **問題**：`do_remove` 同時負責輸入檢查、輸出路由、環境變數載入、副檔名推斷；`_remove_image`/`_remove_anim_or_video` 混入進度回報、檔名、ffmpeg 子程序與暫存清理。
- **影響**：流程難以掌握，thread safety 不佳，狀態與副作用分散在多個欄位。
- **建議**：拆分成「輸入檢查與路由」「靜態圖處理」「影片/動畫管線」「輸出編碼」等 service；以 context manager 管理 ffmpeg 與暫存目錄，並用 dataclass 傳遞 export/max_fps/loop 等設定。

### `_remove_anim_or_video`
- **問題**：單一函式包含抽幀、逐幀處理、進度回報、fps 探測、輸出參數覆寫、ffmpeg 編碼與暫存清理。
- **影響**：難以測試與除錯，任何修改都可能影響整條管線。
- **建議**：拆成抽幀、逐幀處理、合成三段 pipeline；將 export 參數改為顯式入參，並以 context manager 管理暫存與統一進度回報。

## `ui/preview_dialog.py`
### `PreviewDialog`
- **問題**：類別同時負責 UI 初始化、事件綁定、影片解碼、動畫快取、滑桿同步、HSV 預覽與取樣等。
- **影響**：大量屬性相互依賴，修改其中一部分容易破壞其他功能。
- **建議**：拆成「媒體來源讀取器」「預覽計算器」「對話框呈現」三層，將影格存取與快取移至 model/service，透過介面取得 `QImage`，事件可由 presenter 處理以降低狀態複雜度。

## 其他可讀性問題
### FPS 限制判斷
- **問題**：寫法 `fps = [int(self.export_max_fps), int(fps)][1] if int(fps) < int(self.export_max_fps) else int(self.export_max_fps)` 可讀性極差。
- **影響**：無法立即辨識意圖，易於導入錯誤或在重構時被誤解。
- **建議**：改用 `fps = min(int(fps), int(self.export_max_fps))` 或明確命名的 helper，並加入單元測試確保上下限行為正確。

以上條目需納入重構排程，可依模組拆分成具體工作項目，逐步改善背景移除與導出流程的可維護性。

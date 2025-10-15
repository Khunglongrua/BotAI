# BotAI

Dự án BotAI là tập hợp các script Python phục vụ cho việc:
- Xử lý và lựa chọn đặc trưng từ dữ liệu thị trường (feature selection).
- Huấn luyện mô hình học máy (training với residual, multi-target).
- Chạy inference trực tiếp trên dữ liệu Binance (real-time/live).
- Gửi tín hiệu LONG/SHORT qua Discord/Telegram.

## 📂 Cấu trúc thư mục

- **.py** – Các script huấn luyện, inference, feature selection.
- **.json** – File cấu hình đặc trưng được chọn (selected_features, top30_features, …).
- **.txt** – File cấu hình phụ trợ (discord, telegram token/chat id, danh sách symbol, …).

## ⚙️ Yêu cầu

- Python 3.9+
- Thư viện: `numpy`, `pandas`, `tensorflow`, `scikit-learn`, `optuna`, …
- Môi trường ảo được khuyến nghị:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

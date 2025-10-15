import asyncio
import sys
from datetime import datetime

# ===============================
# Run ALL scripts sequentially, then wait a single countdown for the next cycle
# ===============================

# Đường dẫn python hiện tại (luôn đúng venv)
PYTHON_EXEC = sys.executable

# Danh sách script cần chạy (tuần tự, có thể thêm nhiều file)
scripts = [
    #"F3_infer_mi38_live_3y_any.py",
    "F3_infer_mi38_live_3y_filter.py",
    # thêm script khác nếu cần
]

# Số phút chờ giữa các vòng lặp (toàn bộ batch)
CYCLE_INTERVAL_MINUTES = 10

# Đếm số lần đã chạy cho từng file
execution_count = {f: 0 for f in scripts}


async def countdown(total_seconds: int, label: str) -> None:
    """Đếm ngược từng giây, in đè trên cùng một dòng."""
    for remaining in range(total_seconds, 0, -1):
        m, s = divmod(remaining, 60)
        print(f"⏳ {label} | còn lại {m:02d}:{s:02d}  ", end="\r", flush=True)
        await asyncio.sleep(1)
    print(" " * 60, end="\r")  # clear the line


async def run_all_scripts_once():
    """Chạy tuần tự toàn bộ scripts một lần."""
    for file in scripts:
        start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{start}] 🔁 Đang chạy {file} ...")
        try:
            process = await asyncio.create_subprocess_exec(
                PYTHON_EXEC, file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            execution_count[file] += 1
            if process.returncode == 0:
                print(f"✅ Hoàn tất {file} (lần {execution_count[file]})")
                if stdout:
                    print(stdout.decode().strip())
            else:
                print(f"⚠️ Lỗi {file} (mã {process.returncode}) (lần {execution_count[file]})")
                if stderr:
                    print(stderr.decode().strip())

        except Exception as e:
            print(f"❌ Không chạy được {file}: {e}")


async def main():
    while True:
        await run_all_scripts_once()
        # Sau khi CHẠY HẾT, mới đếm ngược 1 lần cho cả vòng
        await countdown(CYCLE_INTERVAL_MINUTES * 60,
                        f"Chờ chạy lại cả batch sau {CYCLE_INTERVAL_MINUTES} phút")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⛔ Đã dừng theo yêu cầu người dùng.")

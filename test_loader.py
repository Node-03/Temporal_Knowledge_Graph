from TKGData import  TKGEnvironment


env = TKGEnvironment(
    data_dir="./data/ICEWS18",
    target_time=5736,
    history_len=0,
    filter_mode="none",        # ← 不刪資料
    save_env=True
)
summary = env.summary()
print(summary)

# print(summary["dataset"])
# print(summary["environment"]["train"])
# print(summary["environment"]["test"])
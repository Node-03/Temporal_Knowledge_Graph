from TKGData import  TKGEnvironment


env = TKGEnvironment(
    data_dir="./data/ICEWS18",
    mode='temporal',
    target_time=5736,
    history_len=0,
    filter_mode="none",
    save_env=True,
    save_dir="cache_env"
)
summary = env.summary()
print(summary)

# print(summary["dataset"])
# print(summary["environment"]["train"])
# print(summary["environment"]["test"])
from TKGData import TKGEnvironment
from belief_rl_agent import BeliefRLAgent


def evaluate(agent, env, num_samples=500):

    correct = 0
    total = 0

    for query in env.test[:num_samples]:

        pred, belief = agent.infer(query)

        if pred == query[2]:
            correct += 1

        total += 1

    acc = correct / total
    print(f"\nTest Accuracy: {acc:.4f}")


if __name__ == "__main__":

    env = TKGEnvironment(
        data_dir="./data/ICEWS18",
        mode='temporal',
        target_time=5736,
        history_len=0,
        filter_mode="none",
        save_env=True,
        save_dir="cache_env"
    )

    print(env.summary())

    agent = BeliefRLAgent(
        env,
        beta=1.0,     # ⭐ 改 0 做 ablation
        k_max=6
    )

    print("\nStart Training...")
    agent.train(epochs=5)

    print("\nStart Evaluation...")
    evaluate(agent, env)
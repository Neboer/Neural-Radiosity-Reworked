import hashlib
from collected_result import CollectedResult
from network import train_model
import pickle
import os

def prepare_collected_data(data_dir):
    """
    准备收集的数据。如果存在已保存的 .pkl 文件，则加载它，
    否则创建一个新的 CollectedResult 并保存到 .pkl 文件。
    
    :param data_dir: 数据目录路径
    :return: CollectedResult 对象
    """
    # 计算 data_dir 的哈希值
    data_dir_hash = hashlib.md5(data_dir.encode()).hexdigest()
    result_dir = "collected_result"
    result_path = os.path.join(result_dir, f"{data_dir_hash}.pkl")

    # 确保存储目录存在
    os.makedirs(result_dir, exist_ok=True)

    # 检查 .pkl 文件是否存在
    if os.path.exists(result_path):
        with open(result_path, "rb") as f:
            print(f"Loading collected data from {result_path}")
            return pickle.load(f)
    else:
        # 创建新的 CollectedResult 对象
        print(f"No existing file found. Creating new CollectedResult for {data_dir}")
        collected_result = CollectedResult('../RadiosityCollecterOutput')

        # 保存到 .pkl 文件
        with open(result_path, "wb") as f:
            pickle.dump(collected_result, f)
        print(f"Collected data saved to {result_path}")

        return collected_result

if __name__ == "__main__":
    collected_result = prepare_collected_data('../RadiosityCollecterOutput')

    # Train the model
    training_data = collected_result.prepare_training_data()
    trained_model = train_model(training_data["posw"], training_data["direction"], training_data["color"], epochs=10, batch_size=1000)
    
    # Save the trained model to a file
    with open('checkpoints/trained_model.pkl', 'wb') as f:
        pickle.dump(trained_model, f)

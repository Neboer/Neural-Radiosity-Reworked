from network import predict
import pickle
from radiosity_data import DataSlice
from collected_result import CollectedResult

def model_render_radiosity_on_image(slice: DataSlice, model):
    """
    使用训练好的模型预测光照结果，并将结果渲染到图像上。
    """
    # 预测
    predicted_color = predict(model, slice.posw_collapsed, slice.direction_collapsed)

    # 渲染
    rendered_image = slice.expand(predicted_color)

    return rendered_image


def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == "__main__":
    loaded_model = load_model('checkpoints/trained_model.pkl')
    test_collected_result = CollectedResult('../RadiosityCollecterOutput2')

    def render_func(slice):
        return model_render_radiosity_on_image(slice, loaded_model)
    
    test_collected_result.write_perdict_images(render_func)
from collected_result import CollectedResult
from network import train_model
import pickle

if __name__ == "__main__":
    collected_result = CollectedResult('../RadiosityCollecterOutput')
    print(collected_result.get_color_matrix().shape)

    # Train the model
    training_data = collected_result.prepare_training_data()
    trained_model = train_model(training_data["posw"], training_data["direction"], training_data["color"], epochs=2, batch_size=320)
    
    # Save the trained model to a file
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(trained_model, f)

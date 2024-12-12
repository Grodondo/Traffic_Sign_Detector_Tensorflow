import cv2
import numpy as np
import sys
import os
import tensorflow as tf
import datetime


def main():
    if len(sys.argv) not in [2, 3]:
        sys.exit(
            "Usage: python traffic_visualization.py [model.h5] or [model.h5 test.jpg]"
        )

    model_path = sys.argv[1]
    model = tf.keras.models.load_model(model_path)

    if len(sys.argv) == 2:
        show_model_plot(model)
    elif len(sys.argv) == 3:
        img = sys.argv[2]
        img = cv2.imread(img)
        predict_individual(img, model)

    print("Model visualized.")


def show_model_plot(model):
    model_plot_dir = "logs/model_images/" + datetime.datetime.now().strftime(
        "model_plot%Y%m%d-%H%M%S.png"
    )
    # model_plot_path = os.path.join("logs", "model_images", "model_plot.png")

    tf.keras.utils.plot_model(
        model, to_file=model_plot_dir, show_shapes=True, show_layer_names=True
    )


def predict_individual(img, model):
    # Load the image
    img = cv2.resize(img, (30, 30))
    img = np.array(img)
    img = img.reshape(1, 30, 30, 3)

    # Predict the image
    prediction = model.predict(img)
    # print(prediction)
    print("The predicted model is: ", np.argmax(prediction))


if __name__ == "__main__":
    main()

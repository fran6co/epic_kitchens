import json
import pathlib
from argparse import ArgumentParser
from collections import deque

import cv2
import numpy as np
import onnxruntime
import tqdm


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--video_path", default="", type=str)
    parser.add_argument("--onnx", default="", type=str)
    parser.add_argument("--class_names_path", default="", type=str)
    args = parser.parse_args()

    class_names = json.load(open(args.class_names_path, "r"))

    get_label_name = lambda frame_label_id: next(
        (
            label_name
            for label_name, label_id in class_names.items()
            if label_id == frame_label_id
        )
    )

    inference = onnxruntime.InferenceSession(
        args.onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    height, width = inference.get_inputs()[0].shape[2:4]
    num_frames = inference.get_inputs()[0].shape[1]

    images = list(sorted(pathlib.Path(args.video_path).glob("*.jpg")))

    frames = deque(maxlen=num_frames)
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    mean_scores = np.array([0.0] * len(class_names))
    rolling_mean_scores = np.array([0.0] * len(class_names))
    output_path = pathlib.Path(args.video_path)
    output_path = output_path.parent / f"{output_path.name}.mp4"
    writer = None
    for i, image_path in enumerate(tqdm.tqdm(images)):
        frame = cv2.imread(str(image_path))
        if writer is None:
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 60, (frame.shape[1], frame.shape[0])
            )

        # Resize the image to fit the expected input, this could be a crop as well
        resized = cv2.resize(frame, (width, height))

        # Normalize image
        normalized = resized.astype(np.float64)
        normalized = normalized / 255.0
        normalized = normalized - mean
        normalized = normalized / std

        frames.append(normalized)
        inputs = np.array(list(frames))

        # Sample frames for num_frames specified
        index = np.linspace(0, inputs.shape[0] - 1, num_frames).astype(np.int64)
        inputs = np.take(inputs, index, axis=0)

        inputs = np.expand_dims(inputs, axis=0)

        scores = inference.run(["scores"], {"images": inputs.astype(np.float32)})[0][0]

        mean_scores += scores

        rolling_mean_scores = (rolling_mean_scores * min(i, 99) + scores) / min(
            i + 1, 100
        )
        video_label_id = np.argmax(rolling_mean_scores, axis=0)
        video_score = rolling_mean_scores[video_label_id]
        video_label_name = get_label_name(video_label_id)

        cv2.putText(
            frame,
            f"{video_label_name} ({video_score:.2f})",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            1,
            2,
        )
        writer.write(frame)
        cv2.imshow("video", frame)
        cv2.waitKey(1)

    mean_scores /= len(images)
    video_label_id = np.argmax(mean_scores, axis=0)
    video_score = mean_scores[video_label_id]
    video_label_name = get_label_name(video_label_id)

    if writer:
        writer.release()
    print(f"{args.video_path}: {video_label_name} ({video_score:.2f})")


if __name__ == "__main__":
    cli_main()

import argparse

from lib.capture import capture_faces
from lib.evaluator import evaluate
from lib.predictor import predict_on_image, predict_on_video
from lib.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Face Recognition System")
    subparsers = parser.add_subparsers(dest="command", help="Commands to execute")

    # Capture command
    capture_parser = subparsers.add_parser(
        "capture", help="Capture faces for the dataset"
    )
    capture_parser.add_argument(
        "--name", required=True, help="Name of the person to capture"
    )
    capture_parser.add_argument(
        "--source", default=0, help="Video source (0 for webcam or path to video)"
    )

    # Train command
    subparsers.add_parser("train", help="Train the face recognition model")

    # Evaluate command
    subparsers.add_parser("evaluate", help="Evaluate the trained model")

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Predict faces in an image or video"
    )
    predict_parser.add_argument("--image", help="Path to the image file")
    predict_parser.add_argument("--video", help="Path to the video file")

    args = parser.parse_args()

    if args.command == "capture":
        capture_faces(args.name, source=args.source)
    elif args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "predict":
        if args.image:
            predict_on_image(args.image)
        elif args.video:
            predict_on_video(args.video)
        else:
            print("Please specify either --image or --video for prediction.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

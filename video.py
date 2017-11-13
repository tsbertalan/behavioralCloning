import os
from moviepy.editor import ImageSequenceClip
import argparse


def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    parser.add_argument(
        '--filename',
        type=str,
        default='output_video.mp4',
        help='File name within the output folder.'
    )
    args = parser.parse_args()

    filename = args.filename
    if not filename.endswith('.mp4'): filename += '.mp4'
    video_file = os.path.join(args.image_folder, filename)
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    clip = ImageSequenceClip(args.image_folder, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()

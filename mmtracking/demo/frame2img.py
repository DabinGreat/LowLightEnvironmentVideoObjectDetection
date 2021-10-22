from argparse import ArgumentParser

import cv2

from demo.add_noise_for_frame import add_noise_for_frames


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='/home/dabingreat666/code/detect/mmtracking/demo/videos/vid_fox.mp4',
                        help='input video file')
    parser.add_argument('--output', type=str,
                        default='/home/dabingreat666/code/detect/mmtracking/demo/videos/fox_noise_n05-l05',
                        help='output video file (mp4 format)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    frame_id = 0
    while(cap.isOpened()):
        flag, frame = cap.read()
        if not flag:
            break

        noise, clean = add_noise_for_frames(frame, constant=[0.5, 0.5])

        if args.output is not None:
            cv2.imwrite(f'{args.output}/%06d.jpg'% frame_id, noise)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(f"has saved {frame_id} images!")
        frame_id += 1
    cap.release()


if __name__ == '__main__':
    main()
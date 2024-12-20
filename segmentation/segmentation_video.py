from utils.segmentation import *

save_dir = ['../dataset/VideoPradictor']

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(nargs='+', dest='video_path', help='Path to the video file')
    parser.add_argument('--output', '-o', nargs=1, default=save_dir)

    video_path_list = parser.parse_args().video_path
    output_path = parser.parse_args().output
    
    return video_path_list, output_path

def main(video_path_list, output_path):
    for video_path in video_path_list:
        segment_video(video_path, output_path)
    print('---------------------------------')
    print('All Segmentation done.')
    print('---------------------------------')        
        
if __name__ == "__main__":
    video_path_list, output_path = get_arguments()
    main(video_path_list, output_path[0])
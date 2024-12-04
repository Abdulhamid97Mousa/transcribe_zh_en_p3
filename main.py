import argparse
import os
import sys
import glob
import pysrt
from pydub import AudioSegment
import logging
import asyncio
import tempfile
import aiohttp  # For asynchronous HTTP requests
from tqdm import tqdm  # For progress bars
import subprocess
from dotenv import load_dotenv  # For loading .env files
import openai

# Constants
OPENAI_API_URL = "https://api.openai.com/v1/audio/speech"

def test_openai_connectivity():
    """
    Test connectivity to OpenAI's API by making a test request.

    Raises:
        Exception: If the connectivity test fails.
    """
    openai.api_key = os.getenv('OPENAI_API_KEY')
    try:
        response = openai.Audio.create(
            model="tts-1-hd",
            input="Testing connectivity.",
            voice="nova",
            response_format="wav",
            speed=1.0
        )
        with open("test_output.wav", "wb") as f:
            f.write(response['data'])
        logging.info("Connectivity Test Passed. Audio generated successfully.")
    except Exception as e:
        logging.error(f"Connectivity Test Failed: {e}")
        raise

def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more detailed logs
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("synthesizer.log")  # Logs saved to 'synthesizer.log'
        ]
    )

def load_api_key(secrets_dir=None, api_token=None):
    """
    Load the OpenAI API key from a .env file located in the secrets directory
    or directly from the provided API token.

    Args:
        secrets_dir (str, optional): Directory containing the .env file.
        api_token (str, optional): Direct OpenAI API token.

    Returns:
        str: The OpenAI API key.

    Raises:
        SystemExit: If the API key cannot be found.
    """
    if api_token:
        logging.info("Using OpenAI API token provided via command-line argument.")
        return api_token
    elif secrets_dir:
        dotenv_path = os.path.join(secrets_dir, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)  # Load the .env file
            api_key = os.getenv('OPENAI_API_KEY')  # Fetch the OpenAI API key from the .env file
            if not api_key:
                logging.error("API key not found in the .env file.")
                sys.exit(1)
            logging.info(f"Loaded OpenAI API key from '{dotenv_path}'.")
            return api_key
        else:
            logging.error(f"No .env file found at '{dotenv_path}'.")
            sys.exit(1)
    else:
        # Attempt to load from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            logging.info("Using OpenAI API key from environment variable 'OPENAI_API_KEY'.")
            return api_key
        else:
            logging.error("OpenAI API key not provided. Use '--secrets-dir' or '--openai-api-token' to provide the API key.")
            sys.exit(1)

def find_files(video_dir):
    """
    Locate 'vocals_processed.wav' and '*_translated.srt' files in the video directory.

    Args:
        video_dir (str): Path to the video directory.

    Returns:
        tuple: Paths to vocals and SRT files.
    """
    # Define the expected paths and patterns
    vocals_filename = 'vocals_processed.wav'
    srt_pattern = '*_translated.srt'
    no_audio_video_pattern = '*_no_audio.mp4'
    
    vocals_path = os.path.join(video_dir, vocals_filename)
    srt_search_pattern = os.path.join(video_dir, srt_pattern)
    no_audio_video_search_pattern = os.path.join(video_dir, no_audio_video_pattern)
    
    # Logging the search patterns
    logging.info(f"Searching for vocals at: {vocals_path}")
    logging.info(f"Searching for SRT files with pattern: {srt_search_pattern}")
    logging.info(f"Searching for video with no_audio: {no_audio_video_search_pattern}")
    
    # Use glob to find SRT files matching the pattern
    srt_files = glob.glob(srt_search_pattern)
    video_no_audio_files = glob.glob(no_audio_video_search_pattern)
    
    # Log the found SRT files
    if srt_files:
        logging.info(f"Found SRT files: {srt_files}")
    else:
        logging.warning(f"No SRT files matching pattern '{srt_pattern}' found in '{video_dir}'.")
        
    # Log the found video with no audio
    if video_no_audio_files:
        logging.info(f"Found no_audio video: {video_no_audio_files}")
    else:
        logging.warning(f"Was not able to find video with no_audio '{video_no_audio_files}' found in '{video_dir}'")
    
    
    # Check if vocals_processed.wav exists
    if not os.path.isfile(vocals_path):
        logging.warning(f"'{vocals_filename}' not found in '{video_dir}'. Skipping this directory.")
        return 
    
    # If no SRT files found, return vocals_path and None
    if not srt_files:
        logging.warning(f"No '_translated.srt' file found in '{video_dir}'. Skipping SRT processing for this directory.")
        return 
    
    # If no video_no_audio
    if not video_no_audio_files:
        logging.warning(f"No '_no_audio.mp4' file found in '{video_dir}'. Skipping no_audio.mp4 processing for this directory.")
        return 
    
    # Assuming only one SRT file per directory; if multiple, you might need to handle accordingly
    srt_path = srt_files[0]
    logging.info(f"Selected SRT file: {srt_path}")
    
    # Assuming only one No_audio per directory.
    no_audio = video_no_audio_files[0]
    logging.info(f"Selected no_audio file: {no_audio}")
    
    return vocals_path, srt_path, no_audio

def get_video_directories(input_dir):
    """
    Retrieve all subdirectories within the input directory.

    Args:
        input_dir (str): Path to the input directory.

    Returns:
        list: List of subdirectory paths.
    """
    subdirs = [
        os.path.join(input_dir, d) for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ]
    logging.info(f"Found {len(subdirs)} video directories in '{input_dir}'.")
    return subdirs

def parse_srt(srt_path):
    """
    Parse the SRT file and extract subtitle entries.
    
    Args:
        srt_path (str): Path to the SRT file.
    
    Returns:
        list: List of tuples containing (start_time_ms, end_time_ms, text).
    """
    try:
        subtitles = pysrt.open(srt_path)
    except Exception as e:
        logging.error(f"Failed to read SRT file '{srt_path}': {e}")
        return []
    
    subtitle_entries = []
    for sub in subtitles:
        start_time = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
        end_time = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds
        text = sub.text.replace('\n', ' ').strip()
        if text:
            subtitle_entries.append((start_time, end_time, text))
    return subtitle_entries

async def call_openai_tts(session, text, model, voice, response_format, speed, retries=3):
    """
    Call OpenAI's TTS API to generate audio for the given text.

    Args:
        session (aiohttp.ClientSession): The HTTP session.
        text (str): The text to synthesize.
        model (str): The TTS model to use.
        voice (str): The voice to use.
        response_format (str): The desired audio format.
        speed (float): The speed of the audio.
        retries (int): Number of retries in case of failure.

    Returns:
        bytes: The audio content in bytes.

    Raises:
        Exception: If all retry attempts fail.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "speed": speed
    }

    proxy = "http://127.0.0.1:7890"
    connector = aiohttp.TCPConnector(ssl=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        for attempt in range(1, retries + 1):
            try:
                async with session.post(OPENAI_API_URL, json=payload, headers=headers, proxy=proxy) as resp:
                    if resp.status == 200:
                        audio_content = await resp.read()
                        return audio_content
                    else:
                        error_text = await resp.text()
                        logging.warning(f"Attempt {attempt}: OpenAI API returned status {resp.status}. Response: {error_text}")
            except Exception as e:
                logging.warning(f"Attempt {attempt}: Failed to call OpenAI API: {e}")
            
            if attempt < retries:
                wait_time = 2 ** attempt  # Exponential backoff
                logging.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
    
    raise Exception("OpenAI API failed after multiple retries.")

async def synthesize_speech(text, model, voice, response_format, speed, session, retries=3):
    """
    Synthesize speech using OpenAI's TTS API.

    Args:
        text (str): The text to synthesize.
        model (str): The TTS model to use.
        voice (str): The voice to use.
        response_format (str): The desired audio format.
        speed (float): The speed of the audio.
        session (aiohttp.ClientSession): The HTTP session.
        retries (int): Number of retries in case of failure.

    Returns:
        AudioSegment: Synthesized audio.
    """
    try:
        audio_bytes = await call_openai_tts(session, text, model, voice, response_format, speed, retries)
        with tempfile.NamedTemporaryFile(suffix=f'.{response_format}') as tmpfile:
            tmpfile.write(audio_bytes)
            tmpfile.flush()
            audio_segment = AudioSegment.from_file(tmpfile.name, format=response_format)
            # Set sample rate to 44100 Hz for better quality and mono channel
            audio_segment = audio_segment.set_frame_rate(44100).set_channels(1)
            return audio_segment
    except Exception as e:
        logging.error(f"Error synthesizing text: {e}")
        raise

async def synthesize_and_place(subtitle, tts_config, session):
    """
    Synthesize speech for a subtitle and place it at the correct timestamp.
    
    Args:
        subtitle (tuple): A tuple containing (start_time_ms, end_time_ms, text).
        tts_config (dict): TTS configuration containing model, voice, response_format, speed.
        session (aiohttp.ClientSession): The HTTP session.
    
    Returns:
        tuple: (start_time_ms, synthesized_audio_segment)
    """
    start_time, end_time, text = subtitle
    duration = end_time - start_time
    try:
        synthesized_audio = await synthesize_speech(
            text,
            model=tts_config['model'],
            voice=tts_config['voice'],
            response_format=tts_config['response_format'],
            speed=tts_config['speed'],
            session=session
        )
        synthesized_audio = synthesized_audio[:duration]  # Trim to subtitle duration
        if len(synthesized_audio) < duration:
            synthesized_audio += AudioSegment.silent(duration=duration - len(synthesized_audio))
        return (start_time, synthesized_audio)
    except Exception as e:
        logging.error(f"Failed to synthesize segment '{text}': {e}")
        raise

async def generate_aligned_speech(tts_config, srt_path, total_duration_ms):
    """
    Generate an AudioSegment with synthesized speech aligned to subtitles.
    
    Args:
        tts_config (dict): TTS configuration containing model, voice, response_format, speed.
        srt_path (str): Path to the SRT file.
        total_duration_ms (int): Total duration of the original audio in milliseconds.
    
    Returns:
        AudioSegment: Aligned synthesized speech audio.
    """
    subtitles = parse_srt(srt_path)
    if not subtitles:
        logging.warning("No subtitles found. Returning silent audio.")
        return AudioSegment.silent(duration=total_duration_ms)
    
    # Create a silent AudioSegment for the entire duration
    aligned_audio = AudioSegment.silent(duration=total_duration_ms)
    
    # Create an aiohttp session
    async with aiohttp.ClientSession() as session:
        # Create tasks for concurrent synthesis
        tasks = [synthesize_and_place(sub, tts_config, session) for sub in subtitles]
        
        # Gather synthesized speech segments with progress bar
        synthesized_segments = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Synthesizing speech"):
            try:
                segment = await coro
                synthesized_segments.append(segment)
            except Exception:
                logging.error("Aborting due to synthesis failure.")
                sys.exit(1)
    
    # Overlay each synthesized speech segment at the correct timestamp
    for start_time, segment in tqdm(synthesized_segments, desc="Placing synthesized speech"):
        aligned_audio = aligned_audio.overlay(segment, position=start_time)
    
    return aligned_audio

def save_synthesized_audio(synthesized_audio, output_path, overwrite=False):
    """
    Save the synthesized audio to a WAV file.

    Args:
        synthesized_audio (AudioSegment): Synthesized audio.
        output_path (str): Path to save the audio file.
        overwrite (bool): Whether to overwrite the file if it exists.

    Raises:
        SystemExit: If the file exists and overwrite is False.
    """
    if os.path.exists(output_path) and not overwrite:
        logging.warning(f"File '{output_path}' already exists. Use '--overwrite' to overwrite.")
        return

    try:
        logging.info(f"Exporting synthesized audio to '{output_path}'.")
        synthesized_audio.export(output_path, format='wav', bitrate="192k")
        logging.info(f"Synthesized audio successfully saved to '{output_path}'.")
    except Exception as e:
        logging.error(f"Failed to save synthesized audio to '{output_path}': {e}")
              
import os
import subprocess
import sys
import logging

def combine_video_audio_ffmpeg(no_audio_video_path, syn_audio_path, combined_path, overwrite=False):
    """
    Combine a video file without audio with a synthesized audio file using ffmpeg.

    Args:
        no_audio_video_path (str): Path to the video file without audio (e.g., '*_no_audio.mp4').
        syn_audio_path (str): Path to the synthesized audio file (e.g., '*_syn_speech_aligned.wav').
        combined_path (str): Path to save the combined video file.
        overwrite (bool): Whether to overwrite the combined file if it exists.

    Raises:
        SystemExit: If ffmpeg fails or the output file exists and overwrite is False.
    """
    if os.path.exists(combined_path) and not overwrite:
        logging.warning(f"Combined file '{combined_path}' already exists. Use '--overwrite' to overwrite.")
        return

    command = [
        'ffmpeg',
        '-y' if overwrite else '-n',  # Overwrite if overwrite is True, else do not overwrite
        '-i', no_audio_video_path,    # Input video without audio
        '-i', syn_audio_path,         # Input synthesized audio
        '-c:v', 'copy',                # Copy the video codec without re-encoding
        '-c:a', 'aac',                 # Encode audio using AAC codec
        '-map', '0:v:0',               # Map the first video stream from the first input
        '-map', '1:a:0',               # Map the first audio stream from the second input
        combined_path                   # Output file path
    ]

    try:
        logging.info(
            f"Combining '{no_audio_video_path}' and '{syn_audio_path}' into '{combined_path}' using ffmpeg."
        )
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info(f"Combined video successfully saved to '{combined_path}'.")
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg failed to combine video and audio: {e.stderr.decode()}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("ffmpeg is not installed or not found in PATH.")
        sys.exit(1)

def add_subtitles_ffmpeg(video_input_path, srt_path, final_output_path, overwrite=False):
    """
    Add subtitles to a video file using ffmpeg.

    Args:
        video_input_path (str): Path to the input video file (e.g., 'combined.mp4').
        srt_path (str): Path to the SRT subtitle file.
        final_output_path (str): Path to save the final video with subtitles.
        overwrite (bool): Whether to overwrite the final output file if it exists.

    Raises:
        SystemExit: If ffmpeg fails or the output file exists and overwrite is False.
    """
    if os.path.exists(final_output_path) and not overwrite:
        logging.warning(f"Final video file '{final_output_path}' already exists. Use '--overwrite' to overwrite.")
        return

    # Ensure the SRT path is correctly escaped for ffmpeg
    srt_path_escaped = srt_path.replace("'", r"'\''")

    # Construct the subtitles filter with styling
    subtitles_filter = f"subtitles='{srt_path}':force_style='FontName=SimSun,FontSize=18'"

    command = [
        'ffmpeg',
        '-y' if overwrite else '-n',  # Overwrite if overwrite is True, else do not overwrite
        '-i', video_input_path,        # Input video
        '-vf', subtitles_filter,        # Video filter for subtitles
        '-c:a', 'copy',                 # Copy the audio stream without re-encoding
        final_output_path               # Output file path
    ]

    try:
        logging.info(
            f"Adding subtitles from '{srt_path}' to '{video_input_path}' and saving as '{final_output_path}' using ffmpeg."
        )
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info(f"Final video with subtitles successfully saved to '{final_output_path}'.")
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg failed to add subtitles: {e.stderr.decode()}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("ffmpeg is not installed or not found in PATH.")
        sys.exit(1)

async def process_video_dir_async(video_dir, tts_config, overwrite):
    """
    Asynchronously process a single video directory:
    1. Find necessary files.
    2. Generate synthesized speech from SRT.
    3. Combine synthesized audio with video.
    4. Add subtitles to the combined video.

    Args:
        video_dir (str): Path to the video directory.
        tts_config (dict): Configuration for text-to-speech synthesis.
        overwrite (bool): Whether to overwrite existing files.

    Raises:
        SystemExit: If critical steps fail.
    """
    logging.info(f"\nProcessing directory: {video_dir}")
    
    # Step 1: Find necessary files
    result = find_files(video_dir)
    if result is None:
        logging.warning(f"Required files not found in '{video_dir}'. Skipping directory.")
        return

    vocals_path, srt_path, video_no_audio_path = result

    if not srt_path:
        logging.warning(f"No SRT file to process in '{video_dir}'. Skipping speech synthesis.")
        return

    # Step 2: Determine the total duration of the original audio
    try:
        original_audio = AudioSegment.from_wav(vocals_path)
        total_duration_ms = len(original_audio)
        logging.info(f"Original audio duration: {total_duration_ms} ms.")
    except Exception as e:
        logging.error(f"Failed to load original audio '{vocals_path}': {e}")
        return

    # Step 3: Generate aligned synthesized speech from SRT
    logging.info("Starting aligned speech synthesis from SRT.")
    try:
        aligned_synthesized_audio = await generate_aligned_speech(tts_config, srt_path, total_duration_ms)
    except Exception as e:
        logging.error(f"Failed to generate aligned synthesized audio for '{video_dir}': {e}")
        return

    if aligned_synthesized_audio is None:
        logging.error(f"Failed to generate aligned synthesized audio for '{video_dir}'. Skipping.")
        return

    # Define output paths
    video_dir_name = os.path.basename(video_dir)
    syn_speech_filename = f"{video_dir_name}_syn_speech_aligned.wav"
    syn_speech_path = os.path.join(video_dir, syn_speech_filename)
    combined_filename = f"combined_{video_dir_name}.mp4"
    combined_output_path = os.path.join(video_dir, combined_filename)
    final_filename = f"final_{video_dir_name}.mp4"
    final_output_path = os.path.join(video_dir, final_filename)

    # Step 4: Save synthesized audio
    save_synthesized_audio(aligned_synthesized_audio, syn_speech_path, overwrite=overwrite)

    # Step 5: Combine synthesized audio with no-audio video using ffmpeg
    combine_video_audio_ffmpeg(
        no_audio_video_path=video_no_audio_path,
        syn_audio_path=syn_speech_path,
        combined_path=combined_output_path,
        overwrite=overwrite
    )

    # Step 6: Add subtitles to the combined video
    add_subtitles_ffmpeg(
        video_input_path=combined_output_path,
        srt_path=srt_path,
        final_output_path=final_output_path,
        overwrite=overwrite
    )

    logging.info(f"Finished processing directory: {video_dir}")


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Process video directories and generate synthesized speech using OpenAI TTS.'
    )
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Input directory containing all video subdirectories.'
    )
    parser.add_argument(
        '--voice',
        default='nova',  # Default voice if not specified
        choices=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
        help='Voice to use when generating the audio.'
    )
    parser.add_argument(
        '--response-format',
        default='wav',
        choices=['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'],
        help='The format of the generated audio.'
    )
    parser.add_argument(
        '--speed-of-audio',
        type=float,
        default=1.0,
        help='The speed of the generated audio. Value from 0.25 to 4.0.'
    )
    parser.add_argument(
        '--secrets-dir',
        help='Directory containing .env file with API key.'
    )
    parser.add_argument(
        '--openai-api-token',
        help='OpenAI API key (if not using secrets-dir).'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files if they exist.'
    )
    return parser.parse_args()



async def main_async(video_dirs, tts_config, overwrite):
    """
    Asynchronous main function to process all video directories.

    Args:
        video_dirs (list): List of video directory paths.
        tts_config (dict): TTS configuration containing model, voice, response_format, speed.
        overwrite (bool): Whether to overwrite existing files.
    """
    tasks = [process_video_dir_async(video_dir, tts_config, overwrite) for video_dir in video_dirs]
    await asyncio.gather(*tasks)

def main():
    """Main function to execute the CLI tool."""
    setup_logging()
    args = parse_arguments()
    
    # Load OpenAI API key using the provided flags
    global OPENAI_API_KEY
    OPENAI_API_KEY = load_api_key(secrets_dir=args.secrets_dir, api_token=args.openai_api_token)
    
    input_dir = args.input_dir
    voice = args.voice
    response_format = args.response_format
    speed = args.speed_of_audio
    overwrite = args.overwrite
    
    # Validate speed parameter
    if not (0.25 <= speed <= 4.0):
        logging.error("The '--speed-of-audio' parameter must be between 0.25 and 4.0.")
        sys.exit(1)
    
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory '{input_dir}' does not exist or is not a directory.")
        sys.exit(1)
    
    video_dirs = get_video_directories(input_dir)
    if not video_dirs:
        logging.warning(f"No subdirectories found in '{input_dir}'. Exiting.")
        sys.exit(0)
    
    tts_config = {
        'model': 'tts-1-hd',
        'voice': voice,
        'response_format': response_format,
        'speed': speed
    }
    
    # Check if ffmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        logging.error("ffmpeg is installed but returned a non-zero exit status.")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("ffmpeg is not installed or not found in PATH. Please install ffmpeg to proceed.")
        sys.exit(1)
    
    # Create and run the event loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main_async(video_dirs, tts_config, overwrite))
    finally:
        loop.close()
    
    logging.info("Processing completed.")

if __name__ == "__main__":
    main()

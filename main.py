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
import re
from dotenv import load_dotenv  # For loading .env files

# Constants
OPENAI_API_URL = "https://api.openai.com/v1/audio/speech"

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
    Locate '*_vocals_processed.wav', '*_non_vocals_processed.wav', '*_translated.srt', 
    and '*_no_audio.mp4' files in the given video directory.

    Args:
        video_dir (str): Path to the video directory.

    Returns:
        tuple or None: 
            If all critical files are found, returns a tuple:
            (vocals_processed_path, non_vocals_processed_path, srt_path, no_audio_video_path)
            Otherwise, returns None.
    """
    # Define the expected file name patterns
    vocals_processed_filename = '*_vocals_processed.wav'
    non_vocals_filename = '*_non_vocals.wav'
    srt_pattern = '*_translated*.srt'
    no_audio_video_pattern = '*_no_audio.mp4'
    
    vocals_processed_search_pattern = os.path.join(video_dir, vocals_processed_filename)
    non_vocals_search_pattern = os.path.join(video_dir, non_vocals_filename)
    srt_search_pattern = os.path.join(video_dir, srt_pattern)
    no_audio_video_search_pattern = os.path.join(video_dir, no_audio_video_pattern)
    
    # Logging the search patterns
    logging.info(f"Searching for vocals at: {vocals_processed_search_pattern}")
    logging.info(f"Searching for non_vocals at: {non_vocals_search_pattern}")
    logging.info(f"Searching for SRT files with pattern: {srt_search_pattern}")
    logging.info(f"Searching for no_audio video: {no_audio_video_search_pattern}")
    
    # Use glob to find files
    vocals_processed_files = glob.glob(vocals_processed_search_pattern)
    non_vocals_files = glob.glob(non_vocals_search_pattern)
    srt_files = glob.glob(srt_search_pattern)
    video_no_audio_files = glob.glob(no_audio_video_search_pattern)
    
    # Logging the found files
    if vocals_processed_files:
        logging.info(f"Found vocals_processed file: {vocals_processed_files}")
    else:
        logging.warning(f"'{vocals_processed_filename}' not found in '{video_dir}'.")

    if non_vocals_files:
        logging.info(f"Found non_vocals_processed file: {non_vocals_files}")
    else:
        logging.warning(f"'{non_vocals_filename}' not found in '{video_dir}'.")

    if srt_files:
        logging.info(f"Found SRT files: {srt_files}")
    else:
        logging.warning(f"No SRT files matching pattern '{srt_pattern}' found in '{video_dir}'.")

    if video_no_audio_files:
        logging.info(f"Found no_audio video: {video_no_audio_files}")
    else:
        logging.warning(f"No '_no_audio.mp4' file found in '{video_dir}'.")

    # Check if all required files are found
    if not vocals_processed_files:
        logging.warning("Missing vocals_processed file. Cannot proceed.")
        return None
    
    if not non_vocals_files:
        logging.warning("Missing non_vocals_processed file. Cannot proceed.")
        return None

    if not srt_files:
        logging.warning("Missing translated SRT file. Cannot proceed.")
        return None

    if not video_no_audio_files:
        logging.warning("Missing no_audio MP4 file. Cannot proceed.")
        return None

    # Select the first matching files (assuming only one per category)
    vocals_path = vocals_processed_files[0]
    non_vocals_path = non_vocals_files[0]
    srt_path = srt_files[0]
    no_audio_path = video_no_audio_files[0]
    
    logging.info(f"Selected vocals_processed file: {vocals_path}")
    logging.info(f"Selected non_vocals_processed file: {non_vocals_path}")
    logging.info(f"Selected SRT file: {srt_path}")
    logging.info(f"Selected no_audio MP4 file: {no_audio_path}")

    return vocals_path, non_vocals_path, srt_path, no_audio_path

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

async def call_openai_tts(session, text, model, voice, response_format, speed, proxy=None, retries=5):
    """
    Call OpenAI's TTS API to generate audio for the given text.

    Args:
        session (aiohttp.ClientSession): The HTTP session.
        text (str): The text to synthesize.
        model (str): The TTS model to use.
        voice (str): The voice to use.
        response_format (str): The desired audio format.
        speed (float): The speed of the audio.
        proxy (str, optional): Proxy URL.
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

    for attempt in range(1, retries + 1):
        try:
            async with session.post(OPENAI_API_URL, json=payload, headers=headers, proxy=proxy) as resp:
                if resp.status == 200:
                    audio_content = await resp.read()
                    logging.info(f"Attempt {attempt}: Successfully received audio.")
                    return audio_content
                else:
                    error_text = await resp.text()
                    logging.warning(f"Attempt {attempt}: OpenAI API returned status {resp.status}. Response: {error_text}")
        except Exception as e:
            logging.warning(f"Attempt {attempt}: Failed to call OpenAI API: {e}")

        if attempt < retries:
            wait_time = min(2 ** attempt, 60)  # Cap the wait time at 60 seconds
            logging.info(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

    raise Exception("OpenAI API failed after multiple retries.")

async def synthesize_speech(text, model, voice, response_format, speed, session, retries=5, proxy=None):
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
        proxy (str, optional): Proxy URL.

    Returns:
        AudioSegment: Synthesized audio.
    """
    try:
        audio_bytes = await call_openai_tts(
            session=session,
            text=text,
            model=model,
            voice=voice,
            response_format=response_format,
            speed=speed,
            proxy=proxy,
            retries=retries
        )
        with tempfile.NamedTemporaryFile(suffix=f'.{response_format}', delete=False) as tmpfile:
            tmpfile.write(audio_bytes)
            tmpfile.flush()
            audio_segment = AudioSegment.from_file(tmpfile.name, format=response_format)
            # Set sample rate to 44000 Hz for better quality and mono channel
            audio_segment = audio_segment.set_frame_rate(44000).set_channels(1)
        # Remove the temporary file
        os.unlink(tmpfile.name)
        return audio_segment
    except Exception as e:
        logging.error(f"Error synthesizing text: {e}")
        raise

async def synthesize_and_place(subtitle, tts_config, session, proxy=None):
    """
    Synthesize speech for a subtitle and place it at the correct timestamp.
    
    Args:
        subtitle (tuple): A tuple containing (start_time_ms, end_time_ms, text).
        tts_config (dict): TTS configuration containing model, voice, response_format, speed.
        session (aiohttp.ClientSession): The HTTP session.
        proxy (str, optional): Proxy URL.

    Returns:
        tuple: (start_time_ms, synthesized_audio_segment)
    """
    start_time, end_time, text = subtitle
    duration = end_time - start_time
    try:
        synthesized_audio = await synthesize_speech(
            text=text,
            model=tts_config['model'],
            voice=tts_config['voice'],
            response_format=tts_config['response_format'],
            speed=tts_config['speed'],
            session=session,
            proxy=proxy
        )
        synthesized_audio = synthesized_audio[:duration]  # Trim to subtitle duration
        if len(synthesized_audio) < duration:
            synthesized_audio += AudioSegment.silent(duration=duration - len(synthesized_audio))
        return (start_time, synthesized_audio)
    except Exception as e:
        logging.error(f"Failed to synthesize segment '{text}': {e}")
        raise

async def generate_aligned_speech(tts_config, srt_path, total_duration_ms, session, proxy=None):
    """
    Generate an AudioSegment with synthesized speech aligned to subtitles.
    
    Args:
        tts_config (dict): TTS configuration containing model, voice, response_format, speed.
        srt_path (str): Path to the SRT file.
        total_duration_ms (int): Total duration of the original audio in milliseconds.
        session (aiohttp.ClientSession): The HTTP session.
        proxy (str, optional): Proxy URL.

    Returns:
        AudioSegment: Aligned synthesized speech audio.
    """
    subtitles = parse_srt(srt_path)
    if not subtitles:
        logging.warning("No subtitles found. Returning silent audio.")
        return AudioSegment.silent(duration=total_duration_ms)
    
    # Create a silent AudioSegment for the entire duration
    aligned_audio = AudioSegment.silent(duration=total_duration_ms)
    
    # Create tasks for concurrent synthesis
    tasks = [
        synthesize_and_place(sub, tts_config, session, proxy=proxy)
        for sub in subtitles
    ]
    
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
        synthesized_audio.export(output_path, format='wav')  # WAV doesn't require bitrate
        logging.info(f"Synthesized audio successfully saved to '{output_path}'.")
    except Exception as e:
        logging.error(f"Failed to save synthesized audio to '{output_path}': {e}")
        sys.exit(1)
              

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

def combine_non_vocals_audio_ffmpeg(combined_video_path, non_vocals_path, output_path, overwrite=False):
    """
    Combine a video (already containing synthesized audio) with non-vocals audio track using ffmpeg.

    Args:
        combined_video_path (str): Path to the video file that already has synthesized audio combined.
        non_vocals_path (str): Path to the non-vocals processed audio.
        output_path (str): Path to save the new combined video file.
        overwrite (bool): Whether to overwrite the output file if it exists.

    Raises:
        SystemExit: If ffmpeg fails or the output file exists and overwrite is False.
    """
    if os.path.exists(output_path) and not overwrite:
        logging.warning(f"Combined file '{output_path}' already exists. Use '--overwrite' to overwrite.")
        return

    # FFmpeg command to merge the video (with one audio) and the non-vocals audio into a single track.
    # The logic here is similar to step 6, but now we are adding a second audio track.
    # If the intention is to *mix* the two audio tracks, we would use a filter_complex to mix them.
    # For simplicity, let's assume we want to merge (mix) both audio tracks into one combined audio:
    command = [
        'ffmpeg',
        '-y' if overwrite else '-n',
        '-i', combined_video_path,   # Input combined video with synthesized audio
        '-i', non_vocals_path,       # Input non-vocals audio
        '-filter_complex', 'amix=inputs=2:duration=longest:dropout_transition=2', 
        '-c:v', 'copy',             # Copy video without re-encoding
        '-c:a', 'aac',              # Encode the mixed audio to AAC
        output_path
    ]

    try:
        logging.info(f"Combining '{combined_video_path}' and '{non_vocals_path}' into '{output_path}' using ffmpeg.")
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info(f"Video with merged audio saved to '{output_path}'.")
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg failed to combine video and non-vocals: {e.stderr.decode()}")
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
    subtitles_filter = f"subtitles='{srt_path_escaped}':force_style='FontName=SimSun,FontSize=18'"

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


async def process_video_dir_async(video_dir, tts_config, overwrite, proxy_host_ip=None, proxy_host_port=None):
    """
    Asynchronously process a single video directory:
    Step 1: Find necessary files.
    Step 2: Determine total duration of original vocals audio.
    Step 3: Set up aiohttp session (with or without proxy).
    Step 4: Generate aligned synthesized speech from SRT.
    Step 5: Save synthesized aligned audio.
    Step 6: Combine synthesized audio with no-audio video.
    Step 7: Combine the resulting video from step 6 with the non-vocals audio.
    Step 8: Add subtitles to the final combined video.

    Args:
        video_dir (str): Path to the video directory.
        tts_config (dict): Configuration for text-to-speech synthesis.
        overwrite (bool): Whether to overwrite existing files.
        proxy_host_ip (str): Proxy host IP address.
        proxy_host_port (str): Proxy host port number.

    Raises:
        SystemExit: If critical steps fail.
    """
    logging.info(f"\nProcessing directory: {video_dir}")

    # Step 1: Find necessary files
    result = find_files(video_dir)
    if result is None:
        logging.warning(f"Required files not found in '{video_dir}'. Skipping directory.")
        return
    
    vocals_path, non_vocals_path, srt_path, video_no_audio_path = result

    # Step 2: Determine the total duration of the original audio
    try:
        original_audio = AudioSegment.from_wav(vocals_path)
        total_duration_ms = len(original_audio)
        logging.info(f"Original audio duration: {total_duration_ms} ms.")
    except Exception as e:
        logging.error(f"Failed to load original audio '{vocals_path}': {e}")
        return

    # Step 3: Set up aiohttp session with or without proxy
    connector = None
    proxy = None
    if proxy_host_ip and proxy_host_port:
        proxy = f"http://{proxy_host_ip}:{proxy_host_port}"
        connector = aiohttp.TCPConnector(ssl=True)
        logging.info(f"Using proxy: {proxy}")
    else:
        logging.info("No proxy settings provided. Proceeding without proxy.")

    async with aiohttp.ClientSession(connector=connector) as session:
        # Step 4: Generate aligned synthesized speech from SRT
        logging.info("Starting aligned speech synthesis from SRT.")
        try:
            aligned_synthesized_audio = await generate_aligned_speech(
                tts_config=tts_config,
                srt_path=srt_path,
                total_duration_ms=total_duration_ms,
                session=session,
                proxy=proxy
            )
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
        final_nonvocals_combined_filename = f"combined_nonvocals_{video_dir_name}.mp4"
        final_nonvocals_combined_path = os.path.join(video_dir, final_nonvocals_combined_filename)
        final_filename = f"final_{video_dir_name}.mp4"
        final_output_path = os.path.join(video_dir, final_filename)

        # Step 5: Save synthesized audio
        save_synthesized_audio(aligned_synthesized_audio, syn_speech_path, overwrite=overwrite)
        
        # Step 6: Combine synthesized audio with no-audio video
        combine_video_audio_ffmpeg(
            no_audio_video_path=video_no_audio_path,
            syn_audio_path=syn_speech_path,
            combined_path=combined_output_path,
            overwrite=overwrite
        )

        # Only proceed to step 7 if step 6 succeeded (the combined_output_path must exist)
        if os.path.exists(combined_output_path):
            # Step 7: Combine non-vocals audio with the combined video from step 6
            combine_non_vocals_audio_ffmpeg(
                combined_video_path=combined_output_path,
                non_vocals_path=non_vocals_path,
                output_path=final_nonvocals_combined_path,
                overwrite=overwrite
            )
            
            # Step 8: Add subtitles to the final combined video (which now includes synthesized + non-vocals audio)
            add_subtitles_ffmpeg(
                video_input_path=final_nonvocals_combined_path,
                srt_path=srt_path,
                final_output_path=final_output_path,
                overwrite=overwrite
            )
        else:
            logging.error(f"Step 6 failed: '{combined_output_path}' not created, skipping step 7 and onwards.")

        logging.info(f"Finished processing directory: {video_dir}")


def parse_arguments():

    parser = argparse.ArgumentParser(description='Process video directories and generate synthesized speech using OpenAI TTS.')
    
    parser.add_argument('--input-dir', required=True, help='Input directory containing all video subdirectories.')
    parser.add_argument('--voice', default='nova', choices=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'], help='Default voice is nova. Voice to use when generating the audio.')
    parser.add_argument('--response-format', default='wav', choices=['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'], help='Default response foramt is wav .The format of the generated audio.')
    parser.add_argument('--speed-of-audio', type=float, default=1.0, help='Default speed is 1.0 which the same speed of the provided audio segment of the srt file. The speed of the generated audio. Value from 0.25 to 4.0.')
    parser.add_argument('--secrets-dir', help='Directory containing .env file with API key.')
    parser.add_argument('--openai-api-token', help='OpenAI API key (if not using secrets-dir).')
    
    # Optional flags for proxy settings
    parser.add_argument('--proxy-host-ip', help='Proxy host IP address. Please specify this if you are using VPN or proxy to connect to OpenAI API.') 
    parser.add_argument('--proxy-host-port', help='Proxy host port number. Please specify this if you are using VPN or proxy to connect to OpenAI API.')
    
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files if they exist.')
    
    return parser.parse_args()



async def main_async(video_dirs, tts_config, overwrite, proxy=None):
    """
    Asynchronous main function to process all video directories.

    Args:
        video_dirs (list): List of video directory paths.
        tts_config (dict): TTS configuration containing model, voice, response_format, speed.
        overwrite (bool): Whether to overwrite existing files.
        proxy (str, optional): Proxy URL.
    """
    
    # divide proxy to host and port remember proxy is with this format "http://{proxy_host_ip}:{proxy_host_port}"
    if proxy and proxy.startswith("http://") and len(proxy.split(":")) == 3:
        proxy_host_ip = proxy.split(":")[1].replace("//", "")
        proxy_host_port = proxy.split(":")[2]
    else:
        proxy_host_ip = None
        proxy_host_port = None
    
    tasks = [process_video_dir_async(video_dir, tts_config, overwrite, proxy_host_ip=proxy_host_ip, proxy_host_port=proxy_host_port) for video_dir in video_dirs]
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
    proxy_host_ip = args.proxy_host_ip
    proxy_host_port = args.proxy_host_port
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
    
    # Validate proxy settings
    if (proxy_host_ip and not proxy_host_port) or (proxy_host_port and not proxy_host_ip):
        logging.error("Both '--proxy-host-ip' and '--proxy-host-port' must be specified together.")
        sys.exit(1)
    
    # Validate the proxy host IP and port to be valid string. ip must be in the dot notation. port must be a number
    if proxy_host_ip:
        ip_pattern = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
        if not ip_pattern.match(proxy_host_ip):
            logging.error("Invalid proxy host IP format. It must be in dot notation (e.g., 192.168.1.1).")
            sys.exit(1)
    
    if proxy_host_port:
        if not proxy_host_port.isdigit() or not (0 <= int(proxy_host_port) <= 65535):
            logging.error("Invalid proxy host port. It must be a number between 0 and 65535.")
            sys.exit(1)
    
    
    # Create the proxy URL if provided
    if proxy_host_ip and proxy_host_port:
        proxy = f"http://{proxy_host_ip}:{proxy_host_port}"
        logging.info(f"Using proxy: {proxy}")
    else:
        proxy = None
        logging.info("No proxy settings provided. Proceeding without proxy.")
    
    # Define TTS configuration
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
    try:
        asyncio.run(main_async(
            video_dirs=video_dirs,
            tts_config=tts_config,
            overwrite=overwrite,
            proxy=proxy
        ))
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        logging.info("Processing completed.")

if __name__ == "__main__":
    main()

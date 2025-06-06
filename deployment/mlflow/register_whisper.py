import mlflow
import whisper
import logging
import os
import shutil
import subprocess # Needed to verify ffmpeg in load_context
import tempfile
import numpy as np
import pandas as pd # Explicitly import pandas for input/output schemas
import torch
import base64

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from mlflow.pyfunc import PythonModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Configuration ---
WHISPER_MODEL_SIZE = "base"

# --- Explicit Pip Requirements for Whisper Model Deployment ---
# Ensure these versions match what you have locally that works.
WHISPER_PIP_REQUIREMENTS = [
    "openai-whisper",
    "torch",
    "torchaudio",
    "ffmpeg-python", # Include this, as it's the Python binding
    "mlflow",
    "soundfile",
    "numpy",
    "pandas",
    # Add any specific versions if you pinned them after `pip show`
    # e.g., "openai-whisper==20231117", "torch==2.3.0", "torchaudio==2.3.0"
]

# --- Paths and Libraries from your `which ffmpeg` and `ldd` output ---
# These are the exact paths from your provided output
LOCAL_FFMPEG_PATH = "/usr/bin/ffmpeg"
LOCAL_FFMPEG_LIB_DIR = "/lib/x86_64-linux-gnu/" # Directory for most shared libraries

# List of specific shared libraries to bundle based on your `ldd` output.
# We're focusing on ffmpeg-specific and commonly problematic multimedia libraries.
# Fundamental system libs like libc.so.6, libm.so.6, libpthread.so.0 are generally
# present in most Linux base images and should NOT be bundled.
FFMPEG_REQUIRED_LIBS = [
    "libavdevice.so.58",
    "libavfilter.so.7",
    "libavformat.so.58",
    "libavcodec.so.58",
    "libpostproc.so.55",
    "libswresample.so.3",
    "libswscale.so.5",
    "libavutil.so.56",
    # Additional multimedia-related libs you found that are not common system libs
    "libraw1394.so.11",
    "libavc1394.so.0",
    "librom1394.so.0",
    "libiec61883.so.0",
    "libjack.so.0",
    "libdrm.so.2",
    "libopenal.so.1",
    "libxcb.so.1",
    "libxcb-shm.so.0",
    "libxcb-shape.so.0",
    "libxcb-xfixes.so.0",
    "libcdio_paranoia.so.2",
    "libcdio_cdda.so.2",
    "libdc1394.so.25",
    "libasound.so.2",
    "libcaca.so.0",
    "libGL.so.1",
    "libpulse.so.0",
    "libSDL2-2.0.so.0",
    "libsndio.so.7",
    "libXv.so.1",
    "libX11.so.6",
    "libXext.so.6",
    "libpocketsphinx.so.3",
    "libsphinxbase.so.3",
    "libbs2b.so.0",
    "liblilv-0.so.0",
    "librubberband.so.2",
    "libmysofa.so.1",
    "libflite_cmu_us_awb.so.1",
    "libflite_cmu_us_kal.so.1",
    "libflite_cmu_us_kal16.so.1",
    "libflite_cmu_us_rms.so.1",
    "libflite_cmu_us_slt.so.1",
    "libflite.so.1",
    "libfribidi.so.0",
    "libass.so.9",
    "libva.so.2",
    "libvidstab.so.1.1",
    "libzmq.so.5",
    "libzimg.so.2",
    # /usr/local/cuda/targets/x86_64-linux/lib/libOpenCL.so.1 is CUDA specific, bundle if needed,
    # but often included if base image is CUDA-enabled. If not, add and adjust path.
    # "libOpenCL.so.1", # If your base image doesn't have this, you'd need to bundle it from /usr/local/cuda...
    "libfontconfig.so.1",
    "libfreetype.so.6",
    "libmfx.so.1",
    "libxml2.so.2",
    "libbz2.so.1.0", # Often standard, but harmless to include if small
    "libgme.so.0",
    "libopenmpt.so.0",
    "libchromaprint.so.1",
    "libbluray.so.2",
    "libz.so.1", # Often standard, but harmless to include if small
    "libgnutls.so.30",
    "librabbitmq.so.4",
    "libsrt-gnutls.so.1.4",
    "libssh-gcrypt.so.4",
    "libvpx.so.7",
    "libwebpmux.so.3",
    "libwebp.so.7",
    "liblzma.so.5", # Often standard, but harmless to include if small
    "libdav1d.so.5",
    "librsvg-2.so.2",
    "libgobject-2.0.so.0",
    "libglib-2.0.so.0",
    "libcairo.so.2",
    "libzvbi.so.0",
    "libsnappy.so.1",
    "libaom.so.3",
    "libcodec2.so.1.0",
    "libgsm.so.1",
    "libmp3lame.so.0",
    "libopenjp2.so.7",
    "libopus.so.0",
    "libshine.so.3",
    "libspeex.so.1",
    "libtheoraenc.so.1",
    "libtheoradec.so.1",
    "libtwolame.so.0",
    "libvorbis.so.0",
    "libvorbisenc.so.2",
    "libx264.so.163",
    "libx265.so.199",
    "libxvidcore.so.4",
    "libsoxr.so.0",
    "libva-drm.so.2",
    "libva-x11.so.2",
    "libvdpau.so.1",
    "libdb-5.3.so",
    "libXau.so.6",
    "libXdmcp.so.6",
    "libcdio.so.19",
    "libusb-1.0.so.0",
    "libslang.so.2",
    "libncursesw.so.6",
    "libtinfo.so.6",
    "libGLdispatch.so.0",
    "libGLX.so.0",
    "libpulsecommon-15.99.so", # Note: This has a specific version, include if present
    "libdbus-1.so.3",
    "libXcursor.so.1",
    "libXinerama.so.1",
    "libXi.so.6",
    "libXfixes.so.3",
    "libXrandr.so.2",
    "libXss.so.1",
    "libXxf86vm.so.1",
    "libgbm.so.1",
    "libwayland-egl.so.1",
    "libwayland-client.so.0",
    "libwayland-cursor.so.0",
    "libxkbcommon.so.0",
    "libdecor-0.so.0",
    "libbsd.so.0",
    "libblas.so.3",
    "liblapack.so.3",
    "libstdc++.so.6",
    "libdl.so.2",
    "libserd-0.so.0",
    "libsord-0.so.0",
    "libsratom-0.so.0",
    "libsamplerate.so.0",
    "libgcc_s.so.1",
    "libflite_usenglish.so.1",
    "libflite_cmulex.so.1",
    "libharfbuzz.so.0",
    "libgomp.so.1",
    "libsodium.so.23",
    "libpgm-5.3.so.0",
    "libnorm.so.1",
    "libgssapi_krb5.so.2",
    "libexpat.so.1",
    "libuuid.so.1",
    "libpng16.so.16",
    "libbrotlidec.so.1",
    "libicuuc.so.70",
    "libmpg123.so.0",
    "libvorbisfile.so.3",
    "libudfread.so.0",
    "libp11-kit.so.0",
    "libidn2.so.0",
    "libunistring.so.2",
    "libtasn1.so.6",
    "libnettle.so.8",
    "libhogweed.so.6",
    "libgmp.so.10",
    "libssl.so.3",
    "libcrypto.so.3",
    "libgcrypt.so.20",
    "libgpg-error.so.0",
    "libcairo-gobject.so.2",
    "libgdk_pixbuf-2.0.so.0",
    "libgio-2.0.so.0",
    "libpangocairo-1.0.so.0",
    "libpango-1.0.so.0",
    "libffi.so.8",
    "libpcre.so.3",
    "libpixman-1.so.0",
    "libxcb-render.so.0",
    "libXrender.so.1",
    "libogg.so.0",
    "libnuma.so.1",
    "libudev.so.1",
    "libsndfile.so.1",
    "libX11-xcb.so.1",
    "libsystemd.so.0",
    "libasyncns.so.0",
    "libapparmor.so.1",
    "libwayland-server.so.0",
    "libxcb-randr.so.0",
    "libmd.so.0",
    "libgfortran.so.5",
    "libgraphite2.so.3",
    "libkrb5.so.3",
    "libk5crypto.so.3",
    "libcom_err.so.2",
    "libkrb5support.so.0",
    "libbrotlicommon.so.1",
    "libicudata.so.70",
    "libgmodule-2.0.so.0",
    "libjpeg.so.8",
    "libmount.so.1",
    "libselinux.so.1",
    "libpangoft2-1.0.so.0",
    "libthai.so.0",
    "libFLAC.so.8",
    "libzstd.so.1",
    "liblz4.so.1",
    "libcap.so.2",
    "libquadmath.so.0",
    "libkeyutils.so.1",
    "libresolv.so.2",
    "libblkid.so.1",
    "libpcre2-8.so.0",
    "libdatrie.so.1",
]


# Define the path to your sample audio file within the AI Studio environment
# This is used *only* for the input_example during registration now.
SAMPLE_AUDIO_LOCAL_PATH = "/home/jovyan/AffectLink/data/sample_audio.wav"

# --- Custom Pyfunc Model for Whisper ---
class WhisperPyfuncModel(PythonModel):

    def load_context(self, context):
        # ... (your existing FFMPEG setup remains unchanged) ...
        logger.info("Loading WhisperPyfuncModel context...")

        bundled_ffmpeg_path = os.path.join(context.artifacts["ffmpeg_binaries"], "ffmpeg")
        bundled_lib_path = context.artifacts["ffmpeg_binaries"]

        os.chmod(bundled_ffmpeg_path, 0o755)

        os.environ["PATH"] = f"{os.path.dirname(bundled_ffmpeg_path)}:{os.environ.get('PATH', '')}"
        current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{bundled_lib_path}:{current_ld_library_path}"

        logger.info(f"Set PATH to: {os.environ['PATH']}")
        logger.info(f"Set LD_LIBRARY_PATH to: {os.environ['LD_LIBRARY_PATH']}")

        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=False)
            logger.info("FFmpeg successfully found and appears to be working after setting PATH/LD_LIBRARY_PATH.")
        except Exception as e:
            logger.error(f"Failed to run bundled ffmpeg: {e}")
            raise RuntimeError(f"FFmpeg not accessible in deployment environment: {e}")

        # 2. Load the Whisper model
        self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        logger.info(f"Whisper model '{WHISPER_MODEL_SIZE}' loaded successfully.")

        # The bundled audio file path is no longer explicitly needed in self,
        # as the model will accept base64 audio directly.
        # However, keep the artifact bundling for the input_example to work if needed.
        # self.sample_audio_deployed_path = context.artifacts["sample_audio_file"]
        # logger.info(f"Sample audio file available at: {self.sample_audio_deployed_path}")


    def predict(self, context, model_input):
        # The input is expected to be a DataFrame with an 'audio_base64' column
        if isinstance(model_input, dict):
            if "inputs" in model_input and "audio_base64" in model_input["inputs"]:
                model_input = pd.DataFrame({"audio_base64": model_input["inputs"]["audio_base64"]})
            else:
                model_input = pd.DataFrame(model_input) # Fallback, though likely to fail if not 'audio_base64'
        
        if 'audio_base64' not in model_input.columns:
            raise ValueError("Input DataFrame must contain an 'audio_base64' column.")

        results = []
        for index, row in model_input.iterrows():
            audio_base64 = row['audio_base64']
            
            try:
                # Decode base64 string to bytes
                audio_bytes = base64.b64decode(audio_base64)
                
                # Save bytes to a temporary file, as whisper.load_audio expects a path
                # Use a BytesIO object if you can ensure whisper can read from it,
                # but a temp file is more robust for libraries expecting a path.
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                    temp_audio_file.write(audio_bytes)
                    temp_audio_file_path = temp_audio_file.name
                    logger.info(f"Received and saved audio to temporary file: {temp_audio_file_path}")

                    # Load audio from the temporary file
                    audio = whisper.load_audio(temp_audio_file_path)
                    
                    # Transcribe
                    transcription_result = self.whisper_model.transcribe(audio, language="en", fp16=torch.cuda.is_available())
                    results.append(transcription_result["text"])
            except Exception as e:
                logger.error(f"Error during transcription: {e}")
                results.append(f"Error: Failed to process audio: {e}") 

        return pd.DataFrame({'transcription': results})

# --- Main Registration Logic ---
def register_whisper_model_with_ffmpeg_bundle():
    mlflow.set_experiment("AffectLink_Model_Registration")

    temp_ffmpeg_dir = tempfile.mkdtemp()
    # temp_audio_dir is no longer needed to bundle the audio file directly for runtime
    # but we can keep it for the input_example generation for registration testing.
    temp_audio_for_input_example_dir = tempfile.mkdtemp() 

    try:
        # ... (your existing ffmpeg copying logic) ...
        shutil.copy(LOCAL_FFMPEG_PATH, os.path.join(temp_ffmpeg_dir, "ffmpeg"))
        logger.info(f"Copied ffmpeg executable to: {temp_ffmpeg_dir}")

        for lib_name in FFMPEG_REQUIRED_LIBS:
            lib_path = os.path.join(LOCAL_FFMPEG_LIB_DIR, lib_name)
            if "libpulsecommon-15.99.so" == lib_name:
                specific_lib_path = os.path.join("/usr/lib/x86_64-linux-gnu/pulseaudio/", lib_name)
            elif "libOpenCL.so.1" == lib_name:
                 specific_lib_path = os.path.join("/usr/local/cuda/targets/x86_64-linux/lib/", lib_name)
            else:
                specific_lib_path = lib_path

            if os.path.exists(specific_lib_path):
                shutil.copy(specific_lib_path, temp_ffmpeg_dir)
                logger.info(f"Copied shared library {lib_name} to: {temp_ffmpeg_dir}")
            else:
                logger.warning(f"Shared library {lib_name} not found at {specific_lib_path}. This might cause issues.")
        
        # Prepare data for input_example
        # We still need the sample audio to generate a valid base64 string for the input_example
        with open(SAMPLE_AUDIO_LOCAL_PATH, "rb") as f:
            sample_audio_bytes = f.read()
        sample_audio_base64 = base64.b64encode(sample_audio_bytes).decode('utf-8')

        # Define artifacts to bundle
        # Only ffmpeg_binaries are critical here for the model's runtime.
        # The sample_audio_file artifact is no longer strictly necessary for the model's *functionality*,
        # but you *could* keep it if you still wanted it accessible as a reference.
        # For simplicity, let's only bundle ffmpeg if audio is passed via base64.
        artifacts = {
            "ffmpeg_binaries": temp_ffmpeg_dir,
            # "sample_audio_file": os.path.join(temp_audio_for_input_example_dir, os.path.basename(SAMPLE_AUDIO_LOCAL_PATH)) # No longer needed if predicting from base64
        }
        
        with mlflow.start_run() as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            logger.info(f"Using explicit pip requirements for Whisper: {WHISPER_PIP_REQUIREMENTS}")

            # NEW: Schema now expects 'audio_base64' as a string
            input_schema = Schema([TensorSpec(np.dtype(str), (-1,), name="audio_base64")])
            output_schema = Schema([TensorSpec(np.dtype(str), (-1,), name="transcription")])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)
            
            # NEW: input_example uses the base64 string
            input_example = pd.DataFrame({"audio_base64": [sample_audio_base64]})

            mlflow.pyfunc.log_model(
                python_model=WhisperPyfuncModel(),
                artifact_path=f"whisper_asr_model_{WHISPER_MODEL_SIZE}",
                registered_model_name=f"WhisperASRModel_{WHISPER_MODEL_SIZE}",
                pip_requirements=WHISPER_PIP_REQUIREMENTS,
                artifacts=artifacts, 
                signature=signature,
                input_example=input_example
            )
            logger.info(f"Registered Whisper '{WHISPER_MODEL_SIZE}' ASR Model with bundled FFmpeg.")

    except Exception as e:
        logger.error(f"Failed to register Whisper ASR Model with bundled FFmpeg: {e}")
        raise e
    finally:
        # Clean up temporary directories
        if os.path.exists(temp_ffmpeg_dir):
            shutil.rmtree(temp_ffmpeg_dir)
            logger.info(f"Cleaned up temporary directory: {temp_ffmpeg_dir}")
        if os.path.exists(temp_audio_for_input_example_dir): # Clean up temp dir for input_example audio
            shutil.rmtree(temp_audio_for_input_example_dir)
            logger.info(f"Cleaned up temporary audio directory for input_example: {temp_audio_for_input_example_dir}")

if __name__ == "__main__":
    logger.info("Starting MLflow model registration process...")
    register_whisper_model_with_ffmpeg_bundle()
    logger.info("MLflow model registration process completed.")
import os
import speech_recognition as sr
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from config import LISTEN_TIMEOUT, PHRASE_TIME_LIMIT

# Load environment variables
load_dotenv()


class VoiceAssistant:
    def __init__(self):
        self.elevenlabs_client = None
        self.tts_enabled = False

        # 1. Initialize ElevenLabs
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            print("[Voice] ⚠ CRITICAL: ELEVENLABS_API_KEY not found in .env file.")
            print("[Voice] ⚠ Text-to-Speech will be DISABLED.")
        else:
            try:
                self.elevenlabs_client = ElevenLabs(api_key=api_key)
                self.tts_enabled = True
                print("[Voice] ✓ ElevenLabs TTS initialized successfully")
            except Exception as e:
                print(f"[Voice] ⚠ ElevenLabs initialization error: {e}")
                self.tts_enabled = False

        # 2. Initialize Speech Recognition (STT)
        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 4000
            self.recognizer.dynamic_energy_threshold = True
            self.stt_enabled = True
            print("[Voice] ✓ Speech Recognition initialized")

        except Exception as e:
            print(f"[Voice] ⚠ Speech recognition initialization failed: {e}")
            self.stt_enabled = False

    def speak(self, text: str, print_text: bool = True) -> bool:
        """
        Speaks the given text using ElevenLabs.
        Saves the audio to 'output.mp3' and plays it.
        """
        if print_text:
            print(f"\n🤖 Assistant: {text}")

        if not self.tts_enabled or not self.elevenlabs_client:
            print("[Voice] ⚠ TTS is disabled or not initialized.")
            return False

        try:
            # print("[Voice] 🔊 Generating audio with ElevenLabs...")

            # Generate audio stream
            audio_stream = self.elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id="cgSgspJ2msm6clMCkdW9",  # 'Jessica' or similar standard voice
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128"
            )

            # Consume generator to bytes to allow both saving and playing
            audio_bytes = b"".join(audio_stream)

            # Save to file
            with open("output.mp3", "wb") as f:
                f.write(audio_bytes)

            # Play audio
            play(audio_bytes)
            return True

        except Exception as e:
            print(f"[Voice] ⚠ ElevenLabs TTS Error: {e}")
            return False

    def listen(self, prompt: str = "Listening...", timeout: int = LISTEN_TIMEOUT) -> str:
        """
        Listens for user input via microphone.
        """
        if not self.stt_enabled:
            print(f"\n❓ {prompt}")
            return input("Type your answer: ").strip()

        try:
            # Speak prompt without printing it twice
            self.speak(prompt, print_text=False)
            print(f"\n🎤 {prompt}")

            with sr.Microphone() as source:
                print("[Voice] Adjusting for background noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

                print("[Voice] 🎙️ Listening... (speak now)")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=PHRASE_TIME_LIMIT
                )

                print("[Voice] Processing speech...")
                text = self.recognizer.recognize_google(audio)

                print(f"✓ You said: \"{text}\"")
                return text.lower().strip()

        except sr.WaitTimeoutError:
            print("[Voice] ⚠ No speech detected (timeout)")
            return self._fallback_text_input(prompt)

        except sr.UnknownValueError:
            print("[Voice] ⚠ Could not understand audio")
            return self._fallback_text_input(prompt)

        except sr.RequestError as e:
            print(f"[Voice] ⚠ Speech service error: {e}")
            return self._fallback_text_input(prompt)

        except Exception as e:
            print(f"[Voice] ⚠ Error: {e}")
            return self._fallback_text_input(prompt)

    def _fallback_text_input(self, prompt: str) -> str:
        """Falls back to keyboard input if voice fails"""
        print(f"\n❓ {prompt}")
        return input("Type your answer: ").strip().lower()

    def ask_yes_no(self, question: str) -> bool:
        """Helper for Yes/No questions"""
        response = self.listen(f"{question} (say yes or no)")
        return response in ['yes', 'y', 'yeah', 'yep', 'sure', 'okay', 'ok']

    def get_choice(self, question: str, options: list) -> str:
        """Helper for multiple choice questions"""
        options_text = ", ".join(options)
        full_question = f"{question} Options are: {options_text}"

        response = self.listen(full_question)

        for option in options:
            if option.lower() in response:
                return option

        return response

import cv2
import os
import random

import librosa
import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
except ImportError:
    sd = None



class _SpeakerStream:
    def __init__(self, sample_rate):
        self.stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
        self.stream.start()

    def play(self, samples):
        if self.stream:
            self.stream.write(samples.astype(np.float32, copy=False))

    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class LiveAudioVisualizer:
    def __init__(self, sample_rate, window_seconds=2.0, width=640, height=240, window_name="Audio Monitor"):
        self.sample_rate = sample_rate
        self.window_size = max(1, int(sample_rate * window_seconds))
        self.width = width
        self.height = height
        self.window_name = window_name
        self.buffer = np.zeros(self.window_size, dtype=np.float32)
        self.active = True
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)
        except Exception as e:
            print(f"Warning: failed to create audio visualizer window: {e}")
            self.active = False

    def update(self, samples):
        if not self.active or samples is None:
            return
        samples = np.asarray(samples, dtype=np.float32).flatten()
        if samples.size == 0:
            return
        self.buffer = np.concatenate([self.buffer, samples])[-self.window_size:]
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mid_y = self.height // 2
        cv2.line(img, (0, mid_y), (self.width - 1, mid_y), (80, 80, 80), 1)
        xs = np.linspace(0, self.width - 1, num=self.buffer.size).astype(np.int32)
        ys = (mid_y - self.buffer * (self.height * 0.4)).astype(np.int32)
        ys = np.clip(ys, 0, self.height - 1)
        pts = np.stack([xs, ys], axis=1)
        if pts.shape[0] >= 2:
            cv2.polylines(img, [pts], False, (0, 255, 0), 1)
        cv2.putText(img, "Audio Waveform", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)

    def close(self):
        if self.active:
            cv2.destroyWindow(self.window_name)
            self.active = False


class Audio:
    def __init__(
        self,
        audio_length=60.0,
        audio_sample_rate=16000,
        sim_timestep=1 / 250,
        enable_speaker=False,
        visualizer_window_seconds=2.0,
        visualizer_window_name="Audio Monitor",
        loop_audio=True,
        **kwargs,
    ):
        """Manage the rolling audio buffer used during simulation."""
        self.sample_rate = audio_sample_rate
        self.audio_length = audio_length
        self.sim_timestep = sim_timestep
        self.samples_per_step = int(audio_sample_rate * sim_timestep)
        self.buffer_size = int(audio_length * audio_sample_rate)
        self.loop_extension_seconds = kwargs.get("loop_extension_seconds", self.audio_length * 2)
        self.loop_crossfade_duration = kwargs.get("loop_crossfade_duration", 0.01)
        self.loop_crossfade_samples = max(0, int(self.sample_rate * self.loop_crossfade_duration))
        self.loop_target_length = max(
            self.buffer_size * 2,
            int(self.sample_rate * self.loop_extension_seconds),
        )
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_index = 0
        self.total_steps = 0
        self.is_playing = False
        self.current_audio = None
        self.current_audio_sr = None
        self.audio_index = 0
        self.play_count = 0
        self.stop_count = 0
        self.loop_audio_default = bool(loop_audio)
        self.loop_audio_enabled = self.loop_audio_default
        self.loaded_audio_raw = None
        self.loaded_audio_loop = None
        self.current_audio_length = 0.0
        self.delay_samples_remaining = 0
        self._pending_start = False
        
        padding_audio_path = os.path.join("assets", "audios", "padding_audio.wav")
        try:
            padding_raw, padding_sr = librosa.load(padding_audio_path, sr=None, mono=True)
            self.padding_audio = librosa.resample(padding_raw, orig_sr=padding_sr, target_sr=self.sample_rate)
            self.padding_audio = self.padding_audio.astype(np.float32)
            self.padding_audio = self._build_continuous_audio(
                self.padding_audio,
                min_length=self.loop_target_length,
            )
            self.padding_audio_length = len(self.padding_audio) / self.sample_rate
        except Exception as e:
            print(f"Warning: Failed to load padding audio {padding_audio_path}: {e}")
            self.padding_audio = np.zeros(self.loop_target_length, dtype=np.float32)
            self.padding_audio_length = self.loop_target_length / self.sample_rate

        self.padding_index = 0
        self._fill_buffer_with_padding()

        self.enable_speaker = bool(enable_speaker and sd is not None)
        if enable_speaker and sd is None:
            print("Warning: sounddevice is unavailable, disable speaker output.")
        self._speaker_stream = None
        self._visualizer = None
        self._visualizer_window_seconds = visualizer_window_seconds
        self._visualizer_window_name = visualizer_window_name

        if self.enable_speaker:
            self._init_realtime_outputs()
 
    def _fill_buffer_with_padding(self):
        """Fill the rolling buffer with padding audio."""
        if len(self.padding_audio) == 0:
            return
            
        filled = 0
        while filled < self.buffer_size:
            remaining = self.buffer_size - filled
            padding_remaining = len(self.padding_audio) - self.padding_index
            
            if padding_remaining >= remaining:
                self.audio_buffer[filled:] = self.padding_audio[self.padding_index:self.padding_index + remaining]
                self.padding_index = (self.padding_index + remaining) % len(self.padding_audio)
                filled = self.buffer_size
            else:
                self.audio_buffer[filled:filled + padding_remaining] = self.padding_audio[self.padding_index:]
                filled += padding_remaining
                self.padding_index = 0
        
        self.buffer_index = 0

    def _apply_loop_setting(self):
        if self.loaded_audio_raw is None:
            self.current_audio = None
            self.current_audio_length = 0.0
            return
        if self.loop_audio_enabled and self.loaded_audio_loop is not None:
            source = self.loaded_audio_loop
        else:
            source = self.loaded_audio_raw
        self.current_audio = source
        self.current_audio_length = len(source) / self.sample_rate if source is not None else 0.0

    def _init_realtime_outputs(self):
        if not self.enable_speaker:
            return
        try:
            self._speaker_stream = _SpeakerStream(self.sample_rate)
        except Exception as e:
            print(f"Warning: failed to start speaker stream: {e}")
            self._speaker_stream = None
        try:
            visualizer = LiveAudioVisualizer(
                self.sample_rate,
                window_seconds=self._visualizer_window_seconds,
                window_name=self._visualizer_window_name,
            )
            self._visualizer = visualizer if visualizer.active else None
        except Exception as e:
            print(f"Warning: failed to create audio visualizer: {e}")
            self._visualizer = None

    def _update_realtime_outputs(self, samples):
        if not self.enable_speaker:
            return
        if self._speaker_stream is not None:
            self._speaker_stream.play(samples)
        if self._visualizer is not None:
            self._visualizer.update(samples)

    def save_continuous_audio(self, samples: np.ndarray, out_path: str):
        """Save audio samples to disk for debugging."""
        if samples is None or len(samples) == 0:
            raise ValueError("samples is empty")
        sf.write(out_path, samples.astype(np.float32), self.sample_rate)

    def _build_continuous_audio(self, audio, min_length):
        """Build a loop-friendly audio track from a short clip."""
        if audio is None:
            return np.zeros(int(min_length), dtype=np.float32)
        base = np.asarray(audio, dtype=np.float32)
        if base.size == 0:
            return np.zeros(int(min_length), dtype=np.float32)
        target_length = max(int(min_length), base.size)
        if base.size >= target_length:
            return base
        segments = [base.copy()]
        total_len = base.size
        while total_len < target_length:
            next_chunk = base.copy()
            crossfade = min(
                self.loop_crossfade_samples,
                len(segments[-1]),
                len(next_chunk) - 1,
            )
            crossfade = max(0, crossfade)
            if crossfade > 0:
                fade_out = np.linspace(1.0, 0.0, crossfade, endpoint=False, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, crossfade, endpoint=False, dtype=np.float32)
                segments[-1][-crossfade:] = (
                    segments[-1][-crossfade:] * fade_out + next_chunk[:crossfade] * fade_in
                )
                tail = next_chunk[crossfade:]
            else:
                tail = next_chunk
            if tail.size == 0:
                tail = np.zeros(1, dtype=np.float32)
            segments.append(tail.copy())
            total_len += tail.size
        continuous = np.concatenate(segments, axis=0)
        # self.save_continuous_audio(continuous, "debug_padding.wav")
        return continuous[:target_length]

    def _get_padding_samples(self, num_samples):
        """Return a fixed number of padding samples."""
        if len(self.padding_audio) == 0:
            return np.zeros(num_samples, dtype=np.float32)
        
        samples = np.zeros(num_samples, dtype=np.float32)
        filled = 0
        
        while filled < num_samples:
            remaining = num_samples - filled
            padding_remaining = len(self.padding_audio) - self.padding_index
            
            if padding_remaining >= remaining:
                samples[filled:] = self.padding_audio[self.padding_index:self.padding_index + remaining]
                self.padding_index = (self.padding_index + remaining) % len(self.padding_audio)
                filled = num_samples
            else:
                samples[filled:filled + padding_remaining] = self.padding_audio[self.padding_index:]
                filled += padding_remaining
                self.padding_index = 0
        
        return samples
        
    def reset(self):
        """Reset recorder and playback state for a new episode."""
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_index = 0
        self.total_steps = 0
        self.is_playing = False
        self.current_audio = None
        self.current_audio_sr = None
        self.audio_index = 0
        self.padding_index = 0
        self.loop_audio_enabled = self.loop_audio_default
        self.delay_samples_remaining = 0
        self._pending_start = False
        self._fill_buffer_with_padding()
        # Start from a random point in the padding track to reduce repetition.
        max_offset_sec = min(10.0, self.padding_audio_length)
        max_offset_samples = int(max_offset_sec * self.sample_rate)
        self.padding_index = random.randint(0, max(0, max_offset_samples - 1)) if max_offset_samples > 0 else 0
    
    def load_audio(self, audio_path):
        """Load an audio file and resample it to the target sample rate."""
        try:
            audio_data, sr = librosa.load(audio_path, sr=None, mono=True)
            audio_resampled = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            self.loaded_audio_raw = audio_resampled.astype(np.float32)
            self.loaded_audio_loop = self._build_continuous_audio(
                self.loaded_audio_raw,
                min_length=max(self.loop_target_length, len(self.loaded_audio_raw)),
            )
            self._apply_loop_setting()
            return True
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            self.current_audio_length = 0.0
            return False
    
    def start_playing(self, audio_path=None, loop_audio=None, randomize_start: bool = True, delay: float = 0.0):
        """Start audio playback.

        Args:
            audio_path: Load this clip before playback when provided.
            loop_audio: Override the default looping behavior.
            randomize_start: Randomize the start point within the first 10 seconds.
            delay: Delay playback while outputting padding audio.
        """
        if audio_path is not None:
            if not self.load_audio(audio_path):
                return False
        if loop_audio is None:
            self.loop_audio_enabled = self.loop_audio_default
        else:
            self.loop_audio_enabled = bool(loop_audio)
        self._apply_loop_setting()
        if self.current_audio is None:
            return False

        if randomize_start:
            max_offset_sec = min(10.0, self.current_audio_length)
            max_offset_samples = int(max_offset_sec * self.sample_rate)
            self.audio_index = random.randint(0, max(0, max_offset_samples - 1)) if max_offset_samples > 0 else 0
        else:
            self.audio_index = 0

        delay = max(0.0, float(delay or 0.0))
        self.delay_samples_remaining = int(round(delay * self.sample_rate))
        if self.delay_samples_remaining > 0:
            self._pending_start = True
            self.is_playing = False
        else:
            self._pending_start = False
            self.is_playing = True
            self.play_count += 1
        return True

    def stop_playing(self):
        """Stop playback and switch back to padding audio."""
        if self.is_playing:
            self.is_playing = False
            max_offset_sec = min(10.0, self.padding_audio_length)
            max_offset_samples = int(max_offset_sec * self.sample_rate)
            self.padding_index = random.randint(0, max(0, max_offset_samples - 1)) if max_offset_samples > 0 else 0
            self.audio_index = 0
            self.delay_samples_remaining = 0
            self._pending_start = False
            self.stop_count += 1

    def update(self):
        """Advance playback and recording state by one simulation step."""
        if self.delay_samples_remaining > 0:
            samples = self._get_padding_samples(self.samples_per_step)
            self.delay_samples_remaining = max(0, self.delay_samples_remaining - self.samples_per_step)
            if self.delay_samples_remaining == 0 and self._pending_start and self.current_audio is not None:
                self._pending_start = False
                self.is_playing = True
                self.play_count += 1
        elif self.is_playing and self.current_audio is not None:
            remaining = len(self.current_audio) - self.audio_index
            if remaining >= self.samples_per_step:
                samples = self.current_audio[self.audio_index:self.audio_index + self.samples_per_step]
                self.audio_index += self.samples_per_step
            else:
                samples = np.zeros(self.samples_per_step, dtype=np.float32)
                if remaining > 0:
                    samples[:remaining] = self.current_audio[self.audio_index:]
                if self.loop_audio_enabled:
                    loop_samples = self.samples_per_step - remaining
                    samples[remaining:] = self.current_audio[:loop_samples]
                    self.audio_index = loop_samples
                else:
                    self.stop_playing()
                    padding = self._get_padding_samples(self.samples_per_step - remaining)
                    samples[remaining:] = padding
        else:
            samples = self._get_padding_samples(self.samples_per_step)

        self._write_to_buffer(samples)
        self._update_realtime_outputs(samples)
        self.total_steps += 1

    def _write_to_buffer(self, samples):
        """Write samples into the rolling ring buffer."""
        write_len = len(samples)
        end_index = self.buffer_index + write_len
        
        if end_index <= self.buffer_size:
            self.audio_buffer[self.buffer_index:end_index] = samples
            self.buffer_index = end_index % self.buffer_size
        else:
            first_part_len = self.buffer_size - self.buffer_index
            self.audio_buffer[self.buffer_index:] = samples[:first_part_len]
            second_part_len = write_len - first_part_len
            self.audio_buffer[:second_part_len] = samples[first_part_len:]
            self.buffer_index = second_part_len
    
    def get_recent_audio(self, duration):
        """Return the most recent audio window."""
        num_samples = int(duration * self.sample_rate)
        num_samples = min(num_samples, self.buffer_size)
        
        start_index = (self.buffer_index - num_samples) % self.buffer_size
        
        if start_index < self.buffer_index:
            audio_data = self.audio_buffer[start_index:self.buffer_index].copy()
        else:
            first_part = self.audio_buffer[start_index:].copy()
            second_part = self.audio_buffer[:self.buffer_index].copy()
            audio_data = np.concatenate([first_part, second_part])
        
        return audio_data
    
    def get_full_audio_buffer(self):
        """Return the full audio recorded for the current episode."""
        return self.export_episode_audio()
    
    def get_recorded_duration(self):
        return min(self.total_steps * self.sim_timestep, self.audio_length)
    
    def export_episode_audio(self):
        max_samples = self._get_recorded_samples()
        if max_samples == 0:
            return np.zeros(0, dtype=np.float32)
        recorded_duration = self.get_recorded_duration()
        requested_samples = int(np.round(recorded_duration * self.sample_rate))
        if requested_samples <= 0:
            requested_samples = max_samples
        num_samples = min(max_samples, requested_samples)
        start_index = (self.buffer_index - num_samples) % self.buffer_size
        if start_index < self.buffer_index:
            return self.audio_buffer[start_index:self.buffer_index].copy()
        first_part = self.audio_buffer[start_index:].copy()
        second_part = self.audio_buffer[:self.buffer_index].copy()
        return np.concatenate([first_part, second_part])
    
    def _get_recorded_samples(self):
        return min(self.total_steps * self.samples_per_step, self.buffer_size)

    def get_status(self):
        """Return playback status metadata."""
        return {
            'is_playing': self.is_playing,
            'play_count': self.play_count,
            'stop_count': self.stop_count,
            'buffer_index': self.buffer_index,
            'total_steps': self.total_steps,
            'audio_loaded': self.current_audio is not None,
            'sample_rate': self.sample_rate,
            'recorded_duration': self.get_recorded_duration(),
            'recorded_samples': self._get_recorded_samples(),
            'loop_audio_enabled': self.loop_audio_enabled,
            'delay_samples_remaining': self.delay_samples_remaining,
        }
    
    def close(self):
        if self._speaker_stream is not None:
            self._speaker_stream.close()
            self._speaker_stream = None
        if self._visualizer is not None:
            self._visualizer.close()
            self._visualizer = None

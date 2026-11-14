"""Bridge: offline speech companion for people with aphasia.

Live mode (normal use):
    python main.py
      hold SPACE  = record the patient, release to stop
      hold C      = record the caregiver's question (used as context next turn)
      1 / 2 / 3   = speak that option,  0 = none of these, try again
      Q or Esc    = quit

Test mode (no keyboard or mic needed; used for end-to-end checks):
    python main.py --wav clip.wav --pick 1 [--caregiver "Are you hungry?"] [--pointing cup]
"""
import argparse
import queue
import sys
import termios
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from audio import SAMPLE_RATE, PushToTalkRecorder, Transcriber, speak
from brain import LLM_MODEL, Brain
from fragments import check_urgent, make_fragments

console = Console()


def ms(seconds):
    return f"{seconds * 1000:.0f} ms"


# ---------------------------------------------------------------------------
# One patient turn: transcribe -> fragments -> urgent check -> Bonsai -> show
# ---------------------------------------------------------------------------

def warm_up(brain):
    """Load Bonsai into memory now, so the patient's first turn isn't slow."""
    t0 = time.perf_counter()
    try:
        brain.warm_up()
        console.print(f"[green]Bonsai warm[/] [dim]({ms(time.perf_counter() - t0)})[/]")
    except Exception as e:
        console.print(f"[red]Bonsai warm-up failed: {e}. Is LM Studio running on port 1235?[/]")


def run_patient_turn(stt, brain, audio=None, wav_path=None, caregiver_question=None,
                     pointing_at=None):
    """Process one recording. Returns (options, timings) or (None, timings) if
    there is nothing to choose from (empty transcript or URGENT alert)."""
    timings = {}

    # 1) Speech to text
    t0 = time.perf_counter()
    if wav_path:
        text, words = stt.transcribe_file(wav_path)
    else:
        text, words = stt.transcribe(audio)
    timings["speech-to-text (WhisperKit)"] = time.perf_counter() - t0

    # 2) Fragments: split on pauses if we got word timestamps, else raw text
    t0 = time.perf_counter()
    fragments, method = make_fragments(text, words)
    timings["fragments"] = time.perf_counter() - t0

    console.print(f"[bold]Raw transcript:[/] {text or '[dim](nothing heard)[/]'}")
    console.print(f"[bold]Fragment method:[/] {method}")
    if words:
        console.print("[dim]Word timings: " + "  ".join(
            f"{w['word']}({w['start']:.2f}-{w['end']:.2f})" for w in words) + "[/]")
    console.print("[bold]Fragments:[/] " + "  |  ".join(f"[cyan]{f}[/]" for f in fragments))

    if not fragments:
        console.print("[yellow]Nothing heard. Try again.[/]")
        show_timings(timings)
        return None, timings

    # 3) Urgent keyword rule: plain Python, runs BEFORE the AI and skips it
    t0 = time.perf_counter()
    reason = check_urgent(fragments)
    timings["urgent check"] = time.perf_counter() - t0
    if reason:
        console.print(Panel(
            f"[bold white]The patient may need help right now.[/]\n\n"
            f"Heard: \"{' '.join(fragments)}\"\nTrigger: {reason}",
            title="[bold]URGENT: CAREGIVER CHECK NOW[/]",
            style="bold white on red", padding=(1, 2)))
        show_timings(timings)
        return None, timings

    # 4) Bonsai suggests 3 sentences (then the first-person filter runs)
    if caregiver_question:
        console.print(f"[bold]Context:[/] Caregiver just asked: \"{caregiver_question}\"")
    if pointing_at:
        console.print(f"[bold]Context:[/] Patient is pointing at: {pointing_at}")
    t0 = time.perf_counter()
    try:
        result = brain.suggest(fragments, caregiver_question, pointing_at)
    except Exception as e:
        console.print(f"[red]Bonsai failed: {e}. Is LM Studio running on port 1235?[/]")
        return None, timings
    timings["Bonsai options" + (" (asked twice)" if result["retried"] else "")] = \
        time.perf_counter() - t0
    options = result["options"]

    for d in result["dropped"]:
        console.print(f"[dim]Filtered out (not first person): {d}[/]")
    if not options:
        console.print("[yellow]No good options this time. Please try again.[/]")
        show_timings(timings)
        return None, timings

    console.print(f"[bold]Options[/] [dim](parsed via {result['how']})[/]:")
    for i, option in enumerate(options, start=1):
        console.print(f"  [bold green]{i}[/]  {option}")
    console.print("  [bold yellow]0[/]  None of these, try again")
    show_timings(timings)
    return options, timings


def show_timings(timings):
    table = Table(title="Time per stage", show_header=True, header_style="bold")
    table.add_column("Stage")
    table.add_column("Time", justify="right")
    for stage, seconds in timings.items():
        table.add_row(stage, ms(seconds))
    console.print(table)


def speak_choice(options, pick):
    """Speak option number `pick` (1-3). Returns the TTS time in seconds."""
    sentence = options[pick - 1]
    console.print(f"[bold green]Speaking:[/] {sentence}")
    t0 = time.perf_counter()
    engine = speak(sentence)
    seconds = time.perf_counter() - t0
    console.print(f"[dim]text-to-speech ({engine}): {ms(seconds)}[/]")
    return seconds


# ---------------------------------------------------------------------------
# Test mode: one turn from a WAV file, pick given on the command line
# ---------------------------------------------------------------------------

def run_test(args):
    stt, brain = Transcriber(), Brain()
    console.rule("[bold]Bridge test turn")
    warm_up(brain)
    console.print(f"[dim]Audio file: {args.wav}[/]")
    options, _ = run_patient_turn(stt, brain, wav_path=args.wav,
                                  caregiver_question=args.caregiver,
                                  pointing_at=args.pointing)
    if not options:
        return
    if args.pick == 0:
        console.print("[yellow]Pick 0: none of these. Nothing spoken.[/]")
    elif args.pick and 1 <= args.pick <= len(options):
        console.print(f"[bold]Pick:[/] {args.pick}")
        speak_choice(options, args.pick)
    else:
        console.print("[yellow]No --pick given, so nothing is spoken.[/]")


# ---------------------------------------------------------------------------
# Live mode: keyboard + microphone
# ---------------------------------------------------------------------------

def key_name(key):
    """Turn a pynput key into a simple string like "space", "c", "1"."""
    from pynput import keyboard
    if key == keyboard.Key.space:
        return "space"
    if key == keyboard.Key.esc:
        return "esc"
    char = getattr(key, "char", None)
    return char.lower() if char else None


class Keys:
    """Listens to the keyboard in the background and queues ("down"/"up", name)
    events. Holding a key makes the OS repeat "down" events; we drop those."""

    def __init__(self):
        from pynput import keyboard
        self.events = queue.Queue()
        self.held = set()
        self.listener = keyboard.Listener(on_press=self._down, on_release=self._up)
        self.listener.start()
        self.trusted = getattr(self.listener, "IS_TRUSTED", True)

    def _down(self, key):
        name = key_name(key)
        if name and name not in self.held:
            self.held.add(name)
            self.events.put(("down", name))

    def _up(self, key):
        name = key_name(key)
        if name:
            self.held.discard(name)
            self.events.put(("up", name))

    def clear(self):
        """Throw away keys pressed while we were busy (thinking or speaking)."""
        while not self.events.empty():
            self.events.get_nowait()

    def wait_down(self, allowed):
        """Block until one of the `allowed` keys is pressed; return its name."""
        while True:
            kind, name = self.events.get()
            if kind == "down" and name in allowed:
                return name

    def wait_up(self, name):
        while True:
            kind, n = self.events.get()
            if kind == "up" and n == name:
                return


def record_while_held(keys, recorder, key, label):
    """Record from the mic until `key` is released."""
    recorder.start()
    t0 = time.perf_counter()
    console.print(f"[bold red]● Recording {label}...[/] (release {key.upper()} to stop)")
    keys.wait_up(key)
    audio = recorder.stop()
    console.print(f"[dim]recorded {ms(time.perf_counter() - t0)}[/]")
    return audio


def run_live():
    stt, brain, recorder = Transcriber(), Brain(), PushToTalkRecorder()
    warm_up(brain)
    keys = Keys()
    if not keys.trusted:
        console.print("[bold yellow]macOS has not allowed this terminal to read the keyboard.[/]\n"
                      "Open System Settings > Privacy & Security > Input Monitoring "
                      "(and Accessibility), turn on your terminal app, then restart Bridge.")

    caregiver_question = None
    console.print(Panel(
        f"[bold]Bridge[/] is ready  (model: {LLM_MODEL})\n"
        "Hold [bold]SPACE[/] = patient speaks     Hold [bold]C[/] = caregiver asks\n"
        "[bold]Q[/] or [bold]Esc[/] = quit", style="blue"))

    while True:
        console.print("\n[bold]Waiting...[/] hold SPACE (patient) or C (caregiver)")
        keys.clear()
        key = keys.wait_down({"space", "c", "q", "esc"})
        if key in ("q", "esc"):
            return

        # ----- Caregiver mode: record the question and remember it -----
        if key == "c":
            audio = record_while_held(keys, recorder, "c", "CAREGIVER")
            if audio is None:
                console.print("[yellow]Too short, try again.[/]")
                continue
            t0 = time.perf_counter()
            text, _ = stt.transcribe(audio)
            console.print(f"[bold cyan]Caregiver asked:[/] {text}  [dim]({ms(time.perf_counter() - t0)})[/]")
            caregiver_question = text or None
            continue

        # ----- Patient turn -----
        audio = record_while_held(keys, recorder, "space", "PATIENT")
        if audio is None:
            console.print("[yellow]Too short, try again.[/]")
            continue
        console.print(f"[dim]audio length: {len(audio) / SAMPLE_RATE:.1f} s[/]")
        try:
            options, _ = run_patient_turn(stt, brain, audio=audio,
                                          caregiver_question=caregiver_question)
        except Exception as e:
            console.print(f"[red]Speech-to-text failed: {e}. Is WhisperKit running on port 50060?[/]")
            continue

        if not options:
            # Empty transcript or URGENT alert: wait for the caregiver to acknowledge.
            console.print("[dim]Press any number key to continue.[/]")
            keys.clear()
            keys.wait_down({"0", "1", "2", "3", "space"})
            continue

        # Patient picks. Nothing is spoken without a pick.
        console.print("[bold]Pick 1, 2, 3, or 0 to try again.[/]")
        keys.clear()
        allowed = {"0"} | {str(i) for i in range(1, len(options) + 1)}
        pick = int(keys.wait_down(allowed))
        if pick == 0:
            console.print("[yellow]OK, let's try again.[/]")
            continue        # keep the caregiver question for the retry
        speak_choice(options, pick)
        caregiver_question = None   # that question has been answered


def main():
    parser = argparse.ArgumentParser(description="Bridge: aphasia speech companion")
    parser.add_argument("--wav", help="test mode: process this WAV file instead of the mic")
    parser.add_argument("--pick", type=int, help="test mode: option to speak (1-3, or 0)")
    parser.add_argument("--caregiver", help="test mode: caregiver question for context")
    parser.add_argument("--pointing", help="test mode: object the patient points at, e.g. cup")
    args = parser.parse_args()

    if args.wav:
        run_test(args)
        return

    # Turn off terminal echo so held keys don't spray spaces/letters on screen.
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] &= ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSADRAIN, new)
    try:
        run_live()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcflush(fd, termios.TCIFLUSH)            # discard typed keys
        termios.tcsetattr(fd, termios.TCSADRAIN, old)    # restore echo
        console.print("\nBridge closed.")


if __name__ == "__main__":
    main()

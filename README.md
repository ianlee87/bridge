# Bridge: offline speech companion for aphasia (Bonsai + Argmax)

Bridge listens to a person with aphasia, turns their broken phrases into up to
3 sentences they probably meant, and speaks the one they pick.
Everything runs on this Mac. It works with Wi-Fi off.

```
hold SPACE -> WhisperKit (speech to text) -> split on pauses -> urgent check
           -> Bonsai (3 options) -> first-person filter -> patient picks 1/2/3
           -> argmax-cli speaks it
```

## Files

| File | What it does |
|---|---|
| `main.py` | The program: keyboard, screen display, and the main loop |
| `audio.py` | Push-to-talk recording, WhisperKit speech-to-text, argmax-cli speech |
| `fragments.py` | Splits words into fragments at pauses > 0.6 s; urgent keyword rule |
| `brain.py` | Bonsai prompt, 5 worked examples, JSON parsing, first-person filter, warm-up |
| `eval.py` | Scores Bonsai on the 15 test cases from text (no audio needed) |
| `test_phrases.txt` | 15 scored test cases: caregiver question, fragments, pointed-at object, expected sentence |

## Setup (one time)

**1. Python environment**

```bash
cd ~/bridge
python3 -m venv .venv
.venv/bin/pip install openai sounddevice numpy rich pynput
```

**2. Argmax speech binary.** Bridge calls the prebuilt
`~/argmax-oss-swift/.build/release/argmax-cli` directly. That's faster than
`swift run`, which checks the build every time. If that file doesn't exist, build it once:

```bash
cd ~/argmax-oss-swift && swift build -c release
```

**3. LM Studio.** Download and load the model `ternary-bonsai-2-4b-mlx`, then start
the local server on port **1235** (Developer tab > server port 1235 > Start).

**4. macOS permissions.** Open System Settings > Privacy & Security and turn on
your terminal app (Terminal, iTerm, VS Code…) under:
- **Input Monitoring**: lets Bridge see SPACE and C being held down and released
- **Accessibility**: also needed by the keyboard library on macOS
- **Microphone**: to record

Quit and reopen the terminal after changing these. If Input Monitoring is off,
Bridge prints a yellow warning and the keys do nothing.

## Every time: start the servers

1. **WhisperKit** (speech to text, port 50060), in its own terminal window:
   ```bash
   cd ~/argmax-oss-swift && BUILD_ALL=1 swift run argmax-cli serve --model base.en --verbose
   ```
   It's ready when `curl localhost:50060/v1/models` stops saying
   "connection refused" (about 25 s).
2. **LM Studio** server on port 1235 with Bonsai loaded (see setup step 3).

## Live mode

```bash
cd ~/bridge
.venv/bin/python main.py
```

On start, Bridge sends Bonsai a tiny request so the model is loaded, and prints **Bonsai warm**.

| Key | Action |
|---|---|
| hold **SPACE** | Record the patient. Release to stop. |
| hold **C** | Record the caregiver's question. It's given to Bonsai as context for the patient's next turn. |
| **1 / 2 / 3** | Speak that option out loud |
| **0** | None of these, try again (nothing is spoken) |
| **Q** / **Esc** | Quit |

Nothing is ever spoken unless the patient presses 1, 2 or 3.

### First-person filter

After Bonsai answers, any option that ends with "?" or starts with "Do you",
"Would you", "Are you" or "Can you" is dropped, because those read like
someone talking *to* the patient. The filter is skipped if the patient's own
fragments look like a question (a fragment ends in "?" or contains
who/what/where/when/why/how/which). If fewer than 2 options survive, Bonsai is
asked once more. So sometimes only 1 or 2 options are shown.

### Urgent alert

If the fragments contain **"chest" + "pain"**, **"can't breathe"** (also
"cant"/"cannot"), or **"help" on its own** ("help", "help me", "help, please"),
Bridge skips the AI and shows a red URGENT box for the caregiver.
"bathroom… help" is a normal request (it has another content word), and
"helpful" never counts. This is a plain keyword check, not the AI, so it is
instant and predictable. It is only a prompt to look at the patient. It is
**not** a medical alarm and does not call anyone.

## --wav test mode (no keyboard or mic)

Runs one turn from a WAV file. You can make one with macOS `say`;
`[[slnc 1000]]` inserts a 1-second pause:

```bash
say -o test.wav --data-format=LEI16@16000 "water [[slnc 1000]] want [[slnc 1000]] cold"
.venv/bin/python main.py --wav test.wav --pick 1
.venv/bin/python main.py --wav test.wav --caregiver "Are you thirsty?" --pointing cup
```

With no `--pick`, nothing is spoken. `--pointing` stands in for the future camera.

## eval.py: score Bonsai on the test cases

```bash
.venv/bin/python eval.py
```

Only LM Studio is needed (no WhisperKit, no audio). For each case in
`test_phrases.txt` it runs the urgent check, then Bonsai plus the filter, and prints:
- the options, plus any that the filter dropped
- a keyword-overlap score against the expected sentence: **HIT** ≥ 0.5,
  **UNSURE** 0.25–0.5 (judge these yourself), **MISS** < 0.25
- whether the URGENT case fired the keyword alert
- the hit rate on the 14 normal cases **with** and **without** `pointing_at`, side by side

The keyword score is rough. It counts shared words, not meaning, so
"I'm not taking this apple" scores as a hit for "I'd like the apple". Read the
options yourself before trusting the number.

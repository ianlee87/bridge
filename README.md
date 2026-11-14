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

On start, Bridge sends Bonsai a tiny request so the model is loaded (**Bonsai warm**),
then silently runs argmax-cli once on the word "ready" so the first spoken
sentence isn't slow (**Speech warm**). Both print how long they took.

| Key | Action |
|---|---|
| hold **SPACE** | Record the patient. Release to stop. |
| hold **C** | Record the caregiver's question. It's given to Bonsai as context for the patient's next turn. |
| **1 / 2 / 3** | Speak that option out loud |
| **0** | None of these, try again (nothing is spoken) |
| **Q** / **Esc** | Quit |

Nothing is ever spoken unless the patient presses 1, 2 or 3.

### First-person filter

After Bonsai answers, options that sound like someone talking *to* the patient
are dropped:
- they start with "Do you", "Would you" or "Are you", **or**
- they contain "you/your" but no "I/me/my".

So "Could I have a blanket?" and "Can you help me call my daughter?" are kept,
while "Do you want cold water?" and "Can you turn that down?" are dropped.

The filter is skipped if the patient's fragments contain
who/what/where/when/why/how. Whisper's "?" is ignored, because it adds one
whenever the voice rises ("Want?"). If fewer than 2 options survive, Bonsai is
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
.venv/bin/python eval.py           # ask Bonsai about every case, then judge them
.venv/bin/python eval.py --judge   # judge saved results again, no Bonsai calls
```

Only LM Studio is needed (no WhisperKit, no audio). Run it in a terminal.

1. **Generate:** each case in `test_phrases.txt` goes through the urgent check,
   then Bonsai plus the filter. The 14 normal cases run twice, **with** and
   **without** `pointing_at`. Everything is saved to `eval_results.csv`.
2. **Judge:** for each row you see the expected sentence and the options, and type
   **y** (an option says what the patient meant), **n**, **s** (skip) or
   **q** (save and quit). Each answer is saved right away. `--judge` only asks
   about rows that aren't marked yet; to redo one, clear its `judge` cell in the CSV.
   The URGENT case is judged automatically (did the alert fire?).
3. **Summary:** your hit rate with vs. without pointing, side by side, plus the
   keyword-overlap score as a rough second column, and average Bonsai time.

The keyword score counts shared words, not meaning. "I'm not taking this apple"
scores well for "I'd like the apple", so trust your y/n over it.

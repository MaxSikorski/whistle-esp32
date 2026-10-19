#!/usr/bin/env python3
"""Needle 2 / Whistle demo console.

Type a request, watch the model stream its reasoning and tool call, and see the
resulting LED action. Built for screen-recording a demo.

With --whistle it shows esp32/whistle_demo instead: nothing is typed - hold the
board's button and speak - and the console follows the board's events live:
recording, transcribing, the transcript, the timings and the LED.

Two interchangeable backends speaking the same line protocol:

  --local            spawns the host engine (rehearse on the Mac)
  --serial <port>    talks to the ESP32-S3 (the real thing)

Protocol, one message per line:
  EVT priming i/n PCT elapsed=Ns eta=Ns   one-time prefix prime at boot
  EVT prefix tokens=N ...                 prime finished
  EVT ready model=... psram_free=...      engine up
  EVT reading i/n                         per-request prefill progress
  EVT prefill tokens=N ms=M tps=T sink=S
  TOK <text>                              one token, literal "\\n" for newline
  CONF <float>
  EVT done tokens=N ms=M tps=T
  ACT led color=red mode=flash duration=2.0   |   ACT none
  EVT think=0|1
  ERR <reason>

A turn ends at ACT/ERR, not at "EVT done" - the device drives the LED after
reporting timings, and the swatch should reflect what the hardware actually did.

Whistle adds (esp32/whistle_demo; host/wh_run --protocol prints the same):
  EVT recording | EVT released held_ms=N  button down / up
  EVT speech ms=N frames=N                speech found, trimmed to N ms
  EVT lang <xx>                           language detected (en de fr es it nl pl)
  EVT no_speech | EVT too_short           nothing to transcribe (red blinks)
  TXT <transcript>
  EVT time total_ms=.. encode_ms=.. decode_ms=.. tokens=N audio_ms=N
  EVT prof <phase> N ms                   mel, stem, encoder, cross_kv, decoder

Usage:
  python3 tools/needle_tui.py --local  --tools tools/led.json
  python3 tools/needle_tui.py --serial /dev/cu.usbmodemXXXX
  python3 tools/needle_tui.py --whistle --serial /dev/cu.usbmodemXXXX
  python3 tools/needle_tui.py --whistle --local clip.wav     # rehearse on the Mac
"""
import argparse
import json
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

CONSOLE = Console()

SWATCH = {
    "red": "#ff3b30", "green": "#34c759", "blue": "#0a84ff",
    "yellow": "#ffd60a", "purple": "#af52de", "white": "#f2f2f7",
    "cyan": "#32ade6", "magenta": "#ff2d55", "pink": "#ff6482",
    "orange": "#ff9f0a", "off": "#3a3a3c",
}


class Backend:
    def send(self, prompt):
        raise NotImplementedError

    def lines(self):
        raise NotImplementedError

    def close(self):
        pass


class LocalBackend(Backend):
    """Spawn the host engine once per prompt."""

    name = "host engine"

    def __init__(self, binary, model, tools, max_new, think):
        self.binary, self.model, self.tools = binary, model, tools
        self.max_new, self.think = max_new, think
        self.proc = None

    def send(self, prompt):
        argv = [self.binary, self.model, "genp", self.tools, prompt,
                str(self.max_new)]
        if not self.think:
            argv.append("nothink")
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True,
                                     bufsize=1)

    def lines(self):
        for line in self.proc.stdout:
            yield line.rstrip("\n")
        self.proc.wait()


class SerialBackend(Backend):
    """Persistent link to the ESP32-S3."""

    def __init__(self, port, baud=115200, assert_dtr=True):
        import serial
        # Open with both lines low: on a board plugged in by its native
        # USB-Serial/JTAG port, pyserial's default open-time DTR/RTS glitch
        # can latch the chip into download mode (README, Whistle section).
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = port, baud, 0.2
        self.s.dtr, self.s.rts = False, False
        self.s.open()
        # The Needle firmware's secondary USB console only emits while the
        # host asserts DTR; whistle_demo's primary one does not need it.
        if assert_dtr:
            self.s.setDTR(True)
        self.name = "ESP32-S3 " + port.split("/")[-1]
        self.q = queue.Queue()
        self._stop = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        buf = b""
        while not self._stop:
            try:
                data = self.s.read(512)
            except Exception:
                return
            if not data:
                continue
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self.q.put(line.decode("utf-8", "replace").rstrip("\r"))

    def send(self, prompt):
        self.s.write((prompt + "\n").encode())
        self.s.flush()

    def drain(self, seconds=0.4):
        """Collect whatever the board has already said (boot banner, etc)."""
        out, deadline = [], time.time() + seconds
        while time.time() < deadline:
            try:
                out.append(self.q.get(timeout=0.1))
            except queue.Empty:
                pass
        return out

    def lines(self, idle_timeout=180):
        """Yield until the turn completes, or the board goes quiet."""
        last = time.time()
        while time.time() - last < idle_timeout:
            try:
                line = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            last = time.time()
            yield line
            if line.startswith("ACT ") or line.startswith("ERR "):
                return

    def close(self):
        self._stop = True
        try:
            self.s.close()
        except Exception:
            pass


class Session:
    def __init__(self, backend_name):
        self.backend_name = backend_name
        self.prompt = ""
        self.raw = ""
        self.status = "idle"
        self.prefill = None
        self.done = None
        self.conf = None
        self.action = None
        self.model_info = ""
        self.error = None
        self.reading = None      # (i, n) during prefill
        self.priming = None      # (i, n, pct, eta)
        self.think = True

    def feed(self, line):
        if line.startswith("TOK "):
            self.raw += line[4:].replace("\\n", "\n")
            self.status = "generating"
        elif line.startswith("EVT priming "):
            m = re.match(r"EVT priming (\d+)/(\d+)\s+(\d+)%.*eta=(\d+)s", line)
            if m:
                self.priming = tuple(int(g) for g in m.groups())
                self.status = "priming"
        elif line.startswith("EVT prefix"):
            self.priming = None
        elif line.startswith("EVT ready"):
            self.model_info = line[len("EVT ready "):]
        elif line.startswith("EVT reading "):
            m = re.match(r"EVT reading (\d+)/(\d+)", line)
            if m:
                self.reading = (int(m.group(1)), int(m.group(2)))
                self.status = "reading prompt"
        elif line.startswith("EVT prefill"):
            self.prefill = _kv(line)
            self.reading = None
            self.status = "generating"
        elif line.startswith("EVT done"):
            self.done = _kv(line)
            self.status = "acting"
        elif line.startswith("EVT think="):
            self.think = line.strip().endswith("1")
        elif line.startswith("CONF "):
            try:
                self.conf = float(line[5:])
            except ValueError:
                pass
        elif line.startswith("ACT "):
            self.action = _kv(line[4:]) if "=" in line else {}
            self.status = "done"
        elif line.startswith("ERR "):
            self.error = line[4:]
            self.status = "error"

    def think_text(self):
        m = re.search(r"<think>(.*?)(</think>|$)", self.raw, re.S)
        return m.group(1).strip() if m else ""

    def call_json(self):
        m = re.search(r"<tool_call>(.*?)(</tool_call>|$)", self.raw, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return m.group(1)

    def led(self):
        if self.action:
            return self.action.get("color"), self.action.get("mode")
        call = self.call_json()
        if isinstance(call, list) and call:
            args = call[0].get("arguments", {})
            return args.get("color"), args.get("mode")
        return None, None


def _kv(line):
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def render(sess):
    layout = Layout()
    layout.split_column(
        Layout(name="head", size=3),
        Layout(name="body"),
        Layout(name="foot", size=7),
    )

    dot = {"idle": "grey50", "priming": "yellow", "reading prompt": "yellow",
           "generating": "cyan", "acting": "magenta", "done": "green",
           "error": "red"}.get(sess.status, "grey50")
    head = Text()
    head.append("  Needle 2 ", style="bold white")
    head.append("· 45M · CQ2 · on-device · ", style="grey58")
    head.append(sess.backend_name, style="bold magenta")
    head.append("    ")
    head.append("●", style=dot)
    head.append(f" {sess.status}", style=dot)
    if not sess.think:
        head.append("   [fast: no reasoning]", style="yellow")
    layout["head"].update(Panel(head, border_style="grey30"))

    blocks = []

    if sess.priming:
        i, n, pct, eta = sess.priming
        bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
        p = Text()
        p.append("priming the tool schema (one time only)\n\n", style="bold yellow")
        p.append(bar + f"  {pct}%  ", style="yellow")
        p.append(f"{i}/{n} tokens · about {eta}s left", style="grey62")
        blocks.append(Align.center(p, vertical="middle"))
    else:
        if sess.prompt:
            p = Text()
            p.append("› ", style="bold green")
            p.append(sess.prompt, style="bold white")
            blocks.append(p)
            blocks.append(Text())

        if sess.reading:
            i, n = sess.reading
            r = Text()
            r.append("reading your request  ", style="yellow")
            r.append("▪" * i + "▫" * max(0, n - i), style="yellow")
            r.append(f"  {i}/{n}", style="grey62")
            blocks.append(r)
            blocks.append(Text())

        think = sess.think_text()
        if think:
            blocks.append(Panel(Text(think, style="italic grey70"),
                                title="[grey58]reasoning[/]", border_style="grey30"))

        call = sess.call_json()
        if call is not None:
            if isinstance(call, list):
                blocks.append(Panel(Text(json.dumps(call, indent=2), style="bold cyan"),
                                    title="[grey58]tool call[/]", border_style="cyan"))
            else:
                blocks.append(Panel(Text(str(call), style="yellow"),
                                    title="[grey58]tool call (streaming)[/]",
                                    border_style="grey30"))

        if sess.error:
            blocks.append(Panel(Text(sess.error, style="bold red"),
                                title="error", border_style="red"))

        if not blocks:
            blocks = [Align.center(Text("type a request and press enter",
                                        style="grey42"), vertical="middle")]

    layout["body"].update(Panel(Group(*blocks), border_style="grey30"))

    color, mode = sess.led()
    swatch = Text()
    if color:
        hexc = SWATCH.get(color, "#888888")
        bar = "█" * 22
        swatch.append(bar + "\n", style=hexc)
        swatch.append(bar + "\n", style="grey15" if mode == "flash" else hexc)
        swatch.append(bar + "\n", style=hexc)
        swatch.append(f"{color} · {mode or '-'}", style="bold " + hexc)
        if sess.action and sess.action.get("duration"):
            swatch.append(f" · {sess.action['duration']}s", style="grey62")
    else:
        swatch.append("\n\n  (no action)", style="grey35")

    stats = Table.grid(padding=(0, 2))
    stats.add_column(style="grey58", justify="right")
    stats.add_column(style="bold white")
    if sess.prefill:
        stats.add_row("prompt", f"{sess.prefill.get('tokens','?')} tok  "
                                f"{sess.prefill.get('tps','?')} tok/s")
    if sess.done:
        stats.add_row("generated", f"{sess.done.get('tokens','?')} tok  "
                                   f"{sess.done.get('tps','?')} tok/s")
        ms = sess.done.get("ms")
        if ms:
            stats.add_row("time", f"{float(ms)/1000:.1f} s")
    if sess.conf is not None:
        filled = int(max(0.0, min(1.0, sess.conf)) * 20)
        col = "green" if sess.conf >= 0.5 else "yellow"
        bar = Text("█" * filled + "░" * (20 - filled), style=col)
        bar.append(f"  {sess.conf:.3f}", style="bold " + col)
        stats.add_row("confidence", bar)
    if not sess.prefill and not sess.done and sess.model_info:
        for part in sess.model_info.split():
            if part.startswith(("layers=", "vocab=", "window=")):
                k, v = part.split("=")
                stats.add_row(k, v)

    foot = Layout()
    foot.split_row(Layout(name="led", ratio=1), Layout(name="stat", ratio=2))
    foot["led"].update(Panel(swatch, title="[grey58]LED[/]", border_style="grey30"))
    foot["stat"].update(Panel(stats, title="[grey58]stats[/]", border_style="grey30"))
    layout["foot"].update(foot)
    return layout


# ------------------------------------------------------------------ Whistle

LANG_NAMES = {"en": "English", "de": "Deutsch", "fr": "Français", "es": "Español",
              "it": "Italiano", "nl": "Nederlands", "pl": "Polski"}


class WhistleSession:
    """State of one push-to-talk turn on esp32/whistle_demo."""

    def __init__(self):
        self.status = "ready"
        self.model_info = ""
        self.text = None
        self.action = None
        self.time = {}
        self.prof = {}
        self.speech = {}
        self.lang = None
        self.error = None
        self.history = []        # (transcript, action) of earlier turns

    def feed(self, line):
        if line.startswith("EVT ready"):
            self.model_info = line[len("EVT ready "):]
            self.status, self.error = "ready", None
        elif line.startswith("EVT selftest"):
            self.status = "self-test"
        elif line.startswith("EVT recording"):
            if self.text is not None:
                self.history = ([(self.text, self.action)] + self.history)[:4]
            self.status, self.text, self.action = "recording", None, None
            self.time, self.prof, self.speech, self.lang = {}, {}, {}, None
            self.error = None
        elif line.startswith("EVT released"):
            self.status = "transcribing"
        elif line.startswith("EVT speech"):
            self.speech = _kv(line)
        elif line.startswith(("EVT no_speech", "EVT too_short")):
            self.status = "nothing heard"
        elif line.startswith("EVT lang "):
            self.lang = line.split()[2]
        elif line.startswith("TXT"):
            self.text = line[4:] if len(line) > 4 else ""
        elif line.startswith("EVT time"):
            self.time = _kv(line)
        elif line.startswith("EVT prof"):
            parts = line.split()
            if len(parts) >= 4:
                self.prof[parts[2]] = parts[3]
        elif line.startswith("ACT "):
            self.action = _kv(line[4:]) if "=" in line else {}
            if self.status != "nothing heard":
                self.status = "done" if self.action else "no colour heard"
        elif line.startswith("ERR "):
            self.error = line[4:]
            self.status, self.action = "error", None
        elif line.startswith("EVT  READY"):
            if self.status in ("self-test", "error"):
                self.status, self.error = "ready", None


def render_whistle(sess, backend_name):
    layout = Layout()
    layout.split_column(
        Layout(name="head", size=3),
        Layout(name="body"),
        Layout(name="foot", size=9),
    )
    col = {"ready": "grey50", "self-test": "yellow", "recording": "bold white",
           "transcribing": "cyan", "done": "green", "nothing heard": "red",
           "no colour heard": "yellow", "error": "bold red"}.get(sess.status, "grey50")
    depth = _kv(sess.model_info).get("depth", "?")
    head = Text()
    head.append("  Whistle ", style="bold white")
    head.append(f"· speech-to-text · {depth}-layer decoder · on-device · ", style="grey58")
    head.append(backend_name, style="bold magenta")
    head.append("    ")
    head.append("●", style=col)
    head.append(f" {sess.status}", style=col)
    layout["head"].update(Panel(head, border_style="grey30"))

    blocks = []
    if sess.error is not None:
        blocks.append(Panel(Text(sess.error, style="red"), title="error", border_style="red"))
    hint = {"ready": "hold the button and speak  ·  \"flash the light blue\" · \"allume la lumière verte\"",
            "self-test": "boot self-test: transcribing a built-in clip...",
            "recording": "listening...  release the button when you are done",
            "transcribing": "transcribing on the ESP32-S3..."}.get(sess.status)
    if sess.text is not None:
        t = Text()
        t.append("“", style="grey50")
        t.append(sess.text or "(nothing)", style="bold white")
        t.append("”", style="grey50")
        title = "[grey58]heard[/]" + (f" [bold cyan]{LANG_NAMES.get(sess.lang, sess.lang)}[/]"
                                      if sess.lang else "")
        blocks.append(Panel(Align.center(t, vertical="middle"), title=title,
                            border_style="cyan", height=5))
    elif hint:
        blocks.append(Align.center(Text(hint, style="grey62"), vertical="middle"))
    for old, act in sess.history:
        h = Text("  " + (old or "(nothing)"), style="grey42")
        if act and act.get("color"):
            h.append(f"   → {act['color']} {act.get('mode', '')}", style="grey35")
        blocks.append(h)
    layout["body"].update(Panel(Group(*blocks), border_style="grey30"))

    swatch = Text()
    a = sess.action or {}
    if a.get("color"):
        hexc = SWATCH.get(a["color"], "#888888")
        bar = "█" * 22
        swatch.append(bar + "\n", style=hexc)
        swatch.append(bar + "\n", style="grey15" if a.get("mode") == "flash" else hexc)
        swatch.append(bar + "\n", style=hexc)
        swatch.append(f"{a['color']} · {a.get('mode', '-')}", style="bold " + hexc)
        if a.get("duration"):
            swatch.append(f" · {a['duration']}s", style="grey62")
        if a.get("period"):
            swatch.append(f" · every {a['period']}s", style="grey62")
    elif sess.status == "nothing heard":
        swatch.append("\n  two red blinks", style="red")
    else:
        swatch.append("\n\n  (no action)", style="grey35")

    stats = Table.grid(padding=(0, 2))
    stats.add_column(style="grey58", justify="right")
    stats.add_column(style="bold white")
    def dur(ms):
        ms = float(ms)
        return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"

    if sess.time:
        stats.add_row("speech", dur(sess.time.get("audio_ms", 0)))
        stats.add_row("total", f"{dur(sess.time.get('total_ms', 0))}  "
                               f"({sess.time.get('tokens', '?')} tokens)")
    if sess.prof:
        stats.add_row("phases", "  ".join(f"{k} {dur(v)}" for k, v in sess.prof.items()
                                          if k in ("mel", "stem", "encoder", "cross_kv", "decoder")))
    if not sess.time and sess.model_info:
        m = _kv(sess.model_info)
        for k in ("depth", "enc_layers", "psram_free"):
            if k in m:
                stats.add_row(k, m[k])

    foot = Layout()
    foot.split_row(Layout(name="led", ratio=1), Layout(name="stat", ratio=2))
    foot["led"].update(Panel(swatch, title="[grey58]LED[/]", border_style="grey30"))
    foot["stat"].update(Panel(stats, title="[grey58]stats[/]", border_style="grey30"))
    layout["foot"].update(foot)
    return layout


def whistle_main(args):
    sess = WhistleSession()
    backend = proc = errors = None
    returncode = 0
    try:
        if args.serial:
            backend = SerialBackend(args.serial, assert_dtr=False)
            name, lines = backend.name, None
        else:
            cmd = [args.whistle_binary, args.whistle_model, args.local_wav, "--protocol",
                   "--gate", "12", "--lang", "auto", "--led-keywords"]
            name = "host " + args.local_wav.split("/")[-1]
            # The host also writes normal diagnostics to stderr. Keep them
            # separate from protocol lines and show them only on failure.
            # A file avoids filling a second pipe while we read stdout.
            errors = tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errors, text=True)
            lines = proc.stdout
        CONSOLE.clear()
        with Live(render_whistle(sess, name), console=CONSOLE, refresh_per_second=8,
                  screen=False) as live:
            while True:
                if lines is not None:
                    line = lines.readline()
                    if not line:
                        break
                    line = line.rstrip("\n")
                else:
                    try:
                        line = backend.q.get(timeout=0.25)
                    except queue.Empty:
                        continue
                sess.feed(line)
                live.update(render_whistle(sess, name))
            if proc is not None:
                returncode = proc.wait()
                if returncode:
                    errors.seek(0)
                    message = errors.read().strip() or f"host engine exited with status {returncode}"
                    sess.feed("ERR " + message)
                    returncode = max(1, returncode)  # signals also mean failure
                live.update(render_whistle(sess, name))
    except OSError as exc:
        sess.feed("ERR " + str(exc))
        CONSOLE.print(Panel(Text(sess.error, style="red"), title="error", border_style="red"))
        returncode = 1
    except KeyboardInterrupt:
        returncode = 130
    finally:
        if backend:
            backend.close()
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            proc.stdout.close()
        if errors is not None:
            errors.close()
    CONSOLE.print("\n[grey50]bye[/]")
    return returncode or int(sess.error is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--serial", metavar="PORT")
    ap.add_argument("--binary", default="./build/nd_dump")
    ap.add_argument("--model", default="model/needle2.cact")
    ap.add_argument("--tools", default="tools/led.json")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--no-think", action="store_true",
                    help="suppress the reasoning block (faster, slightly less accurate)")
    ap.add_argument("--whistle", action="store_true",
                    help="follow esp32/whistle_demo (push-to-talk) instead of Needle")
    ap.add_argument("--whistle-binary", default="./build/wh_run")
    ap.add_argument("--whistle-model", default="model/whistle/whistle_d4.cact")
    ap.add_argument("local_wav", nargs="?", help="with --whistle --local: a 16 kHz WAV")
    args = ap.parse_args()
    if args.whistle:
        if args.local and not args.local_wav:
            ap.error("--whistle --local needs a WAV file")
        if not args.serial and not args.local:
            ap.error("pass --serial PORT or --local clip.wav")
        return whistle_main(args)

    if args.serial:
        backend = SerialBackend(args.serial)
    elif args.local:
        backend = LocalBackend(args.binary, args.model, args.tools,
                               args.max_new, not args.no_think)
    else:
        ap.error("pass --local or --serial PORT")

    sess = Session(backend.name)
    sess.think = not args.no_think

    CONSOLE.clear()

    # If the board is mid-prime, show the progress bar until it is ready.
    if args.serial:
        with Live(render(sess), console=CONSOLE, refresh_per_second=8) as live:
            for line in backend.drain(1.0):
                sess.feed(line)
            live.update(render(sess))
            if sess.priming:
                for line in backend.lines(idle_timeout=30):
                    sess.feed(line)
                    live.update(render(sess))
                    if line.startswith("EVT ready") or "READY" in line:
                        break
        # Set reasoning explicitly rather than toggling: the board keeps its
        # state across client restarts.
        backend.send("!think 0" if args.no_think else "!think 1")
        for line in backend.lines(idle_timeout=5):
            sess.feed(line)
            if line.startswith("EVT think"):
                break

    try:
        while True:
            CONSOLE.clear()
            CONSOLE.print(render(sess))
            try:
                prompt = CONSOLE.input("\n[bold green]›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt or prompt in ("quit", "exit"):
                break

            turn = Session(backend.name)
            turn.think = sess.think
            turn.model_info = sess.model_info
            turn.prompt = prompt
            turn.status = "reading prompt"

            CONSOLE.clear()
            backend.send(prompt)
            with Live(render(turn), console=CONSOLE, refresh_per_second=12) as live:
                for line in backend.lines():
                    turn.feed(line)
                    live.update(render(turn))
                live.update(render(turn))
            sess = turn
    finally:
        backend.close()
    CONSOLE.print("\n[grey50]bye[/]")


if __name__ == "__main__":
    sys.exit(main())
